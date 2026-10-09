"""Offline execution-contract regressions; only external model/tool boundaries are faked."""
import ast
import asyncio
import contextlib
import dataclasses
import inspect
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
import types
import typing
import unittest
import uuid
from collections import Counter
from string import Template
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class Base:
    def __init__(self, name, config=None):
        self.name, self.config = name, config or {}
        self.evaluator_name = self.config.get("evaluator")
        self.format_prompt = "FINAL FORMAT"
        self.meta_memory = self.evaluator = None


class Usage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens, self.completion_tokens, self.total_tokens = prompt_tokens, completion_tokens, total_tokens


class Message:
    def __init__(self, content, **kwargs):
        self.content = content
        self.__dict__.update(kwargs)


class Client:
    instances = []
    def __init__(self, **kwargs):
        self.settings, self.calls, self.replies = kwargs, [], []
        self.chat = types.SimpleNamespace(completions=self)
        Client.instances.append(self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0) if self.replies else "APPROVE"
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=Message(reply))], usage=Usage(2, 1, 3))

    async def close(self):
        self.closed = True


class Bench:
    instances = []
    events = []
    def __init__(self, **kwargs):
        self.config, self.name, self.calls = kwargs, kwargs.get("name", "bench"), []
        Bench.instances.append(self)

    async def run_agent_step(self, augmented_question, additional_args):
        self.calls.append((augmented_question, additional_args))
        Bench.events.append(self.name)
        return {"final_answer": f"answer-{self.name}-{len(self.calls)}", "messages": [
            {"role": "assistant", "content": "answer", "usage_metadata": Usage(2, 1, 3)}]}

    async def run_agent(self, problem, **kwargs):
        return await self.run_agent_step(problem["problem"], {"id": problem.get("id")})

    async def aclose(self):
        self.closed = True

    def _extract_token_usage_from_agent(self):
        return Usage(2, 1, 3)


def load_source(filename, **overrides):
    """Load actual function/class bodies without importing optional API dependencies."""
    path = ROOT / "mas_arena" / "agents" / filename
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    module = types.ModuleType("offline_" + path.stem)
    sys.modules[module.__name__] = module
    module.__dict__.update(vars(typing))
    module.__dict__.update(dict(
        __name__="offline_" + path.stem, os=os, re=re, asyncio=asyncio, time=time, json=json,
        random=random, uuid=uuid, inspect=inspect, contextlib=contextlib, Counter=Counter,
        Template=Template, dataclass=dataclasses.dataclass, field=dataclasses.field,
        AgentSystem=Base, BenchAgent=Bench, Tool=object, CompletionUsage=Usage,
        AIMessage=Message, HumanMessage=Message, SystemMessage=Message,
        AsyncOpenAI=Client, logger=logging.getLogger(__name__), override=lambda fn: fn,
        DEFAULT_MODEL_NAME="gpt-4.1-2025-04-14", RetryWrapper=lambda model: model,
        get_model_name=lambda default="gpt-4.1-2025-04-14": os.getenv("MODEL_NAME", default),
        get_openai_api_base=lambda: "https://api.openai.com/v1",
        normalize_openai_api_base=lambda value, default: value or default,
        ALL_EXTERNAL_TOOLS={}, PythonInterpreterTool=object, FinalAnswerTool=object,
        WikipediaSearchTool=object, MultiStepAgent=object, OpenAIServerModel=object,
        CodeAgent=type("Code", (), {}), ToolCallingAgent=type("Search", (), {}),
        prompts={"build_search_keywords_prompt": "$question $true_answer"},
        AgentSystemRegistry=types.SimpleNamespace(register=lambda *args, **kwargs: None),
        print_step=lambda *args: None, print_agent_info=lambda *args: None,
        Colors=types.SimpleNamespace(**{name: "" for name in ["BLUE", "GREEN", "YELLOW", "RED", "CYAN", "ENDC", "BOLD"]}),
    ))
    helper = ROOT / "mas_arena" / "agents" / "workflow_protocol.py"
    if helper.exists() and filename != "workflow_protocol.py":
        module.__dict__.update({key: value for key, value in vars(load_source("workflow_protocol.py")).items()
                               if key not in {"__name__", "__file__", "__builtins__"}})
    module.__dict__.update(overrides)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


class CoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.module = load_source("bench_agent.py")
        cls = self.module.BenchAgent
        cls._initialize_model = lambda instance: None
        cls._initialize_tools = lambda instance: None
        cls._create_agents = lambda instance: {"workers": [
            types.SimpleNamespace(agent=self.module.CodeAgent()), types.SimpleNamespace(agent=self.module.ToolCallingAgent())]}

    def test_registry_config_and_explicit_overrides(self):
        agent = self.module.BenchAgent(config={"model_name": "registry", "max_steps": 7,
            "search_max_steps": 4, "api_key": "offline", "manager_tools": [], "search_tools": [], "evaluator": "hotpotqa"})
        self.assertEqual(agent.config["model_name"], "registry")
        self.assertEqual(agent.config["max_steps"], 7)
        self.assertEqual(agent.config["search_max_steps"], 4)
        self.assertEqual(agent.config["api_key"], "offline")
        self.assertEqual(agent.manager_tools_config, [])
        explicit = self.module.BenchAgent(model="explicit", max_steps=15, config={"model_name": "registry", "max_steps": 7})
        self.assertEqual((explicit.config["model_name"], explicit.config["max_steps"]), ("explicit", 15))

    def test_model_settings_reach_wrapped_client_and_explicit_model_wins(self):
        calls = []
        module = load_source("bench_agent.py", OpenAIServerModel=lambda **kwargs: calls.append(kwargs) or types.SimpleNamespace(**kwargs))
        agent = object.__new__(module.BenchAgent)
        agent.config = {"api_key": "offline", "model_name": "explicit"}
        with patch.dict(os.environ, {"MODEL_NAME": "environment"}):
            agent._initialize_model()
        self.assertEqual(len(calls), 1)
        self.assertEqual(agent.llm.model_id, "explicit")
        self.assertEqual((agent.llm.temperature, agent.llm.top_p, agent.llm.max_completion_tokens), (0.2, 1.0, 8192))

    async def test_core_never_reads_solution_during_inference(self):
        agent = self.module.BenchAgent(config={})
        seen = []
        async def run_step(question, additional_args, **kwargs):
            seen.append((question, additional_args))
            return {"final_answer": "prediction", "messages": []}
        agent.run_agent_step = run_step
        agent._usage_totals = lambda: (0, 0)
        result = await agent.run_agent({"problem": "question", "id": "one", "solution": "SECRET_GOLD"})
        self.assertEqual(result["final_answer"], "prediction")
        self.assertNotIn("SECRET_GOLD", str(seen))
        self.assertNotIn("score", result)

    def test_shared_monitor_counted_once_and_current_call_only(self):
        agent = self.module.BenchAgent(config={})
        monitor = types.SimpleNamespace(total_input_token_count=17, total_output_token_count=9)
        agent.workers = [types.SimpleNamespace(agent=types.SimpleNamespace(monitor=monitor)) for _ in range(2)]
        agent._usage_start = (10, 5)
        usage = agent._extract_token_usage_from_agent()
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.total_tokens), (7, 4, 11))

    async def test_step_strips_label_arguments_and_reports_usage_delta(self):
        agent = self.module.BenchAgent(config={})
        monitor = types.SimpleNamespace(total_input_token_count=10, total_output_token_count=5)
        agent.workers = [types.SimpleNamespace(agent=types.SimpleNamespace(monitor=monitor)) for _ in range(2)]
        seen = []
        def execute(question, args):
            seen.append((question, args))
            monitor.total_input_token_count += 2
            monitor.total_output_token_count += 1
            return "prediction"
        agent._run_agent_sync = execute
        agent._extract_agent_steps = lambda: ([], [])
        for _ in range(2):
            result = await agent.run_agent_step("question", {"id": "task", "expected_answer": "SECRET_GOLD"})
            self.assertEqual(result["messages"][-1]["usage_metadata"].total_tokens, 3)
        self.assertNotIn("SECRET_GOLD", str(seen))

    async def test_file_preprocessing_tokens_are_in_same_invocation(self):
        agent = self.module.BenchAgent(config={})
        monitor = types.SimpleNamespace(total_input_token_count=0, total_output_token_count=0)
        agent.workers = [types.SimpleNamespace(agent=types.SimpleNamespace(monitor=monitor))]
        def file_description(path, question):
            monitor.total_input_token_count += 7
            return "Description of attachment"
        agent._get_file_description = file_description
        agent._run_agent_sync = lambda question, args: "prediction"
        agent._extract_agent_steps = lambda: ([], [])
        result = await agent.run_agent({"problem": "question", "files": ["chart.png"]})
        self.assertEqual(result["messages"][-1]["usage_metadata"].total_tokens, 7)

    async def test_core_closes_sync_and_async_model_clients(self):
        agent = self.module.BenchAgent(config={})
        closed = []
        client = Client()
        agent.llm = types.SimpleNamespace(client=types.SimpleNamespace(close=lambda: closed.append("sync")), async_client=client)
        await agent.aclose()
        self.assertEqual(closed, ["sync"])
        self.assertTrue(client.closed)


class WrapperTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Bench.instances, Bench.events, Client.instances = [], [], []
        self.config = {"model_name": "shared", "api_key": "offline", "api_base": "https://offline.invalid/v1",
                       "evaluator": "hotpotqa", "max_steps": 7, "search_max_steps": 4,
                       "manager_tools": ["python_interpreter"], "search_tools": ["wikipedia"]}

    def test_all_role_workers_share_benchmark_config(self):
        for file, class_name in [("chateval_newcore.py", "ChatEvalNewCore"), ("llm_debate_newcore.py", "LLMDebate"),
                                 ("camel_newcore.py", "Camel"), ("autogen_newcore.py", "AutoGen")]:
            with self.subTest(workflow=class_name):
                Bench.instances = []
                getattr(load_source(file), class_name)(config=self.config)
                for worker in Bench.instances:
                    merged = {**worker.config.get("config", {}), **worker.config}
                    self.assertEqual(merged.get("max_steps"), 7)
                    self.assertEqual(merged.get("search_max_steps"), 4)
                    self.assertEqual(merged.get("manager_tools"), ["python_interpreter"])
                    self.assertEqual(merged.get("search_tools"), ["wikipedia"])
                    self.assertEqual(merged.get("evaluator"), "hotpotqa")

    async def test_chateval_two_rounds_no_reference_and_extractor_usage(self):
        module = load_source("chateval_newcore.py")
        agent = module.ChatEvalNewCore(config=self.config)
        self.assertTrue(hasattr(agent, "client"), "Extractor must retain API usage")
        agent.client.replies = ["final"]
        result = await agent.run_agent({"problem": "question", "solution": "SECRET_GOLD", "files": ["chart.png"]})
        self.assertEqual(len(Bench.events), 6)
        self.assertEqual(len(agent.client.calls), 1)
        self.assertEqual(result["messages"][-1]["usage_metadata"].total_tokens, 3)
        self.assertNotIn("SECRET_GOLD", str([worker.calls for worker in agent.agents]))
        self.assertIn("chart.png", str(agent.agents[0].calls))

    async def test_debate_alternates_and_later_roles_see_prior_turn(self):
        agent = load_source("llm_debate_newcore.py").LLMDebate(config=self.config)
        result = await agent.run_agent({"problem": "question", "expected_answer": "SECRET_GOLD"})
        self.assertEqual(Bench.events, ["debate_agent_1", "debate_agent_2"] * 3 + ["aggregator_bench_agent"])
        self.assertIn("answer-debate_agent_1-1", agent.debate_agents[1].calls[0][0])
        self.assertNotIn("SECRET_GOLD", str([worker.calls for worker in Bench.instances]))
        self.assertEqual(result["rounds_completed"], 3)

    async def test_autogen_approval_does_not_skip_fixed_rounds(self):
        agent = load_source("autogen_newcore.py").AutoGen(config=self.config)
        result = await agent.run_agent({"problem": "question"})
        self.assertEqual(len(agent.openai_client.calls), 5)
        self.assertEqual(len(agent.bench_agent.calls), 5)
        self.assertEqual(result["final_answer"], "answer-bench-5")
        self.assertEqual(agent.openai_client.calls[0]["temperature"], 0.7)
        self.assertEqual(agent.openai_client.calls[0]["max_tokens"], 8192)

    async def test_autogen_critic_failure_retains_usage_and_marks_error(self):
        agent = load_source("autogen_newcore.py").AutoGen(config=self.config)

        async def fail_critic(**kwargs):
            raise RuntimeError("offline critic failure")

        agent.openai_client.create = fail_critic
        result = await agent.run_agent({"problem": "question"})
        self.assertIn("offline critic failure", result["error"])
        self.assertEqual(result["final_answer"], "answer-bench-1")
        self.assertEqual(result["messages"][0]["usage_metadata"].total_tokens, 3)
        self.assertEqual(len(agent.bench_agent.calls), 1)

    async def test_autogen_primary_error_does_not_call_critic(self):
        agent = load_source("autogen_newcore.py").AutoGen(config=self.config)

        async def fail_primary(**kwargs):
            return {"final_answer": "partial", "error": "offline primary failure", "messages": [
                {"usage_metadata": Usage(2, 1, 3)}]}

        agent.bench_agent.run_agent_step = fail_primary
        result = await agent.run_agent({"problem": "question"})
        self.assertIn("offline primary failure", result["error"])
        self.assertEqual(result["messages"][0]["usage_metadata"].total_tokens, 3)
        self.assertEqual(agent.openai_client.calls, [])

    def test_explicit_request_cap_ignores_invalid_environment_cap(self):
        module = load_source("workflow_protocol.py")
        with patch.dict(os.environ, {"MAX_TOKEN_SIZE": "invalid"}):
            self.assertEqual(module.request_settings({"max_completion_tokens": 8192})["max_tokens"], 8192)

    async def test_camel_fixed_cap_and_case_sensitive_termination(self):
        agent = load_source("camel_newcore.py").Camel(config=self.config)
        agent.client.replies = ["task_finished", "first", "TASK_FINISHED"]
        result = await agent.run_agent({"problem": "question"})
        self.assertEqual(len(agent.client.calls), 3)
        self.assertEqual(len(agent.bench_agent.calls), 1)
        self.assertEqual(agent.client.calls[0]["temperature"], 0.2)
        self.assertEqual(agent.max_rounds, 3)

    async def test_wrappers_close_their_workers_and_direct_clients(self):
        for filename, classname, client_name in [("autogen_newcore.py", "AutoGen", "openai_client"),
                                                  ("camel_newcore.py", "Camel", "client"),
                                                  ("chateval_newcore.py", "ChatEvalNewCore", "client")]:
            with self.subTest(workflow=classname):
                Bench.instances = []
                agent = getattr(load_source(filename), classname)(config=self.config)
                self.assertTrue(hasattr(agent, "aclose"), "Runner must be able to close all workflow clients")
                await agent.aclose()
                self.assertTrue(all(worker.closed for worker in Bench.instances))
                self.assertTrue(getattr(agent, client_name).closed)


class EvoTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        Bench.instances, Bench.events = [], []

    @staticmethod
    def replies():
        return [json.dumps({"name": f"EVO-C-{index}", "system_prompt": f"crossover-{index}"}) for index in range(5)] + [
            json.dumps({"name": f"EVO-M-{index}", "system_prompt": f"mutation-{index}"}) for index in range(8)] + ["FINAL"]

    async def test_answer_agreement_stable_ties_and_missing_zero(self):
        module = load_source("evoagent_newcore.py")
        agent = object.__new__(module.EvoAgent)
        self.assertTrue(hasattr(agent, "_rank_candidates"), "Selection requires reference-free candidate agreement")
        population = [types.SimpleNamespace(score=99, result={"extracted_answer": answer}) for answer in ["B", " A ", "A", ""]]
        original = list(population)
        agent._rank_candidates(population)
        self.assertEqual(population, [original[1], original[2], original[0], original[3]])
        self.assertEqual(original[3].score, 0)
        self.assertEqual(original[1].score, original[2].score)
        self.assertEqual(original[1].score, 2 / 3)

    def test_agreement_only_strips_surrounding_whitespace(self):
        agent = object.__new__(load_source("evoagent_newcore.py").EvoAgent)
        candidates = [types.SimpleNamespace(result={"extracted_answer": answer}) for answer in ["A", "a", "a  b", "a b"]]
        agent._rank_candidates(candidates)
        self.assertEqual([candidate.score for candidate in candidates], [0.25] * 4)

    def test_evo_candidates_inherit_entire_substrate(self):
        module = load_source("evoagent_newcore.py")
        agent = module.EvoAgent(config={"model_name": "shared", "api_key": "offline", "evaluator": "hotpotqa",
            "max_steps": 7, "search_max_steps": 4, "manager_tools": [], "search_tools": ["wikipedia"]})
        workers = agent._initialize_base_agents()
        self.assertEqual(len(workers), 3)
        for worker in workers:
            settings = {**worker.bench_agent.config.get("config", {}), **worker.bench_agent.config}
            self.assertEqual(settings.get("max_steps"), 7)
            self.assertEqual(settings.get("manager_tools"), [])
            self.assertEqual(settings.get("search_tools"), ["wikipedia"])

    async def test_full_schedule_counts_each_call_and_resets_every_task(self):
        agent = load_source("evoagent_newcore.py").EvoAgent(config={"model_name": "shared", "seed": 42})
        for task_id in ["FIRST_QUESTION", "SECOND_QUESTION"]:
            agent.client.replies = self.replies()
            with contextlib.redirect_stdout(io.StringIO()):
                result = await agent.run_agent({"problem": task_id, "id": task_id, "files": ["data.csv"], "solution": "SECRET_GOLD"})
            metrics = result["evolution_metrics"]
            self.assertEqual([metrics[key] for key in ["initial_agents", "crossover_agents", "mutation_agents", "final_agents"]], [3, 6, 9, 5])
            self.assertEqual(len(agent._candidates), 16)
            self.assertTrue(all(len(candidate.bench_agent.calls) == 1 for candidate in agent._candidates))
            usage = 0
            for message in result["messages"]:
                data = message.get("usage_metadata") if isinstance(message, dict) else getattr(message, "usage_metadata", None)
                if data:
                    usage += data.get("total_tokens", 0) if isinstance(data, dict) else data.total_tokens
            self.assertEqual(usage, 90)  # 16 workers + 13 prompt variants + 1 aggregation, each 3 tokens.
            summary_prompt = agent.client.calls[-1]["messages"][0]["content"]
            self.assertIn("BenchWorker-EVO-1", summary_prompt)
            self.assertIn("BenchWorker-EVO-M-3", summary_prompt)
            self.assertNotIn("BenchWorker-EVO-M-4", summary_prompt)
            self.assertNotIn("SECRET_GOLD", str(agent.client.calls) + str([c.bench_agent.calls for c in agent._candidates]))
        self.assertNotIn("FIRST_QUESTION", str(agent._candidates[-1].bench_agent.calls))
        self.assertTrue(all(worker.closed for worker in Bench.instances[:16]))

    async def test_timeouts_retain_recorded_usage_and_are_not_aggregated(self):
        module = load_source("evoagent_newcore.py")
        async def slow_solve(candidate, task):
            await asyncio.sleep(1)
        module.BenchEnhancedAgent.solve = slow_solve
        agent = module.EvoAgent(config={"agent_task_timeout_seconds": 0.001})
        agent.client.replies = self.replies()
        with contextlib.redirect_stdout(io.StringIO()):
            result = await agent.run_agent({"problem": "question"})
        self.assertIn("error", result)
        self.assertEqual(len(agent.client.calls), 13)
        self.assertEqual(sum(message["usage_metadata"]["total_tokens"] for message in result["messages"]), 87)


if __name__ == "__main__":
    unittest.main()
