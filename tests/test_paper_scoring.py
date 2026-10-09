"""Offline regressions for the paper's scoring and recorded-usage contract.

Load source definitions without importing API clients or optional evaluator packages.
The scoring, runner, and accounting methods themselves run unchanged.
"""

import abc
import ast
import asyncio
import collections
import contextlib
from decimal import Decimal, ROUND_HALF_UP
import io
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import re
import random
import tempfile
import time
import types
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[1]


def definitions(relative_path, **overrides):
    source = ROOT / relative_path
    tree = ast.parse(source.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
            body.append(node)
    namespace = dict(abc=abc, asyncio=asyncio, collections=collections, json=json,
                     logging=logging, os=os, Path=Path, re=re, time=time,
                     logger=logging.getLogger(__name__), BaseEvaluator=object,
                     isclose=math.isclose, isfinite=math.isfinite,
                     Decimal=Decimal, ROUND_HALF_UP=ROUND_HALF_UP,
                     CompletionUsage=type("CompletionUsage", (), {}),
                     custom_json_serializer=str, rprint=print)
    namespace.update(overrides)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace


class RunnerScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ns = definitions("mas_arena/benchmark_runner.py")
        self.runner = self.ns["BenchmarkRunner"].__new__(self.ns["BenchmarkRunner"])
        self.runner.timestamp = "offline-test"
        self.runner.metrics_registry = Mock()
        self.runner.metrics_collector = Mock()
        self.runner._live_total = 0

    def summarize(self, results):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.runner._finalize_benchmark(
                results, "math", "offline", Path(self.directory.name) / "results.json", False)

    def test_errors_and_timeouts_remain_in_accuracy_denominator(self):
        summary = self.summarize([
            {"score": 1, "status": "success"}, {"score": 0, "status": "success"},
            {"score": 0, "status": "error"}, {"score": 0, "status": "timeout"},
        ])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["total_problems"], 4)
        self.assertEqual(summary["accuracy"], 0.25)

    def test_error_cannot_retain_stale_correct_score(self):
        summary = self.summarize([{"score": 1, "is_correct": True, "status": "error"}])
        self.assertEqual(summary["correct"], 0)
        result = json.loads((Path(self.directory.name) / "results.json").read_text())["results"][0]
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["is_correct"])

    def test_usage_includes_failed_attempts_and_reports_missing_coverage(self):
        summary = self.summarize([
            {"score": 1, "llm_usage": {"total_tokens": 10}},
            {"score": 0, "status": "error", "llm_usage": {"total_tokens": 31}},
            {"score": 0, "llm_usage": {}},
            {"score": 0, "llm_usage": {"total_tokens": 0, "message_count": 0, "agent_usage": []}},
        ])
        self.assertEqual(summary.get("token_usage_observed_problems"), 2)
        self.assertEqual(summary.get("token_usage_missing_problems"), 2)
        self.assertEqual(summary.get("token_usage_coverage"), 0.5)
        self.assertEqual(summary.get("avg_tokens_observed"), 20.5)
        self.assertEqual(summary.get("rounded_avg_tokens_per_run"), 21)
        self.assertEqual(summary.get("total_recorded_tokens"), 41)

    def test_absent_usage_is_not_zero(self):
        summary = self.summarize([{"score": 0, "status": "error"}])
        self.assertIsNone(summary.get("avg_tokens_observed", "missing"))
        self.assertIsNone(summary.get("avg_input_tokens", "missing"))
        self.assertEqual(summary.get("token_usage_coverage"), 0)

    def test_observed_zero_usage_is_retained(self):
        summary = self.summarize([{"score": 0, "llm_usage": {
            "total_tokens": 0, "message_count": 1,
            "agent_usage": [{"prompt_tokens": 0, "completion_tokens": 0}],
        }}])
        self.assertEqual(summary.get("token_usage_observed_problems"), 1)
        self.assertEqual(summary.get("avg_tokens_observed"), 0)

    def test_component_means_include_errors_without_fabricating_components(self):
        summary = self.summarize([
            {"score": 1, "llm_usage": {"total_tokens": 15, "agent_usage": [
                {"prompt_tokens": 10, "completion_tokens": 5}]}},
            {"score": 0, "status": "error", "llm_usage": {"total_tokens": 30,
                "agent_usage": [{"prompt_tokens": 20, "completion_tokens": 10}]}},
            {"score": 0, "llm_usage": {"total_tokens": 90}},
        ])
        self.assertEqual(summary["avg_input_tokens"], 15)
        self.assertEqual(summary["avg_output_tokens"], 7.5)

    def test_planned_denominator_is_retained_when_result_is_missing(self):
        summary = self.runner._finalize_benchmark(
            [{"score": 1}], "math", "offline", Path(self.directory.name) / "results.json", False,
            expected_total=3)
        self.assertEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["missing_results"], 2)

    def test_initialization_failure_does_not_abort_other_tasks(self):
        async def evaluate(problem, **kwargs):
            return types.SimpleNamespace(raw_responses={"score": 1, "final_answer": "42"})

        count = 0

        def create_agent(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise ValueError("offline initialization failure")
            return types.SimpleNamespace(name="offline", evaluate=evaluate, set_metrics_registry=lambda r: None)

        async def gather(*tasks, **kwargs):
            return await asyncio.gather(*tasks)

        self.ns.update(create_agent_system=create_agent, normalize_problem_keys=lambda p, keys, i: p,
                       tqdm=types.SimpleNamespace(gather=gather))
        self.runner._setup_logging = lambda path: None
        self.runner._prepare_benchmark = lambda *args: ([{"id": str(i), "problem": "question", "solution": "42"}
                                                        for i in range(3)], {},
                                                       Path(self.directory.name) / "results.json")
        self.runner.agent_config = {}
        self.runner.problem_timeout_seconds = None
        self.runner.metrics_collector.stop_timer.return_value = 1
        with contextlib.redirect_stdout(io.StringIO()):
            summary = asyncio.run(self.runner.arun(verbose=False))
        self.assertEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(summary["errored"], 1)

    def test_pass_at_k_unwraps_agent_results_and_retains_all_usage(self):
        async def evaluate(problem, **kwargs):
            return types.SimpleNamespace(raw_responses={"score": 1, "llm_usage": {"total_tokens": 8}})

        self.ns.update(create_agent_system=lambda *a, **k: types.SimpleNamespace(
            name="offline", evaluate=evaluate, set_metrics_registry=lambda r: None),
            normalize_problem_keys=lambda p, keys, i: p)
        self.runner.problem_timeout_seconds = None
        self.runner._live_result_lock = None
        self.runner._live_completed = self.runner._live_correct = 0
        self.runner.metrics_collector.stop_timer.return_value = 1
        result = asyncio.run(self.runner._process_one_problem_pass_at_k(
            0, {"id": "q", "problem": "question", "solution": "42"}, "offline", {}, {}, False, 2))
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["llm_usage"]["total_tokens"], 16)

    def test_runner_preserves_full_final_answer_and_numeric_zero(self):
        self.ns["normalize_problem_keys"] = lambda p, keys, i: p
        self.runner.problem_timeout_seconds = None
        self.runner._live_result_lock = None
        self.runner._live_completed = self.runner._live_correct = 0
        self.runner.metrics_collector.stop_timer.return_value = 1
        for final_answer, extracted in (("A full answer " * 20, "A full answer ..."), (0, 0)):
            async def evaluate(problem, **kwargs):
                return types.SimpleNamespace(raw_responses={"score": 1, "final_answer": final_answer,
                                                            "extracted_answer": extracted, "f1": 0.5})

            with self.subTest(final_answer=final_answer):
                result = asyncio.run(self.runner._process_one_problem(
                    0, {"id": "q", "problem": "question", "solution": "answer"},
                    types.SimpleNamespace(name="offline", evaluate=evaluate), {}, False))
                self.assertEqual(result["prediction"], final_answer)
                self.assertEqual(result.get("extracted_answer"), extracted)
                self.assertEqual(result.get("f1"), 0.5)

    def test_selected_data_path_reaches_evaluator_without_resetting_rng(self):
        source = Path(self.directory.name) / "custom.jsonl"
        source.write_text("\n".join(json.dumps({"id": i}) for i in range(10)))
        self.ns.update(random=random, BENCHMARKS={"math": {}}, AVAILABLE_AGENT_SYSTEMS={})
        self.runner.results_dir = self.directory.name
        before = random.getstate()
        problems, _, _ = self.runner._prepare_benchmark(
            "math", str(source), 4, "offline", {}, False, None, None)
        self.assertEqual(self.runner.agent_config.get("data_path"), str(source))
        self.assertEqual([p["id"] for p in problems], random.Random(42).sample(list(range(10)), 4))
        self.assertEqual(random.getstate(), before)


class EvaluatorScoringTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.metrics = definitions("mas_arena/evaluators/utils/metrics.py", Counter=collections.Counter,
                                   string=__import__("string"))
        self.extract = definitions("mas_arena/evaluators/utils/answer_extraction.py")

    def evaluator(self, name, class_name):
        ns = definitions(f"mas_arena/evaluators/{name}_evaluator.py", **{
            **self.metrics, **self.extract,
            "_ANS_TAG_RE": re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.I),
            "_FINAL_RE": re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-]?\s*([\s\S]+)", re.I),
        })
        evaluator = ns[class_name].__new__(ns[class_name])
        evaluator.log_path = self.directory.name
        evaluator.name = name
        evaluator.run_evaluator = Mock()
        evaluator._make_run = Mock()
        return evaluator

    def test_qa_threshold_is_binary_and_inclusive(self):
        for name, class_name in (("drop", "DROPEvaluator"), ("hotpotqa", "HotpotQAEvaluator")):
            evaluator = self.evaluator(name, class_name)
            for answer, expected in (("red one two three four five six", 0),
                                     ("red one two three four five", 0),
                                     ("red blue green four five six seven eight nine ten", 1),
                                     ("", 0)):
                # 3 common tokens out of 10+10 gives F1 exactly 0.3.
                reference = "red blue green alpha beta gamma delta epsilon zeta eta"
                with self.subTest(benchmark=name, answer=answer):
                    result = asyncio.run(evaluator.evaluate(
                        {"id": "q", "problem": "question", "solution": reference, "context": []},
                        {"final_answer": answer}))
                    self.assertEqual(result["score"], expected)
                    self.assertEqual(result.get("is_correct"), bool(expected))
                    self.assertIn("f1", result)

    def test_drop_uses_best_pipe_separated_pair(self):
        evaluator = self.evaluator("drop", "DROPEvaluator")
        result = asyncio.run(evaluator.evaluate(
            {"id": "q", "problem": "question", "solution": "unrelated | blue river"},
            {"final_answer": "wrong | blue river"}))
        self.assertEqual(result["score"], 1)

    def math_evaluator(self):
        ns = definitions("mas_arena/evaluators/math_evaluator.py")
        evaluator = ns["MathEvaluator"].__new__(ns["MathEvaluator"])
        evaluator.evaluate_type = 0
        return evaluator, ns

    def test_math_tolerance_is_absolute_only(self):
        evaluator, _ = self.math_evaluator()
        self.assertFalse(evaluator.math_equal("1000000000001", "1000000000000"))
        self.assertTrue(evaluator.math_equal("1.0005", "1"))
        self.assertFalse(evaluator.math_equal("1.002", "1"))

    def test_math_blank_answers_receive_zero_credit(self):
        evaluator, _ = self.math_evaluator()
        self.assertEqual(evaluator.simple_calculate_score("", "")[0], 0)

    def test_math_and_aime_score_explicit_final_answer(self):
        math_evaluator, math_ns = self.math_evaluator()
        aime_ns = definitions("mas_arena/evaluators/aime_evaluator.py", MathEvaluator=math_ns["MathEvaluator"])
        aime = aime_ns["AIMEEvaluator"].__new__(aime_ns["AIMEEvaluator"])
        aime.evaluate_type = 0
        for evaluator in (math_evaluator, aime):
            for final_answer, expected in (("42", 1), ("", 0)):
                with self.subTest(evaluator=type(evaluator).__name__, answer=final_answer):
                    result = asyncio.run(evaluator.evaluate(
                        {"solution": "42"}, {"messages": [("assistant", "42" if not final_answer else "13")],
                                               "final_answer": final_answer}))
                    self.assertEqual(result["score"], expected)

    def test_bbh_option_normalization_is_case_insensitive(self):
        evaluator = self.evaluator("bbh", "BBHEvaluator")
        for prediction in ("a", "[A]", "[a]", "(a)"):
            with self.subTest(prediction=prediction):
                self.assertEqual(evaluator.calculate_score("(A)", prediction, "choice_1")[0], 1)

    def test_bbh_word_sorting_rejects_incorrect_order(self):
        evaluator = self.evaluator("bbh", "BBHEvaluator")
        for prediction in ("banana apple cherry", "cherry banana apple"):
            with self.subTest(prediction=prediction):
                score, extracted, message = evaluator.calculate_score(
                    "apple banana cherry", f"<answer>{prediction}</answer>", "word_sorting_1")
                self.assertEqual(score, 0)
                self.assertEqual(extracted, prediction)
                self.assertIn("Incorrect word sorting", message)

    def test_bbh_word_sorting_preserves_duplicate_counts(self):
        evaluator = self.evaluator("bbh", "BBHEvaluator")
        for reference, prediction in (("apple apple banana", "apple banana"),
                                      ("apple banana", "apple apple banana")):
            with self.subTest(reference=reference, prediction=prediction):
                self.assertEqual(evaluator.calculate_score(reference, prediction, "word_sorting_2")[0], 0)

    def test_bbh_word_sorting_accepts_correct_sequence_with_formatting(self):
        evaluator = self.evaluator("bbh", "BBHEvaluator")
        for prediction in ("apple apple banana", "<answer> Apple\tapple   BANANA </answer>"):
            with self.subTest(prediction=prediction):
                self.assertEqual(evaluator.calculate_score(
                    "apple apple banana", prediction, "word_sorting_3")[0], 1)

    def test_hotpotqa_keeps_capitalized_names_in_final_answer(self):
        evaluator = self.evaluator("hotpotqa", "HotpotQAEvaluator")
        for reference in ("NASA", "True Detective"):
            with self.subTest(reference=reference):
                self.assertEqual(evaluator.calculate_score(reference, reference)[0], 1)

    def test_ifeval_requires_a_complete_nonempty_instruction_set(self):
        ns = definitions("mas_arena/evaluators/ifeval_evaluator.py")
        aggregate = ns["IFEvalEvaluator"]._aggregate_metrics
        for ids, flags in (([], []), (["test:one", "test:two"], [True]),
                           (["test:one", "test:two"], [True, False])):
            with self.subTest(ids=ids, flags=flags):
                result = aggregate(types.SimpleNamespace(instruction_id_list=ids, follow_instruction_list=flags))
                self.assertFalse(result["prompt_followed"])
        result = aggregate(types.SimpleNamespace(instruction_id_list=["test:one"], follow_instruction_list=[True]))
        self.assertTrue(result["prompt_followed"])

    def test_ifeval_strict_receives_complete_unmodified_answer(self):
        captured = []
        def check(inp, mapping):
            captured.append(mapping[inp.prompt])
            return types.SimpleNamespace(instruction_id_list=["test:exact"], follow_instruction_list=[True])
        ns = definitions("mas_arena/evaluators/ifeval_evaluator.py", InputExample=types.SimpleNamespace,
                         test_instruction_following_strict=check, test_instruction_following_loose=check)
        evaluator = ns["IFEvalEvaluator"].__new__(ns["IFEvalEvaluator"])
        answer = "  " + "content " * 30 + "\n"
        result = asyncio.run(evaluator.evaluate(
            {"problem": "task", "instruction_id_list": ["test:exact"], "kwargs": [{}]},
            {"final_answer": answer}))
        self.assertEqual(captured[0], answer)
        self.assertEqual(result["extracted_answer"], answer)

    def test_mbpp_supplies_test_imports_to_the_test_environment(self):
        code_ns = definitions("mas_arena/evaluators/base_code_evaluator.py")
        mbpp_ns = definitions("mas_arena/evaluators/mbpp_evaluator.py", BaseCodeEvaluator=code_ns["BaseCodeEvaluator"],
                              sanitize=lambda code, entrypoint: code,
                              run_with_timeout=lambda fn, args=(), timeout=None: fn(*args))
        evaluator = mbpp_ns["MBPPEvaluator"].__new__(mbpp_ns["MBPPEvaluator"])
        evaluator.config = {}
        evaluator.create_run = Mock()
        evaluator.run_evaluator = Mock()
        evaluator.save_results = Mock()
        result = asyncio.run(evaluator.evaluate({"id": "q", "problem": "question", "entry_point": "f",
            "test": "def check():\n    assert f() == math.sqrt(4)", "test_imports": ["import math"]},
            {"final_answer": "def f():\n    return 2"}))
        self.assertEqual(result["score"], 1)

    def test_gaia_rejects_non_boolean_judgment(self):
        ns = definitions("mas_arena/evaluators/gaia_evaluator.py", TypedDict=TypedDict, lru_cache=lru_cache)
        evaluator = ns["GaiaEvaluator"].__new__(ns["GaiaEvaluator"])
        ns.update(_get_eval_model_name=lambda: "gpt-4o-mini")

        async def judge(**kwargs):
            return {"is_correct": "false", "confidence": 1, "reasoning": "incorrect"}

        evaluator._evaluate_with_openai = judge
        from unittest.mock import patch
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-not-a-real-key"}):
            result = asyncio.run(evaluator.evaluate({"problem": "question", "solution": "42"},
                                                     run_result={"final_answer": "13"}))
        self.assertEqual(result["score"], 0)

    def test_math_uses_selected_file_without_hidden_resampling(self):
        evaluator, _ = self.math_evaluator()
        evaluator.name = "math"
        evaluator.data_path = str(Path(self.directory.name) / "selected.jsonl")
        records = [{"id": i} for i in range(7)]
        def load(path):
            self.assertEqual(path, evaluator.data_path)
            return records
        evaluator._load_dateset_from_path = load
        evaluator._load_data()
        self.assertEqual(evaluator._test_data, records)

    def test_code_evaluator_does_not_require_unused_auxiliary_files(self):
        ns = definitions("mas_arena/evaluators/base_code_evaluator.py")
        evaluator = ns["BaseCodeEvaluator"].__new__(ns["BaseCodeEvaluator"])
        evaluator.name = "custom"
        evaluator.data_path = str(Path(self.directory.name) / "selected.jsonl")
        records = [{"id": 1}]
        def load(path):
            self.assertEqual(path, evaluator.data_path)
            return records
        evaluator._load_dateset_from_path = load
        evaluator._load_data()
        self.assertEqual(evaluator._test_data, records)
        self.assertEqual(evaluator._dev_data, [])
        self.assertEqual(evaluator._test_cases, [])

    def test_base_evaluator_accepts_omitted_optional_config(self):
        ns = definitions("mas_arena/evaluators/base_evaluator.py", ABCMeta=abc.ABCMeta)
        with patch.object(os, "makedirs"), patch.object(logging, "FileHandler"), \
                patch.object(logging, "basicConfig"):
            evaluator = ns["BaseEvaluator"]("offline")
        self.assertEqual(evaluator.data_path, "data/offline_test.jsonl")


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.ns = definitions("mas_arena/agents/base.py", AgentResult=types.SimpleNamespace,
                              public_task=lambda p: {k: v for k, v in p.items() if k in {"id", "problem", "files"}})
        base = self.ns["AgentSystem"]
        base.__abstractmethods__ = frozenset()
        self.agent = base.__new__(base)
        self.agent.name = "offline"
        self.agent.metrics_collector = Mock()

    def test_dictionary_messages_without_names_retain_usage(self):
        result = self.agent._record_token_usage("q", 10, [{"usage_metadata": {
            "input_tokens": 5, "output_tokens": 3, "total_tokens": 8}}])
        self.assertEqual(result["total_tokens"], 8)
        self.assertEqual(result["message_count"], 1)

    def test_absent_metadata_does_not_claim_zero_usage(self):
        self.assertEqual(self.agent._record_token_usage("q", 10, [{"content": "answer"}]), {})

    def test_usage_does_not_depend_on_metrics_collector(self):
        self.agent.metrics_collector = None
        result = self.agent._record_token_usage("q", 10, [types.SimpleNamespace(name="agent", usage_metadata={
            "input_tokens": 5, "output_tokens": 3, "total_tokens": 8})])
        self.assertEqual(result.get("total_tokens"), 8)

    def test_incomplete_metadata_does_not_fabricate_token_counts(self):
        result = self.agent._record_token_usage("q", 10, [
            {"usage_metadata": {"model": "unknown"}},
            {"usage_metadata": {"total_tokens": 8}},
        ])
        self.assertEqual(result["message_count"], 1)
        self.assertEqual(result["total_tokens"], 8)
        self.assertIsNone(result["agent_usage"][0].get("prompt_tokens"))
        self.assertIsNone(result["agent_usage"][0].get("completion_tokens"))

    def test_evaluator_failure_preserves_recorded_usage_and_messages(self):
        async def run_agent(problem, **kwargs):
            return {"final_answer": "42", "messages": [types.SimpleNamespace(name="solver", usage_metadata={
                "input_tokens": 5, "output_tokens": 3, "total_tokens": 8})]}

        async def fail_evaluation(**kwargs):
            raise ValueError("offline evaluation failure")

        async def no_file(*args):
            return None

        self.agent.evaluator_name = "math"
        self.agent.evaluator = types.SimpleNamespace(name="math", evaluate=fail_evaluation)
        self.agent.metrics_registry = None
        self.agent.meta_memory = None
        self.agent._initialize_evaluator = Mock()
        self.agent._initialize_metrics_collector = Mock()
        self.agent.generate_run_id = lambda: "offline"
        self.agent.run_agent = run_agent
        self.agent._record_agent_responses = Mock()
        self.agent.save_agent_responses = no_file
        self.agent.save_visualization_data = no_file
        with contextlib.redirect_stdout(io.StringIO()):
            result = asyncio.run(self.agent.evaluate({"id": "q", "problem": "question", "solution": "42"}))
        self.assertEqual(result.raw_responses["score"], 0)
        self.assertEqual(result.raw_responses["llm_usage"].get("total_tokens"), 8)
        self.assertEqual(len(result.raw_responses["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
