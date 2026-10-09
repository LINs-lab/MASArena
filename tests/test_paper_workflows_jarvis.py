"""Offline workflow tests; fake only model/BenchAgent boundaries, never call APIs."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from test_paper_workflows import load_source
from unittest.mock import patch

SOURCE = Path(os.environ.get('JARVIS_SOURCE', Path(__file__).resolve().parents[1] / 'mas_arena/agents/jarvis_newcore.py'))


class FakeBase:
    def __init__(self, name, config):
        self.name, self.config = name, config
        self.format_prompt = 'FINAL FORMAT'


class FakeModel:
    def __init__(self, **kwargs):
        self.model_id = kwargs['model_id']
        self.monitor = types.SimpleNamespace(total_input_token_count=11, total_output_token_count=5)
        self.client = types.SimpleNamespace(close=lambda: None)
        self.async_client = FakeClient()


class FakeBench:
    instances = []
    active = 0
    peak = 0
    fail_text = None

    def __init__(self, **config):
        self.config = {'model_name': config.get('model'), **config}
        self.closed = False
        self.calls = []
        self.__class__.instances.append(self)
        if hasattr(self, '_initialize_model'):
            self._initialize_model()

    async def run_agent(self, problem, **kwargs):
        return {'final_answer': 'OLD SINGLE AGENT'}

    def _usage_totals(self):
        return (0, 0)

    async def run_agent_step(self, augmented_question, additional_args, **kwargs):
        self.calls.append((augmented_question, additional_args))
        FakeBench.active += 1
        FakeBench.peak = max(FakeBench.peak, FakeBench.active)
        try:
            await asyncio.sleep(0.01)
            if FakeBench.fail_text and FakeBench.fail_text in augmented_question:
                return {'error': 'executor failed', 'messages': [{'content': 'failure trace'}]}
            return {'final_answer': 'RESOURCE_VALUE',
                    'messages': [{'content': 'executor trace', 'usage_metadata': {'total_tokens': 3}}],
                    'manager_agent_steps': [{'step': 1}], 'search_agent_steps': []}
        finally:
            FakeBench.active -= 1

    def _get_file_description(self, path, question):
        return f"Attachment description: {path}"

    async def aclose(self):
        self.closed = True


class FakeClient:
    def __init__(self, **kwargs):
        self.replies = []
        self.calls = []
        self.closed = False
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        calls = None
        if kwargs.get('tools'):
            calls = [types.SimpleNamespace(function=types.SimpleNamespace(
                name='final_answer', arguments=json.dumps({'answer': content})))]
            content = None
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content, tool_calls=calls))],
            usage=types.SimpleNamespace(total_tokens=7))

    async def close(self):
        self.closed = True


def load_module():
    modules = {}
    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        modules[name] = mod
    module('openai', AsyncOpenAI=FakeClient)
    module('mas_arena.tools.final_answer', FinalAnswerTool=type('FinalAnswerTool', (), {
        'name': 'final_answer', 'description': 'Submit final answer', 'inputs': {'answer': {'type': 'string'}},
        'forward': lambda self, answer: answer,
    }))
    modules['mas_arena.agents.workflow_protocol'] = load_source('workflow_protocol.py')
    module('mas_arena.agents.base', AgentSystem=FakeBase,
           AgentSystemRegistry=types.SimpleNamespace(register=lambda *args: None))
    module('mas_arena.agents.bench_agent', BenchAgent=FakeBench)
    module('mas_arena.agents.agent_core', OpenAIServerModel=FakeModel)
    module('mas_arena.utils.llm_utils', RetryWrapper=lambda model: model)
    module('mas_arena.utils.env', get_openai_api_base=lambda: None)
    module('mas_arena.utils.openai_compat', normalize_openai_api_base=lambda value, default: value or default)
    spec = importlib.util.spec_from_file_location('jarvis_under_test', SOURCE)
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(loaded)
    return loaded


JARVIS = load_module()


def task(task_id, dep=None, text='subtask'):
    return {'id': task_id, 'task': 'text-generation', 'dep': dep or [-1], 'args': {'text': text}}


class JarvisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeBench.instances = []
        FakeBench.active = FakeBench.peak = 0
        FakeBench.fail_text = None

    def agent(self, tasks, **config):
        agent = JARVIS.JarvisAgent(config={'api_key': 'offline-test', 'model_name': 'shared', **config})
        self.assertTrue(hasattr(agent, 'client'), 'Missing four-stage controller client')
        # Assignment is per task, independently of its position in the DAG.
        agent.client.replies = [json.dumps(tasks, indent=2)] + [
            json.dumps({'id': 'shared', 'reason': 'matches task description'}) for _ in tasks
        ] + ['FINAL ANSWER']
        return agent

    async def test_four_stages_and_out_of_order_dependency(self):
        tasks = [task(2, [0, 1], 'Combine <resource>-0 and <resource-1>'), task(0), task(1)]
        agent = self.agent(tasks)
        result = await agent.run_agent({'problem': 'question', 'solution': 'SECRET_GOLD', 'files': ['data.csv']})
        self.assertEqual(result['final_answer'], 'FINAL ANSWER')
        self.assertEqual([t['status'] for t in result['executed_tasks']], ['completed'] * 3)
        self.assertEqual(len(FakeBench.instances), 3)
        self.assertGreaterEqual(FakeBench.peak, 2)
        self.assertTrue(all(worker.closed for worker in FakeBench.instances))
        joined = '\n'.join(c[0] for worker in FakeBench.instances for c in worker.calls)
        self.assertIn('Combine RESOURCE_VALUE and RESOURCE_VALUE', joined)
        self.assertIn('data.csv', joined)
        self.assertNotIn('SECRET_GOLD', joined)
        self.assertEqual(tasks[0]['args']['text'], 'Combine <resource>-0 and <resource-1>')
        self.assertEqual(result['tasks'], tasks)
        self.assertEqual(len(result['messages']), 8)  # plan + 3 selections + 3 executions + summary
        self.assertEqual(len(result['manager_agent_steps']), 3)
        prompts = json.dumps(agent.client.calls)
        self.assertNotIn('SECRET_GOLD', prompts)
        self.assertIn('RESOURCE_VALUE', prompts)
        self.assertIn('FINAL FORMAT', agent.client.calls[-1]['messages'][0]['content'])
        await agent.aclose()
        self.assertTrue(agent.client.closed)

    async def test_response_uses_final_answer_tool_and_common_request_limits(self):
        agent = self.agent([task(0)])
        result = await agent.run_agent({'problem': 'question'})
        self.assertEqual(result['final_answer'], 'FINAL ANSWER')
        request = agent.client.calls[-1]
        self.assertEqual(request.get('tool_choice'), {'type': 'function', 'function': {'name': 'final_answer'}})
        self.assertEqual(request['temperature'], 0.2)
        self.assertEqual(request['top_p'], 1.0)
        self.assertEqual(request['max_tokens'], 8192)

    async def test_failed_dependency_blocks_descendant_but_runs_independent_task(self):
        FakeBench.fail_text = 'FAIL_THIS'
        agent = self.agent([task(0, text='FAIL_THIS'), task(1, [0], '<resource>-0'), task(2)])
        result = await agent.run_agent({'problem': 'question'})
        self.assertEqual([t['status'] for t in result['executed_tasks']], ['failed', 'blocked', 'completed'])
        self.assertEqual(len(FakeBench.instances), 2)
        self.assertIn('failure trace', str(result['messages']))
        self.assertEqual(result['workflow_status'], 'partial')
        self.assertIn('blocked', agent.client.calls[-1]['messages'][1]['content'])

    async def test_invalid_plans_fail_explicitly_without_execution(self):
        invalid = [[], [task(0), task(0)], [task(0, [7])],
                   [task(0, [1]), task(1, [0])], [task(0, text='<resource>-9')],
                   [{'id': True, 'task': 'x', 'dep': [-1], 'args': {}}]]
        for plan in invalid:
            with self.subTest(plan=plan):
                agent = self.agent(plan)
                result = await agent.run_agent({'problem': 'question'})
                self.assertIn('error', result)
                self.assertEqual(result['workflow_status'], 'failed')
                self.assertEqual(len(agent.client.calls), 1)
        self.assertEqual(FakeBench.instances, [])

    async def test_fenced_multiline_json_and_nested_resources(self):
        tasks = [task(0), task(1, [0])]
        tasks[1]['args'] = {'nested': [{'value': '<resource>-0'}, 42]}
        agent = self.agent(tasks)
        agent.client.replies[0] = '```json\n' + json.dumps(tasks, indent=2) + '\n```'
        result = await agent.run_agent({'problem': 'question'})
        self.assertEqual(result['workflow_status'], 'completed')
        self.assertIn('RESOURCE_VALUE', FakeBench.instances[-1].calls[0][0])

    async def test_unknown_model_is_error_not_fallback(self):
        agent = self.agent([task(0)])
        agent.client.replies[1] = '{"id":"invented","reason":"guess"}'
        result = await agent.run_agent({'problem': 'question'})
        self.assertIn('error', result)
        self.assertEqual(FakeBench.instances, [])

    async def test_selected_model_and_tools_are_used(self):
        models = [{'id': 'specialist', 'model': 'expert-model', 'description': 'text specialist'}]
        agent = self.agent([task(0)], jarvis_models=models, manager_tools=['python_interpreter'], search_tools=['wikipedia'])
        agent.client.replies[1] = '{"id":"specialist","reason":"text specialist"}'
        with patch.dict(os.environ, {'MODEL_NAME': 'other-model'}):
            result = await agent.run_agent({'problem': 'question'})
        self.assertEqual(result['workflow_status'], 'completed')
        worker = FakeBench.instances[0]
        self.assertEqual(worker.llm.model_id, 'expert-model')
        self.assertEqual(worker.config['manager_tools'], ['python_interpreter'])
        self.assertEqual(worker.config['search_tools'], ['wikipedia'])
        self.assertIsNone(worker.config.get('evaluator'))

    async def test_concurrency_limit(self):
        agent = self.agent([task(0), task(1)], jarvis_max_parallel=1)
        result = await agent.run_agent({'problem': 'question'})
        self.assertEqual(result['workflow_status'], 'completed')
        self.assertEqual(FakeBench.peak, 1)

    async def test_invalid_json_retains_planning_usage(self):
        agent = self.agent([task(0)])
        agent.client.replies[0] = 'not json'
        result = await agent.run_agent({'problem': 'question'})
        self.assertIn('error', result)
        self.assertEqual(result['messages'][0]['usage_metadata'].total_tokens, 7)

    async def test_dependency_outputs_without_resource_placeholders(self):
        agent = self.agent([task(0), task(1, [0], 'Summarize the prerequisite')])
        result = await agent.run_agent({'problem': 'question'})
        self.assertEqual(result['workflow_status'], 'completed')
        self.assertIn('RESOURCE_VALUE', FakeBench.instances[-1].calls[0][0])

    async def test_attachments_are_preprocessed(self):
        agent = self.agent([task(0)])
        await agent.run_agent({'problem': 'question', 'files': ['chart.png']})
        self.assertIn('Attachment description: chart.png', FakeBench.instances[0].calls[0][0])

    async def test_usage_counts_shared_model_monitor_once(self):
        agent = self.agent([task(0)])
        await agent.run_agent({'problem': 'question'})
        usage = FakeBench.instances[0]._extract_token_usage_from_agent()
        self.assertEqual(usage['total_tokens'], 16)

    async def test_multiple_runs_do_not_share_task_state(self):
        agent = self.agent([task(0)])
        first = await agent.run_agent({'problem': 'first'})
        agent.client.replies = [json.dumps([task(8)]), '{"id":"shared","reason":"suitable"}', 'SECOND']
        second = await agent.run_agent({'problem': 'second'})
        self.assertEqual(first['final_answer'], 'FINAL ANSWER')
        self.assertEqual(second['final_answer'], 'SECOND')
        self.assertEqual([t['id'] for t in second['executed_tasks']], [8])
        self.assertNotIn('first', FakeBench.instances[-1].calls[0][0])

    async def test_plan_limit(self):
        agent = self.agent([task(0), task(1)], jarvis_max_tasks=1)
        result = await agent.run_agent({'problem': 'question'})
        self.assertIn('error', result)
        self.assertEqual(FakeBench.instances, [])


if __name__ == '__main__':
    unittest.main()
