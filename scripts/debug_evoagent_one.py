#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import pdb
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mas_arena.benchmark_runner import BENCHMARKS, normalize_problem_keys
from mas_arena.agents import create_agent_system


def parse_tool_list(value):
    if value is None:
        return None
    if value.lower() == 'none':
        return []
    if value == 'ALL':
        return 'ALL'
    return [item.strip() for item in value.split(',') if item.strip()]


def load_problem(data_path, data_id=None, index=None):
    with open(data_path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if data_id:
        for i, row in enumerate(rows):
            for key in ('id', 'task_id', 'problem_id'):
                if str(row.get(key)) == data_id:
                    return i, row
        raise SystemExit(f'data id not found: {data_id}')
    if index is None:
        index = 0
    if index < 0 or index >= len(rows):
        raise SystemExit(f'index out of range: {index}, total={len(rows)}')
    return index, rows[index]


def summarize_run_output(run_output):
    print('\n=== run_agent returned ===', flush=True)
    print('keys:', sorted(run_output.keys()), flush=True)
    print('final_answer:', repr(run_output.get('final_answer'))[:1000], flush=True)
    print('execution_time_ms:', run_output.get('execution_time_ms'), flush=True)
    print('error:', run_output.get('error'), flush=True)
    messages = run_output.get('messages') or []
    print('messages:', len(messages), flush=True)
    for offset, msg in enumerate(messages[-5:], start=max(0, len(messages) - 5)):
        content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
        name = msg.get('name') if isinstance(msg, dict) else getattr(msg, 'name', None)
        usage = msg.get('usage_metadata') if isinstance(msg, dict) else getattr(msg, 'usage_metadata', None)
        print(f'  msg[{offset}] name={name!r} content={content!r} usage={usage!r}', flush=True)


async def main():
    parser = argparse.ArgumentParser(description='Run one EvoAgent GAIA problem with debug checkpoints.')
    parser.add_argument('--data', default='data/gaia_validate_level2.jsonl')
    parser.add_argument('--data-id', default=None)
    parser.add_argument('--index', type=int, default=None, help='0-based row index; ignored when --data-id is set')
    parser.add_argument('--model-name', default=os.getenv('MODEL_NAME', 'qwen3-32b'))
    parser.add_argument('--manager-tools', default='ALL')
    parser.add_argument('--search-tools', default='ALL')
    parser.add_argument('--evo-agent-timeout-seconds', type=float, default=None)
    parser.add_argument('--break-before', action='store_true')
    parser.add_argument('--break-after-run-agent', action='store_true')
    parser.add_argument('--no-evaluate', action='store_true', help='Only call run_agent; skip evaluator.evaluate')
    args = parser.parse_args()

    benchmark_config = BENCHMARKS['gaia']
    row_index, row = load_problem(args.data, args.data_id, args.index)
    problem = normalize_problem_keys(row, benchmark_config.get('normalization_keys', {}), row_index)

    agent_config = {
        'model_name': args.model_name,
        'manager_tools': parse_tool_list(args.manager_tools),
        'search_tools': parse_tool_list(args.search_tools),
    }
    if args.evo_agent_timeout_seconds is not None:
        agent_config['agent_task_timeout_seconds'] = args.evo_agent_timeout_seconds

    print('=== debug problem ===', flush=True)
    print('row_index:', row_index, flush=True)
    print('id:', problem.get('id'), flush=True)
    print('solution:', problem.get('solution'), flush=True)
    print('problem:', problem.get('problem', '')[:2000], flush=True)
    print('agent_config:', agent_config, flush=True)

    agent = create_agent_system('evoagent', agent_config)
    agent.evaluator_name = 'gaia'
    agent._initialize_evaluator()

    if args.break_before:
        pdb.set_trace()

    try:
        run_output = await agent.run_agent(problem)
    except Exception:
        print('\n=== run_agent exception ===', flush=True)
        traceback.print_exc()
        pdb.post_mortem(sys.exc_info()[2])
        raise

    summarize_run_output(run_output)

    if args.break_after_run_agent:
        pdb.set_trace()

    if not args.no_evaluate:
        try:
            eval_result = await agent.evaluator.evaluate(problem=problem, run_result=run_output)
            print('\n=== evaluator result ===', flush=True)
            print(json.dumps(eval_result, indent=2, ensure_ascii=False, default=str), flush=True)
        except Exception:
            print('\n=== evaluator exception ===', flush=True)
            traceback.print_exc()
            pdb.post_mortem(sys.exc_info()[2])
            raise


if __name__ == '__main__':
    asyncio.run(main())
