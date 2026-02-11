#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独测试 chateval_newcore 在 GAIA 中的「搜索类」题目。

从 GAIA 数据中根据 Annotator Metadata 的 Tools 字段筛选出需要搜索/网页/百科的题目，
写入一份仅含这些题目的 jsonl，然后仅对这份数据跑 chateval_newcore。

用法（建议在项目根目录下用 uv run 执行）:
  # 使用默认 GAIA level1 数据，筛选并跑全部搜索题
  uv run python scripts/run_chateval_newcore_gaia_search_only.py

  # 只跑前 5 道搜索题（便于快速验证）
  uv run python scripts/run_chateval_newcore_gaia_search_only.py --limit 5

  # 只生成搜索题 jsonl，不跑 benchmark
  uv run python scripts/run_chateval_newcore_gaia_search_only.py --no-run

  # 指定输入/输出数据路径与日志
  uv run python scripts/run_chateval_newcore_gaia_search_only.py \\
    --data_path data/gaia_validate_level1.jsonl \\
    --output_data data/gaia_validate_level1_search_only.jsonl \\
    --log_file logs/chateval_newcore_gaia_search_20260204.log
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 保证可以 import mas_arena
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mas_arena.benchmark_runner import BenchmarkRunner


def is_gaia_search_problem(problem: dict) -> bool:
    """
    根据 GAIA 题目的 Annotator Metadata 判断是否为「搜索类」题目。
    若 Tools 字段中包含 search / browser / google / web（不区分大小写），则视为需要搜索。
    """
    meta = problem.get("Annotator Metadata") or {}
    tools_str = meta.get("Tools") or ""
    if not isinstance(tools_str, str):
        tools_str = str(tools_str)
    t = tools_str.lower()
    if "search" in t or "browser" in t or "google" in t or "web browser" in t:
        return True
    # 题目描述中明确需要上网/百科的也纳入（无 Tools 时兜底）
    question = (problem.get("Question") or "").lower()
    if not t and (
        "wikipedia" in question
        or "http" in question
        or "url" in question
        or "search the web" in question
    ):
        return True
    return False


def filter_gaia_search_problems(
    data_path: str,
    output_path: str,
) -> tuple[list[dict], int]:
    """
    从 data_path 的 jsonl 中读取 GAIA 题目，筛选出搜索类题目并写入 output_path。
    返回 (筛选后的题目列表, 原始总题数)。
    """
    with open(data_path, "r", encoding="utf-8") as f:
        all_problems = [json.loads(line) for line in f]
    total = len(all_problems)
    search_problems = [p for p in all_problems if is_gaia_search_problem(p)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in search_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return search_problems, total


def parse_args():
    p = argparse.ArgumentParser(
        description="Run chateval_newcore on GAIA search-only problems."
    )
    p.add_argument(
        "--data_path",
        default="data/gaia_validate_level1.jsonl",
        help="GAIA jsonl 路径",
    )
    p.add_argument(
        "--output_data",
        default="data/gaia_validate_level1_search_only.jsonl",
        help="筛选出的搜索题 jsonl 输出路径",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多跑多少道题，0 表示全部搜索题",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="并发数（搜索题建议适当降低，减轻 API 限流）",
    )
    p.add_argument(
        "--log_file",
        default=None,
        help="日志文件路径，不指定则自动生成到 logs/",
    )
    p.add_argument(
        "--no-run",
        action="store_true",
        help="只筛选并写入 output_data，不跑 benchmark",
    )
    p.add_argument(
        "--skip-filter",
        action="store_true",
        help="不筛选，直接用 data_path 作为题目列表跑 benchmark（用于已有搜索题 jsonl）",
    )
    return p.parse_args()


async def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.data_path):
        print(f"Error: data file not found: {args.data_path}")
        sys.exit(1)

    if args.skip_filter:
        with open(args.data_path, "r", encoding="utf-8") as f:
            search_problems = [json.loads(line) for line in f]
        total = len(search_problems)
        print(f"跳过筛选，直接使用 {args.data_path}，题数: {total}")
        run_data_path = args.data_path
    else:
        search_problems, total = filter_gaia_search_problems(
            args.data_path,
            args.output_data,
        )
        print(f"GAIA 总题数: {total}, 筛选出的搜索题数: {len(search_problems)}")
        print(f"搜索题已写入: {args.output_data}")
        run_data_path = args.output_data

    if args.no_run:
        return

    if not search_problems:
        print("没有搜索题，退出.")
        sys.exit(0)

    log_file = args.log_file
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/chateval_newcore_gaia_search_{timestamp}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    evaluator = "gaia"
    agent_config = {
        "num_rounds": 4,
        "model_name": "gpt-4.1",
        "manager_tools": ["ALL"],
        "search_tools": ["ALL"],
        "memory": None,
        "verbosity_level": 100,
        "evaluator": evaluator,
        "max_steps": 15,
    }

    print(f"Running chateval_newcore on GAIA search-only problems. Log: {log_file}")
    summary = await BenchmarkRunner().arun(
        benchmark_name=evaluator,
        agent_system="chateval_newcore",
        agent_config=agent_config,
        limit=args.limit if args.limit > 0 else 0,
        pass_at_k=1,
        log_file=log_file,
        data_path=run_data_path,
        concurrency=args.concurrency,
    )
    print("\n==== chateval_newcore GAIA 搜索题 测试结果 ====")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
