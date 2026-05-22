#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 bench_agent GAIA 结果中筛选因 API 429 / TPM 限流导致失败的任务，统计并重跑。

典型失败 prediction:
  Code agent encountered an error: Error in code agent step 1: Error code: 429 - ...

用法（项目根目录）:
  # 仅统计
  python scripts/rerun_gaia_429_failures.py \\
    --results results/gaia_bench_agent_20260522_004730.json

  # 统计 + 重跑（并发 3）
  python scripts/rerun_gaia_429_failures.py \\
    --results results/gaia_bench_agent_20260522_004730.json \\
    --run

  # 只生成重跑 jsonl，不执行 benchmark
  python scripts/rerun_gaia_429_failures.py \\
    --results results/gaia_bench_agent_20260522_004730.json \\
    --no-run

  # 合并基线 + 重跑结果（重跑文件覆盖基线中同 task_id）
  python scripts/rerun_gaia_429_failures.py \\
    --merge \\
    --baseline results/gaia_bench_agent_20260522_004730.json \\
    --rerun-results results/gaia_bench_agent_20260522_XXXXXX.json

  # 仅查看任意结果 JSON 的统计
  python scripts/rerun_gaia_429_failures.py \\
    --report results/gaia_bench_agent_20260522_004736.json

  # 从结果文件统计全部错题并生成重跑 jsonl（不限 429）
  python scripts/rerun_gaia_429_failures.py \\
    --results results/gaia_bench_agent_20260522_004736.json \\
    --failed-only --no-run

  python scripts/rerun_gaia_429_failures.py \\
    --results results/gaia_bench_agent_20260522_004736.json \\
    --failed-only --run --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mas_arena.benchmark_runner import BenchmarkRunner

RATE_LIMIT_PATTERN = re.compile(
    r"Error code:\s*429|rate limiting|TPM limit reached",
    re.IGNORECASE,
)

DEFAULT_GAIA_LOOKUP_PATHS = [
    "data/gaia_validate.jsonl",
    "data/gaia_validate_level1.jsonl",
    "data/gaia_validate_level2_noaudio.jsonl",
    "data/gaia_validate_level3.jsonl",
]

LEVEL_DATA_PATHS = {
    "1": "data/gaia_validate_level1.jsonl",
    "2": "data/gaia_validate_level2_noaudio.jsonl",
    "3": "data/gaia_validate_level3.jsonl",
}


def parse_tool_config(raw: str) -> list[str]:
    if raw.strip().upper() == "ALL":
        return ["ALL"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_rate_limit_failure(result: dict) -> bool:
    prediction = result.get("prediction") or ""
    if not prediction:
        return False
    if "Code agent encountered an error" not in prediction:
        return False
    return bool(RATE_LIMIT_PATTERN.search(prediction))


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gaia_problems(data_path: str) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_gaia_index(lookup_paths: list[str]) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for path in lookup_paths:
        p = Path(path)
        if not p.exists():
            continue
        for problem in load_gaia_problems(str(p)):
            by_id[problem["task_id"]] = problem
    return by_id


def extract_rerun_problems(
    problem_ids: list[str],
    output_path: str,
    gaia_data_path: str | None = None,
    lookup_paths: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    by_id = build_gaia_index(lookup_paths or DEFAULT_GAIA_LOOKUP_PATHS)
    if gaia_data_path and Path(gaia_data_path).exists():
        for problem in load_gaia_problems(gaia_data_path):
            by_id[problem["task_id"]] = problem

    missing_ids = [pid for pid in problem_ids if pid not in by_id]
    rerun_problems = [by_id[pid] for pid in problem_ids if pid in by_id]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for problem in rerun_problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    return rerun_problems, missing_ids


def classify_failure(result: dict) -> str:
    if is_rate_limit_failure(result):
        return "429"
    pred = (result.get("prediction") or "").strip()
    if pred.startswith("```"):
        return "code_snippet"
    if "Code agent encountered an error" in pred:
        return "agent_error"
    return "wrong_answer"


def print_statistics(results_payload: dict, rate_limit_items: list[dict]) -> None:
    summary = results_payload.get("summary", {})
    all_results = results_payload.get("results", [])
    total = summary.get("total_problems") or len(all_results)
    correct = summary.get("correct")
    if correct is None:
        correct = sum(1 for r in all_results if r.get("is_correct"))
    accuracy = summary.get("accuracy")
    if accuracy is None and total:
        accuracy = correct / total

    print("\n" + "=" * 72)
    print("429 / TPM 限流失败任务统计")
    print("=" * 72)
    print(f"来源结果文件: {summary.get('results_file', '(unknown)')}")
    print(f"原始总题数:   {total}")
    print(f"原始正确数:   {correct}")
    print(f"原始准确率:   {accuracy * 100:.2f}%" if accuracy is not None else "原始准确率:   N/A")
    print(f"429 失败数:   {len(rate_limit_items)}")
    print(f"429 占比:     {len(rate_limit_items) / total * 100:.2f}%" if total else "429 占比:     N/A")

    if total:
        hypothetical_correct = correct + len(rate_limit_items)
        hypothetical_acc = hypothetical_correct / total * 100
        print(
            f"若 429 题全部重跑成功: {hypothetical_correct}/{total} "
            f"= {hypothetical_acc:.2f}%"
        )
        need_for_30 = int(total * 0.3 + 0.999999)
        gap = max(0, need_for_30 - correct)
        print(f"达到 30% 至少还需答对: {gap} 题 (目标 {need_for_30}/{total})")

    print("\n429 失败 task_id 列表:")
    for idx, item in enumerate(rate_limit_items, 1):
        pid = item["problem_id"]
        step_match = re.search(r"code agent step (\d+)", item.get("prediction", ""), re.I)
        step_info = f"step={step_match.group(1)}" if step_match else "step=?"
        question = (item.get("problem") or "").replace("\n", " ")
        if len(question) > 90:
            question = question[:87] + "..."
        print(f"  {idx:2d}. {pid}  ({step_info})  {question}")

    print("=" * 72 + "\n")


def print_failed_statistics(results_payload: dict, failed_items: list[dict]) -> None:
    all_results = results_payload.get("results", [])
    total = len(all_results)
    correct = sum(1 for r in all_results if r.get("is_correct"))
    by_kind: dict[str, list[dict]] = {}
    for item in failed_items:
        by_kind.setdefault(classify_failure(item), []).append(item)

    print("\n" + "=" * 72)
    print("错题统计（is_correct=false）")
    print("=" * 72)
    print(f"来源: {results_payload.get('summary', {}).get('results_file', '(unknown)')}")
    print(f"总题数: {total}  正确: {correct}  错题: {len(failed_items)}")
    if total:
        print(f"准确率: {correct / total * 100:.2f}%")
    for kind in ("429", "agent_error", "code_snippet", "wrong_answer"):
        items = by_kind.get(kind, [])
        if items:
            print(f"  - {kind}: {len(items)}")
    print("\n错题列表 (task_id | expected | 类型 | prediction 摘要):")
    for idx, item in enumerate(failed_items, 1):
        pid = item["problem_id"]
        expected = item.get("expected", "")
        kind = classify_failure(item)
        pred = (item.get("prediction") or "").replace("\n", " ")
        if len(pred) > 70:
            pred = pred[:67] + "..."
        print(f"  {idx:2d}. {pid}")
        print(f"      expected={expected!r}  [{kind}]  {pred}")
    print("=" * 72 + "\n")


def print_result_report(path: str, payload: dict, expected_429_ids: set[str] | None = None) -> None:
    results = payload.get("results", [])
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    rate429 = [r for r in results if is_rate_limit_failure(r)]
    still429_ids = {r["problem_id"] for r in rate429}

    print("\n" + "=" * 72)
    print(f"结果报告: {path}")
    print("=" * 72)
    print(f"题数: {total}  正确: {correct}  准确率: {correct / total * 100:.2f}%" if total else "题数: 0")
    print(f"仍含 429: {len(rate429)}")
    if expected_429_ids is not None:
        overlap = still429_ids & expected_429_ids
        print(f"与基线 429 列表交集: {len(overlap)}/{len(expected_429_ids)}")
        if len(overlap) == 0 and total > 0:
            print("Warning: 此结果文件与基线 429 重跑题集无交集，可能不是 429 重跑产物。")
    print("=" * 72 + "\n")


def merge_baseline_and_rerun(
    baseline_path: str,
    rerun_path: str,
    output_path: str | None,
    gaia_data_path: str,
) -> dict:
    baseline = load_results(baseline_path)
    rerun = load_results(rerun_path)
    baseline_by_id = {r["problem_id"]: r for r in baseline.get("results", [])}
    rerun_by_id = {r["problem_id"]: r for r in rerun.get("results", [])}

    orig429_ids = {pid for pid, r in baseline_by_id.items() if is_rate_limit_failure(r)}
    overlap = orig429_ids & set(rerun_by_id.keys())

    merged_results = []
    for pid, row in baseline_by_id.items():
        merged_results.append(rerun_by_id.get(pid, row))

    merged_correct = sum(1 for r in merged_results if r.get("is_correct"))
    total = len(merged_results)
    still429 = sum(1 for r in merged_results if is_rate_limit_failure(r))
    fixed = len([pid for pid in orig429_ids if pid in rerun_by_id and not is_rate_limit_failure(rerun_by_id[pid])])
    newly_correct = len(
        [
            pid
            for pid in overlap
            if not baseline_by_id[pid].get("is_correct") and rerun_by_id[pid].get("is_correct")
        ]
    )

    merged_payload = {
        "summary": {
            "benchmark": baseline.get("summary", {}).get("benchmark", "gaia"),
            "agent_system": baseline.get("summary", {}).get("agent_system", "bench_agent"),
            "total_problems": total,
            "correct": merged_correct,
            "errored": 0,
            "accuracy": merged_correct / total if total else 0.0,
            "baseline_file": baseline_path,
            "rerun_file": rerun_path,
            "rerun_overlap_with_429": len(overlap),
            "rerun_fixed_429": fixed,
            "rerun_newly_correct": newly_correct,
            "merged_still_429": still429,
        },
        "results": merged_results,
    }

    print("\n" + "=" * 72)
    print("合并统计（基线 + 重跑覆盖）")
    print("=" * 72)
    print(f"基线:     {baseline_path}")
    print(f"重跑:     {rerun_path}")
    print(f"基线 429: {len(orig429_ids)} 题")
    print(f"重跑覆盖: {len(overlap)} / {len(orig429_ids)} 题在重跑结果中有记录")
    if len(overlap) == 0:
        print("Warning: 重跑结果与基线 429 题无交集，请确认 --rerun-results 是否为 429 重跑输出。")
    print(f"基线正确: {sum(1 for r in baseline_by_id.values() if r.get('is_correct'))}/{total}")
    print(f"合并正确: {merged_correct}/{total} = {merged_correct / total * 100:.2f}%")
    print(f"429 修复(有答案且非429): {fixed}")
    print(f"重跑新答对: {newly_correct}")
    print(f"合并后仍 429: {still429}")
    need_for_30 = int(total * 0.3 + 0.999999)
    print(f"距 30% 目标({need_for_30}/{total}): 还差 {max(0, need_for_30 - merged_correct)} 题")
    print("=" * 72 + "\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_payload, f, ensure_ascii=False, indent=4)
        print(f"已写入合并结果: {output_path}")

    return merged_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计并重跑 GAIA bench_agent 结果中的 429/TPM 限流失败任务。"
    )
    parser.add_argument(
        "--results",
        default="results/gaia_bench_agent_20260522_004730.json",
        help="bench_agent GAIA 结果 JSON 路径",
    )
    parser.add_argument(
        "--gaia-data",
        default="data/gaia_validate_level1.jsonl",
        help="GAIA 原始 jsonl 数据路径",
    )
    parser.add_argument(
        "--output-data",
        default="data/gaia_validate_level1_429_rerun.jsonl",
        help="筛选出的重跑题目 jsonl 输出路径",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="benchmark 结果输出目录",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="重跑并发数（默认 3）",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="重跑日志路径；默认写入 logs/bench_agent_gaia_429_rerun_<timestamp>.log",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="统计后执行重跑 benchmark",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="只统计并生成重跑 jsonl，不执行 benchmark",
    )
    parser.add_argument(
        "--manager-tools",
        default=os.environ.get("MANAGER_TOOLS", "ALL"),
        help="manager 工具列表，默认 ALL",
    )
    parser.add_argument(
        "--search-tools",
        default=os.environ.get("SEARCH_TOOLS", "ALL"),
        help="search 工具列表，默认 ALL",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合并基线与重跑结果并输出合并准确率",
    )
    parser.add_argument(
        "--baseline",
        default="results/gaia_bench_agent_20260522_004730.json",
        help="合并模式下的基线结果 JSON",
    )
    parser.add_argument(
        "--rerun-results",
        default=None,
        help="合并模式下的重跑结果 JSON",
    )
    parser.add_argument(
        "--merged-output",
        default=None,
        help="合并结果输出路径（可选）",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="RESULTS_JSON",
        help="仅统计指定结果文件（可与 --merge 联用对比 429 列表）",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="筛选全部错题（is_correct=false），不限 429",
    )
    parser.add_argument(
        "--lookup-data",
        action="append",
        default=None,
        help="查找题目的 GAIA jsonl，可多次指定；默认搜索 validate 全量/level1/2/3",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["1", "2", "3"],
        help="只处理指定 Level 的题目（按对应 jsonl 中的 task_id 过滤）",
    )
    return parser.parse_args()


def load_level_task_ids(level: str) -> set[str]:
    path = LEVEL_DATA_PATHS.get(level)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Level {level} data not found: {path}")
    return {json.loads(line)["task_id"] for line in open(path, encoding="utf-8") if line.strip()}


async def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.report:
        report_path = Path(args.report)
        if not report_path.exists():
            print(f"Error: report file not found: {report_path}")
            return 1
        report_payload = load_results(str(report_path))
        expected_429 = None
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            baseline_payload = load_results(str(baseline_path))
            expected_429 = {
                r["problem_id"]
                for r in baseline_payload.get("results", [])
                if is_rate_limit_failure(r)
            }
        print_result_report(str(report_path), report_payload, expected_429)

    if args.merge:
        if not args.rerun_results:
            print("Error: --merge 需要 --rerun-results <重跑结果.json>")
            return 1
        if not Path(args.baseline).exists():
            print(f"Error: baseline not found: {args.baseline}")
            return 1
        if not Path(args.rerun_results).exists():
            print(f"Error: rerun results not found: {args.rerun_results}")
            return 1
        merge_baseline_and_rerun(
            args.baseline,
            args.rerun_results,
            args.merged_output,
            args.gaia_data,
        )
        if args.report:
            return 0
        if not args.run:
            return 0

    if args.report and not args.run and not args.merge:
        return 0

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        return 1

    lookup_paths = args.lookup_data or DEFAULT_GAIA_LOOKUP_PATHS
    gaia_data_path = args.gaia_data if Path(args.gaia_data).exists() else None
    if not any(Path(p).exists() for p in lookup_paths) and not gaia_data_path:
        print(f"Error: no GAIA data found. Tried: {lookup_paths}, {args.gaia_data}")
        return 1

    payload = load_results(str(results_path))
    all_results = payload.get("results", [])

    if args.failed_only:
        target_items = [r for r in all_results if not r.get("is_correct")]
        print_failed_statistics(payload, target_items)
    else:
        target_items = [r for r in all_results if is_rate_limit_failure(r)]
        print_statistics(payload, target_items)

    if args.level:
        level_ids = load_level_task_ids(args.level)
        before = len(target_items)
        target_items = [r for r in target_items if r["problem_id"] in level_ids]
        print(f"Level {args.level} 过滤: {before} -> {len(target_items)} 题")
        if args.output_data in (
            "data/gaia_validate_level1_429_rerun.jsonl",
            f"data/gaia_429_rerun_{results_path.stem}.jsonl",
        ):
            args.output_data = f"data/gaia_level{args.level}_429_rerun.jsonl"

    problem_ids = [r["problem_id"] for r in target_items]

    if not problem_ids:
        label = "错题" if args.failed_only else "429/TPM 限流失败"
        print(f"未发现{label}任务，无需重跑。")
        return 0

    default_429_out = "data/gaia_validate_level1_429_rerun.jsonl"
    if not args.failed_only and args.output_data == default_429_out:
        stem = results_path.stem
        args.output_data = f"data/gaia_429_rerun_{stem}.jsonl"
    if args.failed_only and args.output_data == default_429_out:
        stem = results_path.stem
        args.output_data = f"data/gaia_failed_rerun_{stem}.jsonl"

    rerun_problems, missing_ids = extract_rerun_problems(
        problem_ids,
        args.output_data,
        gaia_data_path=gaia_data_path,
        lookup_paths=lookup_paths,
    )
    print(f"已写入重跑 jsonl: {args.output_data} ({len(rerun_problems)} 题)")
    if missing_ids:
        print(f"Warning: 以下 task_id 在 GAIA 数据中未找到: {', '.join(missing_ids)}")

    if args.no_run or not args.run:
        if not args.run and not args.no_run:
            print("提示: 加上 --run 可立即重跑上述任务。")
        return 0

    log_file = args.log_file
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "failed" if args.failed_only else "429"
        log_file = f"logs/bench_agent_gaia_{prefix}_rerun_{timestamp}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    agent_config = {
        "manager_tools": parse_tool_config(args.manager_tools),
        "search_tools": parse_tool_config(args.search_tools),
        "evaluator": "gaia",
        "max_steps": 15,
    }

    print(f"开始重跑 {len(rerun_problems)} 题, concurrency={args.concurrency}")
    print(f"Log: {log_file}")

    summary = await BenchmarkRunner(results_dir=args.results_dir).arun(
        benchmark_name="gaia",
        agent_system="bench_agent",
        agent_config=agent_config,
        data_path=args.output_data,
        concurrency=args.concurrency,
        pass_at_k=1,
        log_file=log_file,
        verbose=True,
    )

    label = "错题" if args.failed_only else "429"
    print(f"\n==== {label} 重跑结果 ====")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
