#!/usr/bin/env python3
"""
对 jarvis 评测结果做批量错误分析与可视化：
1. 在 results/ 下找到各数据集（benchmark）准确率最高的那次 jarvis summary
2. 对每次运行：若存在 failed_responses 目录则运行 failure inference（错误归因）
3. 对每次运行生成可视化
4. 汇总并输出错误分析内容（inference 的 txt 报告）
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FAILURE_DIR = RESULTS_DIR / "failure"


def find_jarvis_summaries():
    """找到所有 jarvis 的 summary 文件"""
    summaries = list(RESULTS_DIR.glob("*_jarvis_*_summary.json"))
    return sorted(summaries)


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_best_per_benchmark(summaries):
    """按 benchmark 分组，每组取准确率最高的一次（相同时取 timestamp 最新）"""
    by_bench = {}
    for p in summaries:
        data = load_summary(p)
        bench = data.get("benchmark", "unknown")
        acc = data.get("accuracy", 0)
        ts = data.get("timestamp", "")
        key = (bench, acc, ts)
        if bench not in by_bench or (
            acc > by_bench[bench]["accuracy"]
            or (acc == by_bench[bench]["accuracy"] and ts > by_bench[bench]["timestamp"])
        ):
            by_bench[bench] = {
                "summary_path": p,
                "accuracy": acc,
                "timestamp": ts,
                "benchmark": bench,
                "data": data,
            }
    return list(by_bench.values())


def failed_responses_dir_for(timestamp):
    """返回该 timestamp 对应的 failed_responses 目录路径"""
    d = FAILURE_DIR / f"failed_responses_{timestamp}"
    return d if d.is_dir() else None


def count_jarvis_failures(failed_dir):
    """目录内 jarvis 开头的 json 数量"""
    if not failed_dir:
        return 0
    return len(list(failed_dir.glob("jarvis_*.json")))


def run_failure_inference(directory_path, output_dir, method="binary_search", model="gpt-4.1"):
    """运行 failure inference，返回 (returncode, stdout, stderr)"""
    script = PROJECT_ROOT / "mas_arena" / "failure" / "inference.py"
    if not script.exists():
        return -1, "", f"Script not found: {script}"
    cmd = [
        "uv", "run", "python", str(script),
        "--method", method,
        "--model", model,
        "--directory_path", str(directory_path),
        "--output_dir", str(output_dir),
    ]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def get_latest_inference_txt(output_dir, method="binary_search", model="gpt-4.1"):
    """获取 output_dir 下最新的 inference txt 报告（按修改时间）"""
    prefix = f"{method}_{model.replace('/', '_')}_agent_responses"
    candidates = list(Path(output_dir).glob(f"{prefix}_*.txt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_visualize(summary_path, output_dir=None, visualizations_dir=None):
    """运行 visualize_benchmark.py visualize --summary <path>"""
    script = PROJECT_ROOT / "mas_arena" / "visualization" / "visualize_benchmark.py"
    if not script.exists():
        return -1, "", f"Script not found: {script}"
    cmd = [
        "uv", "run", "python", str(script),
        "visualize",
        "--summary", str(summary_path),
        "--no-open-browser",  # 批量/子进程环境通常无显示，不尝试打开浏览器
    ]
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    if visualizations_dir:
        cmd.extend(["--visualizations-dir", str(visualizations_dir)])
    try:
        r = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return -1, "", str(e)


def main():
    ap = argparse.ArgumentParser(description="Jarvis 评测批量错误分析与可视化")
    ap.add_argument("--skip-inference", action="store_true", help="跳过 failure inference（仅可视化和汇总已有结果）")
    ap.add_argument("--skip-viz", action="store_true", help="跳过可视化")
    ap.add_argument("--method", default="binary_search", choices=["all_at_once", "step_by_step", "binary_search"], help="Failure inference 方法")
    ap.add_argument("--model", default="gpt-4.1", help="Failure inference 模型")
    ap.add_argument("--output-report", type=str, default="", help="将错误分析汇总写入该文件（默认只打印到 stdout）")
    args = ap.parse_args()

    summaries = find_jarvis_summaries()
    if not summaries:
        print("未找到任何 *_jarvis_*_summary.json")
        return 1

    best = pick_best_per_benchmark(summaries)
    print("=" * 80)
    print("Jarvis 各数据集准确率最高的一次运行")
    print("=" * 80)
    for b in best:
        d = b["data"]
        print(f"  {b['benchmark']}: {b['summary_path'].name}  accuracy={d.get('accuracy', 0):.4f}  total={d.get('total_problems')}  correct={d.get('correct')}  timestamp={b['timestamp']}")
    print()

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Jarvis 批量错误分析与可视化报告")
    report_lines.append("=" * 80)

    for b in best:
        bench = b["benchmark"]
        summary_path = b["summary_path"]
        timestamp = b["timestamp"]
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append(f"数据集: {bench}  准确率: {b['accuracy']:.4f}  timestamp: {timestamp}")
        report_lines.append("-" * 80)

        failed_dir = failed_responses_dir_for(timestamp)
        n_failures = count_jarvis_failures(failed_dir) if failed_dir else 0

        if failed_dir and n_failures > 0:
            report_lines.append(f"失败样本数: {n_failures}  目录: {failed_dir}")
            if not args.skip_inference:
                report_lines.append("运行 failure inference ...")
                code, out, err = run_failure_inference(
                    directory_path=failed_dir,
                    output_dir=FAILURE_DIR,
                    method=args.method,
                    model=args.model,
                )
                if code != 0:
                    report_lines.append(f"Inference 退出码: {code}")
                    if err:
                        report_lines.append("stderr: " + err[:500])
                else:
                    txt_path = get_latest_inference_txt(FAILURE_DIR, method=args.method, model=args.model)
                    if txt_path:
                        report_lines.append(f"错误分析报告: {txt_path}")
                        try:
                            content = txt_path.read_text(encoding="utf-8")
                            report_lines.append("")
                            report_lines.append("--- 错误分析内容（摘要，前 15000 字符）---")
                            report_lines.append(content[:15000])
                            if len(content) > 15000:
                                report_lines.append(f"... [已截断，完整见 {txt_path}]")
                        except Exception as e:
                            report_lines.append(f"读取报告失败: {e}")
            else:
                report_lines.append("未运行 inference，无错误分析报告。去掉 --skip-inference 可执行归因。")
        else:
            report_lines.append("无失败样本或 failed_responses 目录不存在，跳过错误分析。")

        if not args.skip_viz:
            code, out, err = run_visualize(summary_path)
            if code != 0:
                report_lines.append(f"可视化失败: {err or out}")
            else:
                report_lines.append("可视化已生成。")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_text = "\n".join(report_lines)

    if args.output_report:
        Path(args.output_report).write_text(report_text, encoding="utf-8")
        print(f"报告已写入: {args.output_report}")

    print(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
