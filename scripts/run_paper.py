#!/usr/bin/env python3
"""Print the paper's 210 broad-suite commands; --execute explicitly runs them."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def build_commands(root, data_dir, results_dir, protocol):
    selection = json.loads((data_dir / "selection.json").read_text())
    if selection["total_instances"] != 3302:
        raise ValueError("Prepared broad suite must contain 3,302 instances")
    for benchmark, count in protocol["benchmark_sizes"].items():
        item = selection["benchmarks"][benchmark]
        path = data_dir / item["filename"]
        if item["count"] != count or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"{benchmark}: prepared dataset checksum/count mismatch")
    commands = []
    for run in range(1, protocol["broad_runs"] + 1):
        for workflow in protocol["workflows"]:
            for benchmark in protocol["benchmark_sizes"]:
                command = [sys.executable, str(root / "main.py"), "--benchmark", benchmark,
                           "--agent-system", workflow, "--model-name", protocol["model"],
                           "--max-completion-tokens", str(protocol["max_completion_tokens"]),
                           "--data", str(data_dir / selection["benchmarks"][benchmark]["filename"]),
                           "--results-dir", str(results_dir / f"run_{run}" / workflow / benchmark),
                           "--pass_at_k", "1", "--manager-tools", "python_interpreter",
                           "--search-tools", "ALL" if benchmark == "hotpotqa" else "none"]
                commands.append(command)
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--execute", action="store_true", help="Run model evaluations (API charges apply)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/paper/protocol.json").read_text())
    try:
        commands = build_commands(root, args.data_dir.resolve(), args.results_dir.resolve(), protocol)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Preflight failed: {error}\n")
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, check=True)
    print(f"{len(commands)} commands {'executed' if args.execute else 'prepared (dry run)'}")


if __name__ == "__main__":
    main()
