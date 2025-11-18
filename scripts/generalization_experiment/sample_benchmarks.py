#!/usr/bin/env python3
"""
Sample 200 lines from several benchmark jsonl files reproducibly.

Creates files named: {benchmark}_test_200.jsonl (benchmark lowercased, spaces removed)

Usage:
    python3 sample_benchmarks.py [--seed SEED] [--n N] [--out-dir PATH]

Defaults: seed=42, n=200, out-dir=data
"""

import argparse
import random
from pathlib import Path
import sys

# Fixed mapping from benchmark name to source jsonl path (relative to repo root)
BENCHFILES = {
    "HotpotQA": "data/hotpotqa_test.jsonl",
    "AIME": "data/aime_test.jsonl",
    "MMLU": "data/mmlu_pro_test.jsonl",
    "BBH": "data/bbh_test.jsonl",
    "Math": "data/math_test.jsonl",
}


def normalize_benchmark(name: str) -> str:
    return name.replace(" ", "").lower()


def sample_file(src_path: Path, n: int, rnd: random.Random):
    if not src_path.exists():
        raise FileNotFoundError(f"source file not found: {src_path}")
    with src_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    if total == 0:
        return []
    if total <= n:
        # If file has fewer than requested, return all lines (deterministic)
        # but shuffle deterministically so output isn't always prefix if user prefers variety
        rnd.shuffle(lines)
        return [line.rstrip("\n") for line in lines]
    # sample indices without replacement, then preserve original order
    indices = rnd.sample(range(total), n)
    indices.sort()
    selected = [lines[i].rstrip("\n") for i in indices]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible sampling")
    parser.add_argument("--n", type=int, default=200, help="number of lines to sample from each source")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="output directory to write sampled files",
    )
    args = parser.parse_args()

    seed = args.seed
    n = args.n
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling n={n} lines per benchmark with seed={seed}. Writing to {out_dir}")

    # Use a single Random object so sampling is reproducible and depends on the ordering of BENCHFILES
    rnd = random.Random(seed)

    for _, (bench, rel_path) in enumerate(BENCHFILES.items()):
        src = Path(rel_path)
        try:
            items = sample_file(src, n, rnd)
        except FileNotFoundError as e:
            print(f"Skipping {bench}: {e}", file=sys.stderr)
            continue
        out_name = f"{normalize_benchmark(bench)}_test_{n}.jsonl"
        out_path = out_dir / out_name
        with out_path.open("w", encoding="utf-8") as out_f:
            for line in items:
                out_f.write(line + "\n")
        print(f"Wrote {len(items)} / {src.name} -> {out_path}")


if __name__ == "__main__":
    main()
