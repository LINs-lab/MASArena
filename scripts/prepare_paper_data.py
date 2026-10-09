#!/usr/bin/env python3
"""Freeze the paper task sets from locally obtained, checksummed source files."""

import argparse
import hashlib
import json
from pathlib import Path
import random


def select_indices(source_count, evaluated_count):
    if not 0 < evaluated_count <= source_count:
        raise ValueError("Evaluated count must be positive and no larger than source count")
    if source_count == evaluated_count:
        return list(range(source_count))
    return random.Random(42).sample(range(source_count), evaluated_count)


def prepare(source_dir, output_dir, manifest):
    prepared = []
    for benchmark, spec in manifest["benchmarks"].items():
        source = source_dir / spec["filename"]
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"{source.name}: source checksum mismatch; obtain the documented processed snapshot")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != spec["source_count"]:
            raise ValueError(f"{source.name}: unexpected source count")
        indices = select_indices(len(rows), spec["evaluated_count"])
        payload = "".join(json.dumps(rows[i], ensure_ascii=False) + "\n" for i in indices)
        prepared.append((benchmark, spec, indices, payload))
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    for benchmark, spec, indices, payload in prepared:
        destination = output_dir / spec["filename"]
        destination.write_text(payload, encoding="utf-8")
        selected[benchmark] = {"filename": destination.name, "count": len(indices),
                               "source_indices_zero_based": indices,
                               "sha256": hashlib.sha256(payload.encode()).hexdigest()}
    report = {"sampling_seed": 42, "total_instances": sum(x["count"] for x in selected.values()),
              "benchmarks": selected}
    (output_dir / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/paper/datasets.json")
    args = parser.parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        parser.error("output-dir must differ from source-dir")
    try:
        report = prepare(args.source_dir, args.output_dir, json.loads(args.manifest.read_text()))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(f"Prepared {len(report['benchmarks'])} benchmarks, {report['total_instances']} instances")


if __name__ == "__main__":
    main()
