#!/usr/bin/env python3
"""Verify paper tables from retained counts and case derivatives, without APIs.

Only arithmetic and internal consistency are checked. Author-confirmed final
results are never replaced with incomplete historical execution records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any


def method_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return {"benchagentcore": "single", "singleagent": "single"}.get(key, key)


def table_text(text: str, label: str) -> str:
    """Locate a table by its label, never by fragile source line numbers."""
    marker = "\\label{" + label + "}"
    if text.count(marker) != 1:
        raise ValueError(f"Expected one table label: {label}")
    position = text.index(marker)
    starts = list(re.finditer(r"\\begin\{(table\*?|longtable)\}", text[:position]))
    if not starts:
        raise ValueError(f"Missing table environment: {label}")
    start = starts[-1]
    end_marker = "\\end{" + start.group(1) + "}"
    end = text.find(end_marker, position)
    if end < 0:
        raise ValueError(f"Unterminated table: {label}")
    return text[start.start():end]


def row_cells(table: str, name: str) -> list[str]:
    matches = [line.split("&")[1:] for line in table.splitlines()
               if line.strip().split("&")[0].strip() == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one row: {name}")
    return matches[0]


def numbers(text: str) -> list[float]:
    return [float(value.replace(",", ""))
            for value in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


class Audit:
    def __init__(self) -> None:
        self.checked: Counter[str] = Counter()
        self.mismatches: list[dict[str, Any]] = []

    def check(self, kind: str, actual: Any, expected: Any, location: str) -> None:
        self.checked[kind] += 1
        if isinstance(actual, (int, float, Decimal)) and isinstance(expected, (int, float, Decimal)):
            equal = math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-8)
        else:
            equal = actual == expected
        if not equal:
            self.mismatches.append({"kind": kind, "actual": actual, "expected": expected,
                                    "location": location})

    def displayed(self, kind: str, actual: float, expected: float, location: str,
                  digits: int = 2) -> None:
        self.check(kind, actual, float(f"{expected:.{digits}f}"), location)


def verify_trace_counts(audit: Audit, table: str, rows: list[dict[str, str]]) -> None:
    for method, display in [("jarvis", "Jarvis"), ("chateval_newcore", "ChatEval")]:
        group = [row for row in rows if row["method"] == method]
        if not group:
            raise ValueError(f"Missing process-log rows: {method}")
        cells = row_cells(table, display)
        for column, fields in [(3, ["step_external_tool_calls"]),
                               (4, ["executed_external_tool_calls"]),
                               (5, ["direct_external_tool_lines", "executing_code_lines"])]:
            expected = sum(int(row[field]) for row in group for field in fields)
            audit.check("raw_log_process_count", numbers(cells[column])[0], expected, f"{display}/column{column}")


def verify(paper_root: Path) -> dict[str, Any]:
    audit = Audit()
    load = lambda name: json.loads((paper_root / name).read_text(encoding="utf-8"))
    acc = load("data/broad_acc_runs.json")
    tok = load("data/broad_token_runs.json")
    verified = load("data/verified_results.json")
    main = (paper_root / "resources/main.tex").read_text(encoding="utf-8")
    appendix = (paper_root / "resources/appendix.tex").read_text(encoding="utf-8")
    counts = (paper_root / "resources/verified_counts.tex").read_text(encoding="utf-8")
    broad = table_text(main, "tab:main_exp1_overall")
    token_table = table_text(counts, "tab:appendix_broad_tokens")
    count_table = table_text(counts, "tab:appendix_broad_correct_counts")
    agents = {method_key(row["agent"]): row for row in acc["agents"]}
    tokens = {method_key(row["agent"]): row for row in tok["agents"]}
    old = {method_key(row["agent"]): row for row in verified["agents"]}
    names = list(agents)
    benchmarks, totals = acc["benchmarks"], acc["totals"]
    audit.check("broad_size", sum(totals), 3302, "data/broad_acc_runs.json")
    audit.check("method_count", len(names), 7, "data/broad_acc_runs.json")
    audit.check("benchmark_count", len(benchmarks), 10, "data/broad_acc_runs.json")
    audit.check("token_order", tok["benchmarks"], benchmarks, "data/broad_token_runs.json")
    audit.check("token_sizes", tok["totals"], totals, "data/broad_token_runs.json")
    stats, pooled, token_means, token_pooled = {}, {}, {}, {}
    for key, agent in agents.items():
        matrix = agent["correct_by_run"]
        audit.check("run_count", len(matrix), 3, key)
        if len(matrix) != 3 or any(len(row) != len(totals) for row in matrix):
            raise ValueError(f"Invalid run-count matrix: {key}")
        stats[key] = {}
        for index, benchmark in enumerate(benchmarks):
            values = [100 * run[index] / totals[index] for run in matrix]
            for run in matrix:
                audit.check("valid_correct_count", isinstance(run[index], int) and
                            0 <= run[index] <= totals[index], True, f"{key}/{benchmark}")
            stats[key][benchmark] = [statistics.mean(values), statistics.stdev(values)]
            entry = next(entry for entry in old[key]["entries"] if entry["benchmark"] == benchmark)
            audit.check("run1_count", matrix[0][index], entry["correct"], f"{key}/{benchmark}")
            audit.check("run1_total", totals[index], entry["total"], f"{key}/{benchmark}")
        run_values = [100 * sum(run) / sum(totals) for run in matrix]
        pooled[key] = {"runs": run_values, "mean": statistics.mean(run_values),
                       "std": statistics.stdev(run_values)}
        token_row = tokens[key]
        token_matrix = [[Decimal(value) for value in run] for run in token_row["tokens_by_run"]]
        if len(token_matrix) != 3 or any(len(row) != len(totals) for row in token_matrix):
            raise ValueError(f"Invalid token matrix: {key}")
        for run_index, run in enumerate(token_matrix):
            for index, value in enumerate(run):
                expected = Decimal(token_row["source_tokens_by_run"][run_index][index]).quantize(
                    Decimal(1), rounding=ROUND_HALF_UP)
                audit.check("token_half_up", float(value), float(expected), f"{key}/run{run_index+1}/{benchmarks[index]}")
        token_means[key] = [float(sum(run[index] for run in token_matrix) / 3)
                            for index in range(len(totals))]
        run_tokens = [float(sum(value * total for value, total in zip(run, totals)) / sum(totals))
                      for run in token_matrix]
        token_pooled[key] = statistics.mean(run_tokens)
        for index, value in enumerate(token_means[key]):
            audit.check("token_json_avg", float(token_row["token_avg"][index]), value, f"{key}/{benchmarks[index]}")
        for index, value in enumerate(run_tokens):
            audit.check("token_json_run_pool", float(token_row["pooled_tokens_by_run"][index]), value, f"{key}/run{index+1}")
        audit.check("token_json_pool", float(token_row["pooled_token_avg"]), token_pooled[key], key)

    for benchmark in benchmarks + ["Pooled Acc.", "Pooled Tok. (avg)"]:
        cells = row_cells(broad, benchmark)
        audit.check("main_columns", len(cells), 7, benchmark)
        for key, cell in zip(names, cells):
            actual = numbers(cell)
            expected = ([token_pooled[key]] if benchmark == "Pooled Tok. (avg)" else
                        [pooled[key]["mean"], pooled[key]["std"]] if benchmark == "Pooled Acc." else
                        stats[key][benchmark])
            audit.check("main_cell_shape", len(actual), len(expected), f"{benchmark}/{key}")
            for value, target in zip(actual, expected):
                audit.displayed("main_broad", value, target, f"{benchmark}/{key}")
    for index, benchmark in enumerate(benchmarks):
        cells = row_cells(token_table, benchmark)
        audit.check("supp_token_columns", len(cells), 7, benchmark)
        for key, cell in zip(names, cells):
            audit.displayed("supp_token", numbers(cell)[0], token_means[key][index], f"{benchmark}/{key}")
    headings: list[str] = []
    seen_counts = 0
    for line in count_table.splitlines():
        if "\\multicolumn{3}" in line:
            headings = re.findall(r"\\textbf\{([^}]+)\}", line)
        key = method_key(line.split("&")[0].strip())
        if key not in agents or not headings:
            continue
        actual_counts = [(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", line)]
        expected = [(agents[key]["correct_by_run"][run][benchmarks.index(benchmark)],
                     totals[benchmarks.index(benchmark)]) for benchmark in headings for run in range(3)]
        audit.check("supp_count_row", actual_counts, expected, f"{key}/{'/'.join(headings)}")
        seen_counts += len(actual_counts)
    audit.check("supp_count_coverage", seen_counts, 210, "verified_counts.tex")
    csv_rows = list(csv.DictReader((paper_root / "data/token_avg.csv").open(encoding="utf-8-sig")))
    audit.check("csv_method_count", len(csv_rows), 7, "data/token_avg.csv")
    for row in csv_rows:
        key = method_key(row["Agent"])
        for index, benchmark in enumerate(benchmarks):
            audit.displayed("token_csv", float(row[benchmark]), token_means[key][index], f"{key}/{benchmark}")
        audit.displayed("token_csv_pool", float(row["Pooled token avg"]), token_pooled[key], key)

    gaia = {}
    for key, agent in old.items():
        entries = [entry for entry in agent["entries"] if entry["benchmark"].startswith("GAIA-")]
        pairs = [(entry["correct"], entry["total"]) for entry in entries]
        audit.check("gaia_level_sizes", [n for _, n in pairs], [53, 86, 26], key)
        gaia[key] = pairs + [(sum(k for k, _ in pairs), 165)]
    cc = verified["cc_workflow"]
    gaia["ccworkflow"] = list(zip(cc["counts"], cc["totals"])) + [(cc["overall_correct"], cc["overall_total"])]
    for source, label, expected_count in [(counts, "tab:appendix_gaia_correct_counts", 8),
                                          (main, "tab:main_exp2_gaia", 7)]:
        seen = set()
        for line in table_text(source, label).splitlines():
            key = method_key(line.split("&")[0].strip())
            if key not in gaia:
                continue
            seen.add(key)
            pcts = [float(value) for value in re.findall(r"(\d+\.\d+)\\%", line)]
            audit.check("gaia_columns", len(pcts), 4, f"{label}/{key}")
            for value, (correct, total) in zip(pcts, gaia[key]):
                audit.displayed("gaia_percentage", value, 100 * correct / total, f"{label}/{key}")
            raw = [(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", line)]
            if source == counts:
                audit.check("gaia_counts", raw, gaia[key], key)
        audit.check("gaia_table_coverage", len(seen), expected_count, label)
    qwen = {}
    for line in table_text(main, "tab:main_qwen32b_gaia_generalization").splitlines():
        key = method_key(line.split("&")[0].strip())
        if key not in agents:
            continue
        values = [float(value) for value in re.findall(r"(\d+\.\d+)\\%", line)]
        if len(values) != 4:
            raise ValueError(f"Expected four Qwen accuracy cells: {key}")
        implied = [round(value * n / 100) for value, n in zip(values[:3], [53, 86, 26])]
        for value, correct, total in zip(values[:3], implied, [53, 86, 26]):
            audit.displayed("qwen_integer_coherence", value, 100 * correct / total, key)
        audit.displayed("qwen_pooled_coherence", values[3], 100 * sum(implied) / 165, key)
        qwen[key] = implied
    audit.check("qwen_coverage", len(qwen), 7, "main.tex")
    glm = load("data/glm_gaia_verified_counts.json")
    glm_row = next(line for line in table_text(appendix, "tab:appendix_glm5_gaia_generalization").splitlines()
                   if line.startswith("Single-Agent &"))
    glm_values = [float(value) for value in re.findall(r"(\d+\.\d+)\\%", glm_row)]
    for entry in glm["entries"]:
        level = int(re.search(r"[23]", entry.get("benchmark", entry.get("level", ""))).group())
        audit.displayed("glm_confirmed_accuracy", glm_values[level-1], 100*entry["correct"]/entry["total"], f"GLM/L{level}")

    case = load("data/chateval_gsm8k_case_study.json")
    rows = case["rows"]
    audit.check("gsm8k_subset", len(rows), 371, "data/chateval_gsm8k_case_study.json")
    audit.check("gsm8k_unique", len({row["task_id"] for row in rows}), len(rows), "gsm8k")
    audit.check("gsm8k_partition", len(rows)+len(case["excluded_tasks"]), case["matched_tasks"], "gsm8k")
    transitions = Counter()
    for row in rows:
        parse = lambda value: Decimal(str(value).replace("$", "").replace(",", "").rstrip("."))
        audit.check("gsm8k_initial_label", abs(parse(row["first_role_answer"])-parse(row["reference_answer"])) < Decimal("0.000001"), row["first_correct"], row["task_id"])
        transitions[f"{'correct' if row['first_correct'] else 'incorrect'}_to_{'correct' if row['final_correct'] else 'incorrect'}"] += 1
    audit.check("gsm8k_transitions", dict(transitions), case["transition_counts"], "gsm8k")
    case_table = table_text((paper_root / "resources/chateval_case_study.tex").read_text(), "tab:appendix_chateval_answer_changes")
    for name, expected in [("Correct", [transitions["correct_to_correct"], transitions["correct_to_incorrect"]]),
                           ("Incorrect", [transitions["incorrect_to_correct"], transitions["incorrect_to_incorrect"]])]:
        audit.check("gsm8k_table", [numbers(cell)[0] for cell in row_cells(case_table, name)], expected+[sum(expected)], name)
    for correct in [True, False]:
        values = [row["recorded_tokens"] for row in rows if row["final_correct"] == correct]
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        expected = {"n":len(values), "mean":statistics.mean(values), "median":statistics.median(values), "q1":quartiles[0], "q3":quartiles[2]}
        recorded = case["tokens_by_final_outcome"]["final_correct" if correct else "final_incorrect"]
        for field, value in expected.items():
            audit.check("gsm8k_tokens", recorded[field], value, f"{correct}/{field}")

    pairs = load("analysis/jarvis_chateval_incremental/paired_tasks.json")
    summary = load("analysis/jarvis_chateval_incremental/summary.json")
    signals = load("analysis/jarvis_chateval_incremental/process_signals.json")
    selected_table = table_text(appendix, "tab:appendix_selected_gaia_response_log_stats")
    with (paper_root / "analysis/gaia_trace_stats/gaia_selected_log_stats.csv").open() as stream:
        verify_trace_counts(audit, selected_table, list(csv.DictReader(stream)))
    for method, display in [("jarvis", "Jarvis"), ("chateval_newcore", "ChatEval")]:
        retained = sum(batch["result_rows"] for batch in summary["batches"] if batch["method"] == method)
        cells = row_cells(selected_table, display)
        audit.check("trace_evaluation_denominator", numbers(cells[0]), [165], display)
        matched = numbers(cells[1])
        audit.check("trace_retained_responses", matched[0], retained, display)
        audit.displayed("trace_retained_fraction", matched[1], retained/165, display)
        for cell in cells[2:]:
            values = numbers(cell)
            if len(values) == 2:
                audit.displayed("trace_event_normalization", values[1], values[0]/165, display)
    audit.check("trace_source_records", sum(batch["result_rows"] for batch in summary["batches"]), summary["source_result_records"], "summary.json")
    pair_table = table_text((paper_root / "resources/paired_gaia_case_study.tex").read_text(), "tab:appendix_paired_gaia")
    audit.check("paired_size", len(pairs), 163, "paired_tasks.json")
    audit.check("paired_unique", len({row["task_id"] for row in pairs}), len(pairs), "paired_tasks.json")
    outcomes = Counter(row["outcome"] for row in pairs)
    audit.check("paired_outcomes", dict(outcomes), summary["overall"]["outcomes"], "summary.json")
    for outcome, label in [("both_correct", "Both correct"), ("jarvis_only_correct", "Jarvis only correct"),
                           ("chateval_only_correct", "ChatEval only correct"), ("both_incorrect", "Both incorrect")]:
        group = [row for row in pairs if row["outcome"] == outcome]
        actual = [numbers(cell)[0] for cell in row_cells(pair_table, label)]
        expected = [len(group)] + [round(statistics.mean(row[method]["tokens"] for row in group)) for method in ["jarvis", "chateval"]]
        audit.check("paired_table", actual, expected, outcome)
    for method in ["jarvis", "chateval"]:
        audit.check("paired_success", sum(row[method]["correct"] for row in pairs), summary["overall"][method+"_correct"], method)
        audit.check("paired_total_tokens", sum(row[method]["tokens"] for row in pairs), summary["overall"][method+"_tokens"]["sum"], method)
    ratios = [row["chateval"]["tokens"]/row["jarvis"]["tokens"] for row in pairs]
    audit.check("paired_median_ratio", statistics.median(ratios), summary["overall"]["paired_token_ratio"]["median"], "summary.json")
    audit.check("paired_more_tokens", sum(ratio>1 for ratio in ratios), summary["overall"]["chateval_more_tokens"], "summary.json")
    limit = "Code agent reached maximum steps (15) without completing the task"
    limit_rows = [[s for s in row["chateval"]["snapshots"] if s["answer"] == limit] for row in pairs]
    audit.check("step_limit_records", sum(map(len,limit_rows)), signals["step_limit_record_count"], "process_signals.json")
    audit.check("step_limit_tasks", sum(bool(items) for items in limit_rows), signals["step_limit_task_count"], "process_signals.json")
    audit.check("step_limit_tokens", sum(s["tokens"] for items in limit_rows for s in items), signals["step_limit_associated_recorded_tokens"], "process_signals.json")

    return {"status": "pass" if not audit.mismatches else "fail", "checks":sum(audit.checked.values()),
            "checked":dict(audit.checked), "mismatches":audit.mismatches, "pooled_accuracy":pooled,
            "pooled_tokens":token_pooled, "qwen_implied_counts":qwen,
            "scope_limits":["Final author-confirmed results are not reconstructed from incomplete logs.",
                            "Qwen checks establish integer and pooling coherence, not raw-result provenance.",
                            "GLM source confirms L2/L3 counts only; L1 and auxiliary token provenance are not independently established.",
                            "Case derivatives are checked arithmetically; separate scripts reproduce them from raw records.",
                            "Protocol prose, source workbook formulas, trace attribution, and provider usage accounting require separate audits.",
                            "verified_results.json contains historical Run-1 macro summaries; current pooled results use broad_*_runs.json."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.paper_root)
    except (OSError, ValueError, KeyError, StopIteration, IndexError, TypeError) as error:
        # Exceptions may contain user paths; keep the report safe to redistribute.
        report = {"status":"error", "error_type":type(error).__name__,
                  "error":"Missing or malformed required paper inputs or table structure."}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"status":report["status"], "checks":report.get("checks",0),
                      "mismatches":len(report.get("mismatches",[]))}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
