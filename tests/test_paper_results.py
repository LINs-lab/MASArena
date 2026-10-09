"""Offline regression checks for the paper verifier; no model calls."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_paper_results.py"
SPEC = importlib.util.spec_from_file_location("verify_paper_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PaperVerifierTests(unittest.TestCase):
    def test_numeric_mismatch_is_reported(self):
        audit = MODULE.Audit()
        audit.displayed("mean", 76.65, 76.65051483949122, "table/core")
        audit.displayed("mean", 76.66, 76.65051483949122, "table/altered")
        self.assertEqual(len(audit.mismatches), 1)
        self.assertEqual(audit.mismatches[0]["location"], "table/altered")

    def test_tables_follow_labels_after_line_insertions(self):
        text = "\n" * 80 + "\\begin{table*}\n\\label{tab:result}\nCore & 76.65 & 0.32 \\\\\n\\end{table*}"
        table = MODULE.table_text(text, "tab:result")
        self.assertEqual([MODULE.numbers(cell) for cell in MODULE.row_cells(table, "Core")], [[76.65], [0.32]])

    def test_missing_or_duplicate_table_cannot_silently_pass(self):
        with self.assertRaises(ValueError):
            MODULE.table_text("", "tab:missing")
        text = "\\begin{longtable}\n\\label{tab:x}\n\\end{longtable}"
        with self.assertRaises(ValueError):
            MODULE.table_text(text+text, "tab:x")

    def test_missing_or_duplicate_row_cannot_silently_pass(self):
        with self.assertRaises(ValueError):
            MODULE.row_cells("Other & 1", "Core")
        with self.assertRaises(ValueError):
            MODULE.row_cells("Core & 1\nCore & 2", "Core")

    def test_process_counts_are_checked_against_recomputed_logs(self):
        audit = MODULE.Audit()
        table = ("Jarvis & 165 & 165 (1.00) & 165 (1.00) & 0 & 0 & 22 (0.13) \\\\\n"
                 "ChatEval & 165 & 163 (0.99) & 1878 (11.38) & 4623 (28.02) & 3916 (23.73) & 1189 (7.21) \\\\\n")
        rows = [{"method":"jarvis", "step_external_tool_calls":"0", "executed_external_tool_calls":"0",
                 "direct_external_tool_lines":"10", "executing_code_lines":"12"},
                {"method":"chateval_newcore", "step_external_tool_calls":"5953", "executed_external_tool_calls":"4925",
                 "direct_external_tool_lines":"665", "executing_code_lines":"811"}]
        MODULE.verify_trace_counts(audit, table, rows)
        self.assertEqual(len(audit.mismatches), 3)


if __name__ == "__main__":
    unittest.main()
