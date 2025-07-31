"""
DROP Evaluator

Standalone evaluator for DROP (Discrete Reasoning Over Paragraphs) problems.

"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Any

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils import calculate_f1_score, normalize_answer

_ANS_TAG_RE   = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.IGNORECASE)
_FINAL_RE     = re.compile(r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:\-]?\s*([\s\S]+)", re.IGNORECASE)


@register_benchmark(
    name="drop",
    normalization_keys={
        "id": "id",
        "problem": "context",
        "solution": "ref_text",
    }
)
class DROPEvaluator(BaseEvaluator):
    """Evaluator for DROP benchmark problems."""

    def __init__(self, name: str = "drop", config: Dict[str, Any] | None = None):
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def _extract_answer(self, raw: Any) -> str:
        """
        Extracts the first plausible answer string from an LLM response.
        1. <answer> … </answer>
        2. "Final Answer:" / "Answer:"
        3. Last non-empty line of text
        Always returns a trimmed string (may be empty).
        """
        txt = str(raw).strip()

        # 1. Check for <answer>...</answer> tags
        m = _ANS_TAG_RE.search(txt)
        if m:
            return m.group(1).strip()

        # 2. Check for "Final Answer:" or "Answer:" prefix
        m = _FINAL_RE.search(txt)
        if m:
            # Take only the first line to avoid including reasoning
            return m.group(1).strip().splitlines()[0].strip()

        # 3. Fallback: last non-empty line
        for line in reversed(txt.splitlines()):
            if line.strip():
                return line.strip()
        return ""

    @staticmethod
    def _normalize(s: Any) -> str:
        """DROP normalization: lowercase -> remove articles/punctuation -> collapse whitespace."""
        return normalize_answer(s)

    def _f1(self, gold: str, pred: str) -> float:
        """Calculates token-level F1 score (AllenNLP-style)."""
        return calculate_f1_score(gold, pred, self._normalize)


    def _make_run(
        self,
        problem: Dict[str, Any],
        final_answer: str,
        extracted: str,
        score: float
    ) -> Run:
        """Creates a LangSmith Run object for evaluation."""
        import uuid
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"context": problem["problem"], "task_id": problem.get("id")},
            outputs={
                "prediction"      : final_answer,
                "extracted_answer": extracted,
                "expected"        : problem["solution"],
                "score"           : score,
                "passed"          : score >= 0.3,  # Matches official threshold of 0.3 for passing
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )


    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a problem against an LLM response.

        Args
        ----
        problem     dict: {"problem": passage+question, "solution": ref_answer, "id": …}
        run_result  dict: {"final_answer": <raw LLM output>, …}

        Returns
        -------
        Dictionary with keys {final_answer, extracted_answer, score}
        """
        raw_out          = run_result.get("final_answer", "")
        extracted_answer = self._extract_answer(raw_out)

        # Support multiple answers (split by |); take the best F1 score for each prediction and gold answer
        gold_list = [x.strip() for x in str(problem["solution"]).split("|") if x.strip()]
        pred_list = [x.strip() for x in extracted_answer.split("|") if x.strip()]

        scores = [
            self._f1(gold, pred)
            for gold in gold_list for pred in pred_list
        ] or [0.0]

        best_f1 = max(scores)

        # Log low-scoring cases
        if best_f1 < 0.3:
            with open(Path(self.log_path) / "mismatches.log", "a", encoding="utf-8") as f:
                f.write("\n" + "-" * 80 + "\n")
                f.write(f"ID       : {problem.get('id')}\n")
                f.write(f"Question : {problem['problem']}\n")
                f.write(f"Expected : {problem['solution']}\n")
                f.write(f"Predicted: {extracted_answer}\n")
                f.write(f"F1 score : {best_f1:.4f}\n")

        # Create LangSmith run and structure the return value
        run = self._make_run(problem, str(raw_out), extracted_answer, best_f1)
        self.run_evaluator.evaluate_run(run=run)

        # Final score: 1.0 if F1 >= 0.3, else use the F1 score directly
        final_score = 1 if best_f1 >= 0.3 else best_f1

        return {
            "final_answer"    : str(raw_out),
            "extracted_answer": extracted_answer,
            "score"           : final_score,
        }