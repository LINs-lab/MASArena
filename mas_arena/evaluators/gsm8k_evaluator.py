"""
GSM8K Evaluator

This module provides a standalone evaluator for GSM8K (Grade School Math 8K) problems.
"""

import re
import time
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark


@register_benchmark(
    name="gsm8k",
    normalization_keys={
        "id": "id",
        "problem": "question",
        "solution": "answer",
    },
)
class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for GSM8K problems"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Create log directory if it doesn't exist
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        # Initialize run evaluator
        self.run_evaluator = RunEvaluator()

    def extract_number(self, text: str) -> Optional[float]:
        """
        Extract the last number from text.

        Args:
            text: The text to extract number from

        Returns:
            The extracted number or None if no number found
        """
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        """
        Calculate score by comparing expected and predicted numbers.

        Args:
            expected_output: The expected number
            prediction: The predicted number

        Returns:
            Tuple of (score, prediction) where score is 1.0 for correct, 0.0 for incorrect
        """
        # Be defensive: missing / unparsable numbers should not crash evaluation
        if expected_output is None or prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    def _get_problem_text(self, problem: Dict[str, Any]) -> str:
        """
        GSM8K problems are normalized upstream to use standard keys:
        - problem["problem"]  : question text
        - problem["solution"] : reference answer text

        But we keep backward compatibility with raw GSM8K keys ("question"/"answer").
        """
        return str(problem.get("problem") or problem.get("question") or "")

    def _get_solution_text(self, problem: Dict[str, Any]) -> str:
        # Standard normalized key first; fall back to raw/legacy keys.
        return str(problem.get("solution") or problem.get("answer") or problem.get("expected") or "")

    def create_run(
        self,
        problem: Dict[str, Any],
        final_answer: str,
        extracted_answer: float,
        score: float,
    ) -> Run:
        """Create a LangSmith run for evaluation"""
        import uuid
        question_text = self._get_problem_text(problem)
        expected_text = self._get_solution_text(problem)

        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"question": question_text},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": expected_text,
                "score": score,
                "passed": score == 1.0,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )

    async def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.

        Args:
            problem: The (normalized) problem dictionary with standard keys "problem" and "solution"
                     (legacy raw keys "question"/"answer" are also supported).
            run_result: The result from running the agent system

        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")

        expected_text = self._get_solution_text(problem)

        # Extract numbers
        expected_number = self.extract_number(expected_text)
        predicted_number = self.extract_number(final_answer)

        # Calculate score
        score, extracted_answer = self.calculate_score(expected_number, predicted_number)

        # Create LangSmith run
        run = self.create_run(problem, final_answer, extracted_answer, score)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)

        # Return evaluation results
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "run_evaluation": run_evaluation,
        }

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)
