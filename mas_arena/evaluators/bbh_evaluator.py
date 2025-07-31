"""
BBH Evaluator

This module provides a standalone evaluator for Big-Bench Hard (BBH) problems.
"""

import re
import time
from typing import Dict, Any, Tuple
import uuid

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils import extract_answer_generic


@register_benchmark(
    name="bbh",
    normalization_keys={
        "id": "task_id",
        "problem": "input",
        "solution": "target",
    }
)
class BBHEvaluator(BaseEvaluator):
    """
    Evaluator for Big-Bench Hard (BBH) problems.

    This evaluator handles diverse BBH tasks, extracting answers from model responses and comparing them with expected outputs.
    """

    def __init__(self, name: str = "bbh", config: Dict[str, Any] = None):
        """
        Initialize the BBH Evaluator.

        Args:
            name: Name of the evaluator (default: "bbh")
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from model output text, expecting '<answer>...</answer>' tags first.

        Args:
            text: The model's output text

        Returns:
            The extracted answer (e.g., "(A)", "True", "] >")
        """
        return extract_answer_generic(text)

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize the answer to handle minor formatting variations.

        Args:
            answer: The extracted answer

        Returns:
            Normalized answer
        """
        answer = answer.strip()
        # Normalize multiple-choice answers (e.g., "A" to "(A)", "[A]" to "(A)")
        if re.match(r"^[A-Z]$", answer):
            return f"({answer})"
        if re.match(r"^\[[A-Z]\]$", answer):
            return f"({answer[1]})"
        # Normalize sequence answers by collapsing extra spaces
        if re.match(r"^[>\]\}\)\[]+\s*[>\]\}\)\[]*\s*$", answer):
            return " ".join(answer.split())
        # Normalize boolean answers to title case
        if answer.lower() in ["true", "false"]:
            return answer.title()
        return answer

    def calculate_score(self, expected_output: str, prediction: str, problem_id: str = "") -> Tuple[float, str, str]:
        """
        Calculate score by comparing expected and predicted answers.

        Args:
            expected_output: The expected answer (solution)
            prediction: The model's raw prediction
            problem_id: The ID of the problem (used to identify word sorting tasks)

        Returns:
            Tuple of (score, extracted_answer, message) where score is 1.0 for correct, 0.0 for incorrect
        """
        extracted_answer = self.extract_answer(prediction)
        normalized_answer = self.normalize_answer(extracted_answer)
        normalized_expected = self.normalize_answer(expected_output.strip())

        # Check if this is a word sorting task
        is_word_sorting = "word_sorting" in problem_id.lower()

        if is_word_sorting:
            # For word sorting tasks, compare words as sets (order doesn't matter)
            predicted_words = set(normalized_answer.lower().split())
            expected_words = set(normalized_expected.lower().split())
            if predicted_words == expected_words:
                return 1.0, extracted_answer, "Correct"
            else:
                error_message = (
                    f"Incorrect word sorting: Expected words '{normalized_expected}', got '{normalized_answer}'"
                )
                with open(f"{self.log_path}/error.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")
                return 0.0, extracted_answer, error_message
        else:
            # For other tasks, use exact string comparison
            if normalized_answer.lower() == normalized_expected.lower():
                return 1.0, extracted_answer, "Correct"

            error_message = f"Incorrect: Expected '{normalized_expected}', got '{normalized_answer}'"
            with open(f"{self.log_path}/error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")
            return 0.0, extracted_answer, error_message

    def create_run(
        self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: float, message: str
    ) -> Run:
        """
        Create a LangSmith run for evaluation.

        Args:
            problem: The problem dictionary
            final_answer: The raw final answer from the model
            extracted_answer: The extracted answer
            score: The score (0.0 or 1.0)
            message: Evaluation message

        Returns:
            A LangSmith Run object
        """
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"], "task_id": problem["id"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["solution"],
                "score": score,
                "message": message,
                "passed": score == 1.0,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a BBH problem given the agent's response.

        Args:
            problem: The problem dictionary with "problem" and "solution" keys
            run_result: The result from running the agent system, including the final answer

        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")
        if not final_answer:
            final_answer = run_result.get("content", "") if isinstance(run_result, dict) else str(run_result)

        # Calculate score using the solution key, passing problem_id
        score, extracted_answer, message = self.calculate_score(problem["solution"], final_answer, problem["id"])

        # Create LangSmith run
        run = self.create_run(problem, final_answer, extracted_answer, score, message)
        self.run_evaluator.evaluate_run(run=run)

        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "message": message,
        }


