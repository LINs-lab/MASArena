"""
AIME Evaluator

Standalone evaluator for AIME-style math problems.
"""

from typing import Dict, Any
from pathlib import Path

from langsmith.evaluation import RunEvaluator

from mas_arena.evaluators.math_evaluator import MathEvaluator
from mas_arena.evaluators.registry import register_benchmark


@register_benchmark(
    name="aime",
    normalization_keys={
        "problem": "question",
        "solution": "answer",
    },
)
class AIMEEvaluator(MathEvaluator):
    """
    Evaluator for AIME-style math problems.

    This evaluator extracts answers from model responses and compares them with expected solutions.
    For AIME problems, answers are expected to be integers between 000 and 999.
    """

    def __init__(self, name: str = "aime", config: Dict[str, Any] = None):
        """
        Initialize the AIME Evaluator.

        Args:
            name: Name of the evaluator
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.evaluate_type = 0  # 0: simple, 1: math_equal

        # Create log directory if it doesn't exist
        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        # Initialize run evaluator for LangSmith compatibility
        self.run_evaluator = RunEvaluator()

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    async def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.

        Args:
            problem: The problem dictionary with "problem" and "solution" keys
            run_result: The result from running the agent system, including messages

        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer from messages
        all_messages = run_result.get("messages", [])
        final_answer = super().extract_final_answer(all_messages)

        if self.evaluate_type == 0:
            # Use the new calculate_score method
            score, extracted_answer = super().simple_calculate_score(problem["solution"], final_answer)
        else:
            # Use the new calculate_score method
            score, extracted_answer = super().calculate_score(problem["solution"], final_answer)

        # Return evaluation results
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
        }
