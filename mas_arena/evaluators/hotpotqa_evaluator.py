"""
HotpotQA Evaluator

This module provides a standalone evaluator for HotpotQA (Multi-hop Question Answering) problems.
"""

import time
from typing import Dict, Any, Tuple

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils import extract_answer_generic, calculate_f1_score, normalize_answer


@register_benchmark(
    name="hotpotqa",
    normalization_keys={
        "id": "_id",
        "problem": "question",
        "context": "context",
        "solution": "answer",
    }
)
class HotpotQAEvaluator(BaseEvaluator):
    """Evaluator for HotpotQA problems"""
    
    def __init__(self, name: str = "hotpotqa", config: Dict[str, Any] = None):
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
        
    def normalize_answer(self, s: str) -> str:
        """
        Normalize answers for evaluation.
        
        Args:
            s: The answer string to normalize
            
        Returns:
            Normalized answer string
        """
        return normalize_answer(s)
        
    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        
        Args:
            ground_truth: The ground truth answer
            prediction: The predicted answer
            
        Returns:
            Tuple of (f1_score, extracted_answer)
        """
        extracted_answer = self.extract_answer(prediction)
        f1_score = calculate_f1_score(ground_truth, extracted_answer)
        return f1_score, extracted_answer
        
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: float) -> Run:
        """Create a LangSmith run for evaluation"""
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={
                "question": problem["problem"],
                "context": problem["context"]
            },
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["solution"],
                "score": score,
                "passed": score >= 0.3,  # HotpotQA uses 0.3 as threshold
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
        
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.
        
        Args:
            problem: The problem dictionary with "question", "context", and "answer" keys
            run_result: The result from running the agent system
            
        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")
        
        # Process context
        paragraphs = [item[1] for item in problem["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        
        # Calculate score
        score, extracted_answer = self.calculate_score(problem["solution"], final_answer)
        
        # # Create LangSmith run
        # run = self.create_run(problem, final_answer, extracted_answer, score)
        # run_evaluation = self.run_evaluator.evaluate_run(run=run)
        
        # Log mismatch if score is too low
        if score < 0.3:
            with open(f"{self.log_path}/mismatches.log", "a") as f:
                f.write(f"\nQuestion: {problem['problem']}\n")
                f.write(f"Context: {context_str}\n")
                f.write(f"Expected: {problem['solution']}\n")
                f.write(f"Predicted: {final_answer}\n")
                f.write(f"Score: {score}\n")
        
        # Final score: 1.0 if score >= 0.3, else use the score directly
        final_score = 1 if score >= 0.3 else score
        
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": final_score,
            "context": context_str
        }