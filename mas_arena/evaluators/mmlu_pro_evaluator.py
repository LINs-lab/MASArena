#!/usr/bin/env python3
"""
MMLU Professional Evaluator

This module provides an evaluator for the MMLU Professional mas_arena.
It evaluates agent performance by exact matching of answers.
"""

import json
from typing import Dict, Any, List
import re

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils import extract_answer_simple_tags


@register_benchmark(
    name="mmlu_pro",
    normalization_keys={
        "id": "id",
        "problem": "question",
        "solution": "answer",
    }
)
class MMLU_ProEvaluator(BaseEvaluator):
    """
    Evaluator for the MMLU Professional mas_arena.
    
    This evaluator assesses agent performance on the MMLU_pro dataset
    using exact matching of answers (A, B, C, etc.).
    """
    
    def __init__(self, name="mmlu_pro", config=None):
        """
        Initialize the MMLU Professional evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration dictionary containing:
                - data_path: Path to the MMLU_pro dataset
                - log_path: Path to save evaluation logs
        """
        super().__init__(name, config or {})
        
        # Weight for exact match score is always 1.0 as it's the only metric
        self.exact_match_weight = 1.0
        
        # Load the dataset
        self._load_dataset()
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def _load_dataset(self):
        """Load the MMLU_pro dataset."""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.dataset = [json.loads(line) for line in f]
            self.logger.info(f"Loaded {len(self.dataset)} problems from {self.data_path}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            self.dataset = []
    
    def check_exact_match(self, reference: str, candidate: str) -> float:
        """
        Check if the candidate exactly matches the reference (case-insensitive).
        
        Args:
            reference: Reference answer (e.g., 'A', 'B', 'C', etc.)
            candidate: Candidate answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Clean and normalize both answers
        ref_clean = reference.strip().upper()
        cand_clean = candidate.strip().upper()
        
        # Check for exact match
        if cand_clean == ref_clean:
            return 1.0
        
        # Check if candidate is an index (e.g., "1", "2", "3") converted to letter
        try:
            if cand_clean.isdigit():
                cand_index = int(cand_clean) - 1
                cand_letter = chr(ord('A') + cand_index)
                if cand_letter == ref_clean:
                    return 1.0
        except Exception:
            pass
            
        return 0.0
    
    def get_correct_answer_text(self, problem: Dict[str, Any]) -> str:
        """
        Get the correct answer text from the problem.
        
        Args:
            problem: Problem dictionary
            
        Returns:
            Correct answer text
        """
        options = problem.get("options", [])
        answer_index = problem.get("answer_index")
        answer_letter = problem.get("answer")
        
        # If options are available and there's a valid answer index
        if options and isinstance(answer_index, int) and 0 <= answer_index < len(options):
            return options[answer_index]
        
        # If answer letter and options are available
        if answer_letter and options:
            try:
                # Convert letter to index (A=0, B=1, etc.)
                idx = ord(answer_letter.upper()) - ord('A')
                if 0 <= idx < len(options):
                    return options[idx]
            except Exception:
                pass
        
        # If we can't get the answer text, just return the answer letter/index
        return str(answer_letter if answer_letter else answer_index)
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer from agent response.
        
        Args:
            response: Complete response text from agent
            
        Returns:
            Extracted answer letter
        """
        return extract_answer_simple_tags(response)
    
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an agent's solution to a MMLU_pro problem.
        
        Args:
            problem: Problem dictionary containing:
                - question: Problem text (with options)
                - answer: Correct answer (letter)
                - answer_index: Index of correct answer (optional)
            run_result: Results from agent's execution, containing:
                - final_answer: Agent's final answer text
                - messages: Agent's message history
            
        Returns:
            Evaluation results
        """
        final_answer = run_result.get("final_answer", "")
        reference_letter = problem.get("solution", "")
        
        # Extract the final letter from the agent's response
        extracted_answer = self.extract_answer_from_response(final_answer)
        
        # Calculate exact match score (letter-based)
        score = self.check_exact_match(reference_letter, extracted_answer)
        
        # Record evaluation results
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
        }
    
    def verify_answer(self, prediction: str, reference: Dict[str, Any]) -> bool:
        """
        Verify if the prediction is correct according to the reference.
        For MMLU_pro, we consider an exact match on the answer letter as correct.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer dictionary or string
            
        Returns:
            True if the answer is correct, False otherwise
        """
        reference_letter = reference.get("answer", "") if isinstance(reference, dict) else str(reference)
        
        exact_match = self.check_exact_match(reference_letter, prediction)
        return exact_match >= 0.9
    
    def batch_evaluate(self, problems: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of problems.
        
        Args:
            problems: List of problem dictionaries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        # Evaluate each problem individually
        for i, problem in enumerate(problems):
            problem_id = problem.get("id", problem.get("question_id", f"unknown_{i}"))
            reference_letter = problem.get("solution", problem.get("answer", ""))
            reference_text = self.get_correct_answer_text(problem)
            response = problem.get("response", "")
            
            # Calculate exact match score
            exact_match = self.check_exact_match(reference_letter, response)
            
            # Record results
            result = {
                "problem_id": problem_id,
                "exact_match": exact_match,
                "combined_score": exact_match,  # Combined score is just the exact match
                "extracted_answer": response,
                "reference_answer": reference_letter,
                "reference_text": reference_text,
                "execution_time_ms": 0,  # Will be updated by the benchmark runner
                "math_score": 1.0 if exact_match >= 0.9 else 0.0  # For compatibility with benchmark runner
            }
            
            results.append(result)
            
            # Log the results
            self.logger.info(f"Problem {problem_id}: Exact={exact_match:.1f}, Combined={exact_match:.4f}")
        
        return results
    
    def format_options(self, options: List[str]) -> str:
        """
        Format options list into a readable string.
        
        Args:
            options: List of options
            
        Returns:
            Formatted options string
        """
        if not options:
            return ""
        
        formatted = []
        for i, option in enumerate(options):
            letter = chr(ord('A') + i)
            formatted.append(f"{letter}. {option}")
        
        return "\n".join(formatted)


if __name__ == "__main__":
    # Test the evaluator with a simple example
    evaluator = MMLU_ProEvaluator()
    
    test_problem = {
        "question_id": 70,
        "question": "Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.",
        "options": [
            "Safe practices, Fear, Jealousy, Trivial",
            "Unsafe practices, Distress, Joy, Trivial",
            "Safe practices, Wants, Jealousy, Trivial",
            "Safe practices, Distress, Fear, Trivial",
            "Unsafe practices, Wants, Jealousy, Serious",
            "Safe practices, Distress, Jealousy, Serious",
            "Safe practices, Wants, Fear, Serious",
            "Unsafe practices, Wants, Fear, Trivial",
            "Unsafe practices, Distress, Fear, Serious"
        ],
        "answer": "I",
        "answer_index": 8,
        "response": "I"
    }
    
    result = evaluator.evaluate(test_problem)
    print(test_problem)
    print(result)
    print(json.dumps(result, indent=2))
