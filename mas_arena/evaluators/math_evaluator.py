"""
Math Evaluator

This module provides a standalone evaluator for mathematical problems.
"""

import re
import time
from typing import Dict, Any, Tuple
from pathlib import Path
from math import isclose

from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils.math_equal import calculate_score
# change

@register_benchmark(
    name="math",
    normalization_keys={
        "id": "id",
        "problem": "problem",
        "solution": "solution",
    }
)

class MathEvaluator(BaseEvaluator):
    """
    Math Evaluator for evaluating math problems.
    
    This evaluator extracts answers from model responses and compares them with expected solutions
    using various mathematical equivalence techniques.
    """
    
    SUPPORTS_CONCURRENCY = False
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the Math Evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration parameters
        """
        super().__init__(name, config)
        self.evaluate_type = 0 # 0: simple, 1: math_equal
        # Create log directory if it doesn't exist
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize run evaluator for LangSmith compatibility
        self.run_evaluator = RunEvaluator()
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)
    
    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from model output text, looking for boxed answers or final statements.
        
        Args:
            text: The model's output text
            
        Returns:
            The extracted answer
        """
        # Look for LaTeX boxed answers first
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()
        
        # If no boxed answer, try to extract the final conclusion
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""
    
    def simple_calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        """
        Calculate a score by comparing the expected and predicted answers.
        
        Args:
            expected_output: The expected answer (solution)
            prediction: The model's prediction
            
        Returns:
            Tuple of (score, extracted_answer) where score is 1 for correct, 0 for incorrect
        """
        extracted_expected = self.extract_answer(expected_output)
        extracted_prediction = self.extract_answer(prediction)
        
        if self.math_equal(extracted_prediction, extracted_expected):
            return 1, extracted_prediction
        else:
            return 0, extracted_prediction
    
    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """
        Check if two mathematical expressions are equivalent.
        
        Args:
            prediction: The predicted answer
            reference: The reference answer
            
        Returns:
            True if the expressions are equivalent, False otherwise
        """
        # Direct string comparison
        if str(prediction) == str(reference):
            return True
        
        # Numeric comparison
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction_val = self.parse_digits(prediction)
                reference_val = self.parse_digits(reference)
                return isclose(prediction_val, reference_val, abs_tol=1e-3)
        except ValueError:
            pass
        
        # Symbolic comparison
        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass
        
        return False
    
    def is_digit(self, num):
        """Check if a string can be parsed as a number"""
        return self.parse_digits(num) is not None
    
    def parse_digits(self, num):
        """Parse a string as a number, handling percentage and commas"""
        num = str(num).replace(",", "")
        try:
            return float(num)
        except ValueError:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except ValueError:
                    pass
        return None
    
    def symbolic_equal(self, a, b):
        """Check symbolic equality using SymPy"""
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except Exception:
                    pass
            return s
        
        a = _parse(a)
        b = _parse(b)
        
        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass
        
        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except Exception:
            pass
            
        return False
    
    def extract_final_answer(self, messages: list) -> str:
        """
        Extract the final answer from a list of messages.
        
        Args:
            messages: List of messages from the agent conversation
            
        Returns:
            The extracted final answer
        """
        final_answer = ""
        
        if not messages:
            return final_answer
            
        last_msg = messages[-1]
        if isinstance(last_msg, tuple) and len(last_msg) > 1:
            final_answer = last_msg[1]
        elif hasattr(last_msg, "content"):
            final_answer = last_msg.content
        elif isinstance(last_msg, dict) and "content" in last_msg:
            final_answer = last_msg["content"]
        
        return final_answer
    
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: int) -> Run:
        """
        Create a LangSmith run for evaluation.
        
        Args:
            problem: The problem dictionary
            final_answer: The raw final answer from the model
            extracted_answer: The extracted answer
            score: The score (0 or 1)
            
        Returns:
            A LangSmith Run object
        """
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["solution"],
                "score": score,
                "passed": score == 1,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
    
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
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
        final_answer = self.extract_final_answer(all_messages)
        
        if self.evaluate_type == 0:
            # Use the new calculate_score method
            score, extracted_answer = self.simple_calculate_score(problem["solution"], final_answer)
        else:
            # Use the new calculate_score method
            score, extracted_answer = calculate_score(problem["solution"], final_answer)
        
        # # Create LangSmith run for evaluation
        # run = self.create_run(problem, final_answer, extracted_answer, score)
        # self.run_evaluator.evaluate_run(run=run)
        
        # Return evaluation results
        return {
            "final_answer": final_answer,
            "score": score,
            "extracted_answer": extracted_answer
        }
