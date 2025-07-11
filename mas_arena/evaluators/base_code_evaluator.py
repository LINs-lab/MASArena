"""
Base Code Evaluator

This module provides a base class for code evaluation tasks, extending the base evaluator
with code-specific functionality.
"""

import re
import time
import logging
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Union

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from mas_arena.evaluators.base_evaluator import BaseEvaluator


class BaseCodeEvaluator(BaseEvaluator):
    """
    Base class for code evaluation tasks.
    Extends BaseEvaluator with specific functionality for code generation and testing.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()
        
        # Create log directory if it doesn't exist
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            filename=f"{self.log_path}/evaluator.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.SUCCESS = 0
        self.FAILURE = 1
        self.TIMEOUT = 2

    def extract_code(self, text: str) -> str:
        """
        Extract Python code from text in several fall-back steps:
        1. Code under a "## Validated Code" heading
        2. First ```python fenced block
        3. The entire text as fallback
        """
        # Try to find validated code section
        validated = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.I)
        if validated:
            return validated.group(1).strip()

        # Try to find any python code block
        fenced = re.search(r"```python\s*([\s\S]*?)```", text, re.I)
        if fenced:
            return fenced.group(1).strip()

        return text.strip()

    def prepare_task(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw dataset input into a standardized structure.
        This method provides a default implementation for code tasks.
        Subclasses can override for dataset-specific processing.
        """
        prompt = raw.get("prompt") or raw.get("problem", "")
        task_id = raw.get("id", "unknown")
        entry_point = raw.get("entry_point", "")

        # Extract function signature
        sig = re.search(r"def\s+(\w+)\((.*?)\)\s*(->\s*[^:]*)?:", prompt)
        params = sig.group(2) if sig else ""
        function_signature = f"def {entry_point}({params})"

        # Extract docstring
        doc_match = re.search(r'"""([\s\S]*?)"""', prompt, re.DOTALL)
        docstring = doc_match.group(1).strip() if doc_match else ""

        # Extract examples and constraints
        examples = re.findall(r">>>(.+)", prompt)
        constraints = re.findall(r"Constraints?:([\s\S]*?)(?=\n\s*\n|$)", docstring, re.DOTALL)

        return {
            "id": task_id,
            "type": "code_generation",
            "description": docstring or prompt.strip()[:120] + "...",
            "requirements": [f"Implement `{entry_point}` function"],
            "constraints": [c.strip() for c in constraints[0].splitlines()] if constraints else [],
            "examples": examples,
            "entry_point": entry_point,
            "function_signature": function_signature,
            "test": raw.get("test", ""),
        }

    def create_run(
        self,
        problem: Dict[str, Any],
        final_answer: str,
        extracted_answer: str,
        score: float,
        message: str,
    ) -> Run:
        """Create a LangSmith Run object for evaluation tracking."""
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"], "task_id": problem["id"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem.get("solution") or problem.get("test", ""),
                "score": score,
                "message": message,
                "passed": score == 1.0,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            trace_id=str(uuid.uuid4()),
        )

    @abstractmethod
    def check_solution(self, code: str, test: str, entry_point: str, **kwargs) -> Tuple[bool, str]:
        """
        Check if the solution is correct.
        Must be implemented by specific evaluators.
        """
        pass

    def verify_answer(self, prediction: str, reference: Union[str, Dict[str, Any]]) -> bool:
        """
        Implementation of BaseEvaluator's verify_answer for code tasks.
        For code tasks, this typically means running tests.
        """
        if isinstance(reference, dict):
            test = reference.get("test", "")
            entry_point = reference.get("entry_point", "")
        else:
            test = str(reference)
            entry_point = ""  # Should be provided in kwargs or extracted from code

        passed, _ = self.check_solution(prediction, test, entry_point)
        return passed

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main evaluation entry point.
        Extends BaseEvaluator's evaluate with code-specific processing.
        """
        final_answer = run_result.get("final_answer", "")
        extracted_answer = self.extract_code(final_answer)

        passed, msg = self.check_solution(
            extracted_answer,
            problem["test"],
            problem["entry_point"],
        )
        score = 1.0 if passed else 0.0

        run = self.create_run(problem, final_answer, extracted_answer, score, msg)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)

        result = {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "message": msg,
            "run_evaluation": run_evaluation,
        }

        # Save results using parent class method
        self.save_results([result])

        return result 