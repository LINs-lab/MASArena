"""
Base Code Evaluator

This module provides a base class for code evaluation tasks, extending the base evaluator
with code-specific functionality.
"""

import re
import time
import logging
import uuid
import random
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Union, List, Optional

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.utils.timeout import run_with_timeout, TimeoutError


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

    def _load_data(self):
        self._train_data = []
        self._dev_data = self._load_dateset_from_path(f"data/{self.name}_validate.jsonl")
        self._test_data = self._load_dateset_from_path(f"data/{self.name}_test.jsonl")
        self._test_cases = self._load_dateset_from_path(f"data/{self.name}_public_test.jsonl")

    def _get_data(
        self,
        data: List[dict],
        indices: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[dict]:
        if indices is None:
            indices = list(range(len(data)))
        if sample_size is not None:
            if seed is not None:
                random.seed(seed)
            indices = random.sample(indices, k=min(sample_size, len(indices)))
        return_data = [data[idx] for idx in indices]
        return return_data

    def get_train_data(
        self,
        indices: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[dict]:
        if self._train_data is None:
            print(f"Train data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return []
        return self._get_data(data=self._train_data, indices=indices, sample_size=sample_size, seed=seed)

    def get_dev_data(
        self,
        indices: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[dict]:
        if self._dev_data is None:
            print(f"Dev data for benchmark {type(self).__name__} is not loaded or None. Return an empty list.")
            return []
        return self._get_data(data=self._dev_data, indices=indices, sample_size=sample_size, seed=seed)

    def get_test_data(
        self,
        indices: Optional[List[int]] = None,
        sample_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[dict]:
        if self._test_data is None:
            print(f"Test data for evaluator {type(self).__name__} is not loaded or None. Return an empty list.")
            return []
        return self._get_data(data=self._test_data, indices=indices, sample_size=sample_size, seed=seed)

    def extract_test_cases_with_entry_point(self, entry_point: str):
        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]
        return None

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
            "constraints": ([c.strip() for c in constraints[0].splitlines()] if constraints else []),
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

    def check_solution(self, code: str, test: str, entry_point: str, **kwargs) -> Tuple[bool, str]:
        """
        Check if the solution is correct.
        Must be implemented by specific evaluators.
        """
        try:
            # Create an isolated namespace
            env: Dict[str, Any] = {}
            # Inject the candidate implementation
            exec(code, env)
            candidate_fn = env[entry_point]
            # Inject and obtain ``check()``
            exec(test, env)
            check_fn = env["check"]
            # If ``check()`` raises, the except block will handle it
            run_with_timeout(check_fn, (candidate_fn,), timeout=60)
            return True, "All tests passed"
        except TimeoutError as te:
            msg = str(te)
        except AssertionError as ae:
            msg = f"Test failed: {ae}"
        except Exception as exc:
            msg = f"Execution error: {exc}"
        return False, msg

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

    async def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
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
