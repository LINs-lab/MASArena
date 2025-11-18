"""
HumanEval Evaluator
"""

import asyncio
import time
import re
import traceback
import random
from threading import Thread
from typing import Dict, Any, Tuple, Callable, List, Optional

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from mas_arena.evaluators.base_code_evaluator import BaseCodeEvaluator
from mas_arena.evaluators.utils.sanitize import sanitize, code_extract
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils.timeout import run_with_timeout, TimeoutError


@register_benchmark(
    name="humaneval",
    normalization_keys={
        "id": "task_id",
        "problem": "prompt",
        "solution": "canonical_solution",
        "test": "test",
        "entry_point": "entry_point",
    },
)
class HumanEvalEvaluator(BaseCodeEvaluator):
    """Evaluator for HumanEval problems"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)

        # LangSmith evaluator for packaging the evaluation run
        self.run_evaluator = RunEvaluator()

    def extract_code(self, text: str) -> str:
        """
        Extract Python code from *text* in several fall-back steps, in order of preference:

        1. A QA Engineer section marked "## Validated Code".
        2. Any generic ```python fenced block.
        3. A bare function-definition-like snippet.
        4. As a last resort, use *sanitize* / *code_extract* helpers.
        """
        self.logger.info(f"Extracting code… snippet: {text[:100]}")

        # ① "## Validated Code" block
        qa_match = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if qa_match:
            code = qa_match.group(1).strip()
            self.logger.info("Found code in 'Validated Code' section.")
            return code

        # ② Any fenced ````python``` block
        block_match = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
        if block_match:
            code = block_match.group(1).strip()
            self.logger.info("Found code in generic fenced block.")
            return code

        # ③ A function-shaped snippet (best effort)
        fn_match = re.search(r"(def\s+\w+\s*\(.*?\):[\s\S]*?)(?=\n{2,}|\Z)", text)
        if fn_match:
            code = fn_match.group(1).strip()
            self.logger.info("Found code by function-like regex.")
            return code

        # ④ Fallback extraction
        try:
            code = sanitize(text)
            self.logger.info("Code extracted via sanitize().")
            return code
        except Exception:
            code = code_extract(text)
            self.logger.info("Code extracted via fallback code_extract().")
            return code

    def check_solution(self, code: str, test: str, entry_point: str) -> Tuple[bool, str]:
        """
        Compile *code*, execute the official *test* (which in turn calls ``check(candidate)``),
        and return ``(passed, message)``.

        Passing criterion: **all assertions inside the test must complete without raising**.
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
            if self.config.get("verbose", False):
                self.logger.error(traceback.format_exc())

        self.logger.error(f"Check failed: {msg}")
        return False, msg

    def calculate_score(self, test_code: str, prediction: str, entry_point: str) -> Tuple[float, str, str]:
        """
        Return ``(score, code_used_for_test, message)`` where *score* is 1.0 on success, 0.0 otherwise.
        """
        passed, message = self.check_solution(prediction, test_code, entry_point)
        return (1.0 if passed else 0.0), prediction, message

    def create_run(
        self,
        problem: Dict[str, Any],
        final_answer: str,
        extracted_answer: str,
        score: float,
        message: str,
    ) -> Run:
        """Package the evaluation result as a ``Run`` object for LangSmith."""
        import uuid

        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"], "task_id": problem["id"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["test"],
                "score": score,
                "message": message,
                "passed": score == 1.0,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            trace_id=str(uuid.uuid4()),
        )

    async def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point – keeps the outer interface unchanged.
        Consumes one *problem* dict and the model *run_result*, returns a detailed evaluation dict.
        """
        final_answer = run_result.get("final_answer", "")
        extracted_answer = final_answer
        if not run_result.get("extracted"):
            extracted_answer = self.extract_code(final_answer)

        score, extracted_answer, message = self.calculate_score(
            problem["test"], extracted_answer, problem["entry_point"]
        )

        run = self.create_run(problem, final_answer, extracted_answer, score, message)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)

        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "message": message,
            "run_evaluation": run_evaluation,
        }

    async def async_evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluate_result = await asyncio.to_thread(self.evaluate, run_result=run_result, problem=problem)
        return evaluate_result

    def extract_test_cases_with_entry_point(self, entry_point: str):
        """
        Extract test cases with the given entry point.
        """

        hardcoded_cases = {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        }
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]

        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]

        return None
