"""
HumanEval Evaluator
"""
import asyncio
import time
import re
import traceback
from threading import Thread
from typing import Dict, Any, Tuple, Callable, List, Optional

import numpy as np
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from mas_arena.core.callbacks import timeout, TimeoutException
from mas_arena.core.lcb_utils import estimate_pass_at_k
from mas_arena.evaluators.base_code_evaluator import BaseCodeEvaluator
from mas_arena.evaluators.utils.normalization import normalize_problem_keys
from mas_arena.evaluators.utils.sanitize import sanitize, code_extract
from mas_arena.evaluators.registry import register_benchmark


@register_benchmark(
    name="humaneval",
    normalization_keys={
        "id": "task_id",
        "problem": "prompt",
        "solution": "canonical_solution",
        "test": "test",
        "entry_point": "entry_point",
    }
)
class HumanEvalEvaluator(BaseCodeEvaluator):
    """Evaluator for HumanEval problems"""

    class TimeoutError(Exception):
        """Raised when execution exceeds the allowed time limit."""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config) 

        # LangSmith evaluator for packaging the evaluation run
        self.run_evaluator = RunEvaluator()

    def run_with_timeout(self, func, args, timeout: int = 60):
        """
        Execute ``func(*args)`` in a separate thread
        and abort if it does not finish within *timeout* seconds.
        """
        result: list[Any] = []
        exception: list[BaseException] = []

        def target():
            try:
                result.append(func(*args))
            except BaseException as e:
                exception.append(e)

        thread = Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise self.TimeoutError("Execution timed out")

        if exception:
            raise exception[0]

        return result[0] if result else None

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
            self.run_with_timeout(check_fn, (candidate_fn,), timeout=60)
            return True, "All tests passed"

        except self.TimeoutError as te:
            msg = str(te)
        except AssertionError as ae:
            msg = f"Test failed: {ae}"
        except Exception as exc:
            msg = f"Execution error: {exc}"
            if self.config.get("verbose", False):
                self.logger.error(traceback.format_exc())

        self.logger.error(f"Check failed: {msg}")
        return False, msg

    def calculate_score(
        self, test_code: str, prediction: str, entry_point: str
    ) -> Tuple[float, str, str]:
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

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point – keeps the outer interface unchanged.
        Consumes one *problem* dict and the model *run_result*, returns a detailed evaluation dict.
        """
        if "solution" in run_result and run_result["solution"]:
            final_answer = run_result["solution"]
            extracted_answer = run_result["solution"]
        else:
            final_answer = run_result.get("final_answer", "")
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

    async def async_evaluate(self, graph: Callable, example: Any, i: int = None) -> float:
        # generate solution
        prompt, entry_point = example["prompt"], example["entry_point"]
        solution = await graph(prompt, entry_point)
        from mas_arena.evaluators import BENCHMARKS
        benchmark_config = BENCHMARKS.get(self.name, {})
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(example,key_mapping,i)
        run_result = {"solution":solution}
        metrics = await asyncio.to_thread(self.evaluate, run_result=run_result, problem=normalized_problem)
        return metrics["score"]

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

    def _load_data(self):

        self._train_data = []
        self._dev_data = self._load_dateset_from_path(f"data/{self.name}_validate.jsonl")
        self._test_data = self._load_dateset_from_path(f"data/{self.name}_test.jsonl")
        self._test_cases = self._load_dateset_from_path(f"data/{self.name}_public_test.jsonl")
