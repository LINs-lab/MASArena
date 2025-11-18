"""
MBPP Evaluator
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, Dict, List, Tuple

from mas_arena.evaluators.base_code_evaluator import BaseCodeEvaluator
from mas_arena.evaluators.utils.sanitize import sanitize
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils.timeout import run_with_timeout, TimeoutError


@register_benchmark(
    name="mbpp",
    normalization_keys={
        "id": "task_id",
        "problem": "prompt",
        "solution": "code",
        "test": "test",
        "entry_point": "entry_point",
        "test_imports": "test_imports"
    }
)
class MBPPEvaluator(BaseCodeEvaluator):
    """Evaluator for MBPP code-generation tasks."""

    def __init__(self, name: str = "mbpp", config: Dict[str, Any] | None = None):
        super().__init__(name, config)

    def check_solution(
        self,
        code: str,
        test: str,
        entry_point: str,
        test_imports: List[str] | None = None,
    ) -> Tuple[bool, str]:
        """
        Compile user code, run official MBPP `check()`.

        Returns:
            (passed: bool, message: str)
            `passed` is True iff all assertions succeed within the time-limit.
        """
        try:
            # Remove Markdown, ensure the target function exists
            code_clean = sanitize(code=code, entrypoint=entry_point)

            # Isolated global namespace
            env: Dict[str, Any] = {}
            exec(code_clean, env)

            # Execute additional import statements if provided
            for stmt in test_imports or []:
                exec(stmt, env)

            if entry_point not in env:
                raise ValueError(f"Function `{entry_point}` is missing in submitted code.")

            # Inject and run the official unit tests
            exec(test, env)
            check_fn = env["check"]

            run_with_timeout(check_fn, timeout=15)  # `check()` takes no args
            return True, "All tests passed"

        except TimeoutError as te:
            return False, str(te)
        except AssertionError as ae:
            return False, f"Assertion failed: {ae}"
        except Exception as exc:  # noqa: BLE001
            if self.config.get("verbose"):
                self.logger.error(traceback.format_exc())
            return False, f"Execution error: {exc}"

    async def async_evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        evaluate_result = await asyncio.to_thread(self.evaluate, run_result=run_result, problem=problem)
        return evaluate_result

    def extract_test_cases_with_entry_point(self, entry_point: str):

        hardcoded_cases = {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        }
        if entry_point in hardcoded_cases:
            return hardcoded_cases[entry_point]

        for case in self._test_cases:
            if case["entry_point"] == entry_point:
                return case["test"]

        return None
