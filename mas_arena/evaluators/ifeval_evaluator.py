"""
IFEval Evaluator – revised to follow the official scoring more closely
"""

import collections
from typing import Dict, Any

from langsmith.evaluation import RunEvaluator

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
from mas_arena.evaluators.utils.ifeval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose,
    InputExample,
    OutputExample,
)


@register_benchmark(
    name="ifeval",
    normalization_keys={
        "id": "key",
        "solution": "instruction_id_list",
        "problem": "prompt",
        "instruction_id_list": "instruction_id_list",
        "kwargs": "kwargs",
    },
)
class IFEvalEvaluator(BaseEvaluator):
    """Evaluates LLM outputs on IFEval tasks (strict & loose modes)."""

    def __init__(self, name: str = "ifeval", config: Dict[str, Any] | None = None):
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def preprocess_input(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardizes the input dictionary from the Runner, extracting fields needed for the agent.
        """
        instr_ids = problem.get("instruction_id_list", problem.get("instruction_ids", []))
        kwargs_ = problem.get("kwargs", [])

        input_example = InputExample(
            key=problem.get("id", problem.get("key", 0)),
            instruction_id_list=instr_ids,
            prompt=problem["problem"],  # (Runner maps to 'problem')
            kwargs=kwargs_,
        )

        return {
            "problem": input_example.prompt,  # Content sent to the agent
            "instruction_id_list": input_example.instruction_id_list,
            "kwargs": input_example.kwargs,
            "key": input_example.key,
            "original_problem": problem,  # Retrieved during evaluation
        }

    @staticmethod
    def _aggregate_metrics(output: OutputExample) -> Dict[str, Any]:
        """
        Converts OutputExample into usable metrics:
        - prompt_followed: bool
        - instruction_accuracy: per-instruction accuracy
        - tier0 / tier1: statistics for tier-0 (prefix) and tier-1 (full ID)
        """
        follow_list = output.follow_instruction_list
        instr_ids = output.instruction_id_list

        # Prompt-level
        prompt_followed = all(follow_list)

        # Instruction-level
        total = len(instr_ids)
        correct = sum(follow_list)
        instr_acc = correct / total if total else 0

        # Tier-0 (prefix) and Tier-1 (full ID)
        tier0 = collections.defaultdict(lambda: {"total": 0, "correct": 0})
        tier1 = collections.defaultdict(lambda: {"total": 0, "correct": 0})

        for iid, ok in zip(instr_ids, follow_list):
            tier0_id = iid.split(":")[0]
            tier0[tier0_id]["total"] += 1
            tier1[iid]["total"] += 1
            if ok:
                tier0[tier0_id]["correct"] += 1
                tier1[iid]["correct"] += 1

        def _ratio(d):  # Helper function
            return {
                k: {
                    "accuracy": v["correct"] / v["total"] if v["total"] else 0,
                    "correct": v["correct"],
                    "total": v["total"],
                }
                for k, v in d.items()
            }

        return {
            "prompt_followed": prompt_followed,
            "instruction_accuracy": instr_acc,
            "instruction_correct": correct,
            "instruction_total": total,
            "instruction_results": follow_list,
            "tier0_accuracies": _ratio(tier0),
            "tier1_accuracies": _ratio(tier1),
        }

    async def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single sample in both strict and loose modes.
        The score is based on whether the prompt is fully followed in strict mode.
        """
        # 1. Retrieve and clean the model's answer (remove BOM/whitespace)
        # The 'final_ans' variable here represents the model's generated output,
        # which corresponds to the "Predicted" value in evaluation logs.
        # If 'run_result' (from the agent/model) does not contain a 'final_answer' key,
        # or if its value is empty, 'final_ans' will default to an empty string.
        # This means an empty "Predicted" field in logs indicates an empty or missing
        # 'final_answer' from the agent's execution.
        final_ans = run_result.get("final_answer", "")
        try:
            final_ans = final_ans.encode("utf-8").decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            final_ans = final_ans.strip()

        # 2. Restore InputExample
        original = problem.get("original_problem", problem)
        instr_ids = original.get("instruction_id_list", [])
        kwargs_ = original.get("kwargs", [])

        inp = InputExample(
            key=original.get("id", original.get("key", 0)),
            instruction_id_list=instr_ids,
            prompt=original["problem"],
            kwargs=kwargs_,
        )

        # 3. Call official evaluation functions
        mapping = {inp.prompt: final_ans}
        strict_out = test_instruction_following_strict(inp, mapping)
        loose_out = test_instruction_following_loose(inp, mapping)

        strict_metrics = self._aggregate_metrics(strict_out)
        loose_metrics = self._aggregate_metrics(loose_out)

        # 4. Main score: prompt-level strict
        score = 1.0 if strict_metrics["prompt_followed"] else 0.0

        return {
            "final_answer": final_ans,
            "extracted_answer": (
                final_ans[:100] + "..." if len(final_ans) > 100 else final_ans
            ),  # For benchmark_runner.py compatibility
            "score": score,
            "details": {
                "strict_evaluation": strict_metrics,
                "loose_evaluation": loose_metrics,
            },
        }
