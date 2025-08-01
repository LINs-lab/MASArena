"""
GAIA Evaluator
"""

from typing import Dict, Any, Optional, List
import sys
import re
import string

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark


def _get_string_value(data: Any) -> str:
    """
    Attempts to extract a string representation from various data types.

    Args:
        data (Any): Input data of any type.

    Returns:
        str: The extracted string value or an empty string if extraction fails.
    """
    if isinstance(data, str):
        return data
    elif hasattr(data, 'content') and isinstance(getattr(data, 'content'), str):
        return getattr(data, 'content')
    elif isinstance(data, dict):
        for key in ["text", "answer", "content", "final_answer", "output"]:
            if key in data and isinstance(data[key], str):
                return data[key]
        print(f"GaiaEvaluator Warning: Received a dict but could not extract a string value from known keys: {data}",
              file=sys.stderr)
        return str(data)
    elif data is None:
        return ""
    else:
        print(f"GaiaEvaluator Warning: Unexpected data type for answer extraction: {type(data)}. Converting to string.",
              file=sys.stderr)
        return str(data)


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")

def split_string(s: str, char_list: Optional[List[str]] = None) -> list[str]:
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)

def normalize_str(input_str, remove_punct=True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    try:
        if is_float(ground_truth):
            normalized_answer = normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)

        elif any(char in ground_truth for char in [",", ";"]):
            gt_elems = split_string(ground_truth)
            ma_elems = split_string(model_answer)

            if len(gt_elems) != len(ma_elems):
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = normalize_str(ma_elem, remove_punct=False)
                    gt_elem = normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            ma_elem = normalize_str(model_answer)
            gt_elem = normalize_str(ground_truth)
            return ma_elem == gt_elem
    except Exception as e:
        return False

@register_benchmark(
    name="gaia",
    normalization_keys={
        "id": "task_id",
        "problem": "Question",
        "solution": "Final answer",
        "files": "file_name",
        "level": "Level",
    }
)
class GaiaEvaluator(BaseEvaluator):
    """
    Evaluator for the GAIA mas_arena.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.

        Args:
            name (str): Name of the evaluator.
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.name = name
        self.config = config

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        model_answer = _get_string_value(run_result.get("final_answer"))
        ground_truth = _get_string_value(problem.get("solution"))
        match = re.search(r"<answer>(.*?)</answer>", model_answer)
        if match:
            answer = match.group(1)

            correct = question_scorer(answer, ground_truth)

            return {
                "score": 1.0 if correct else 0.0,
                 "prediction": answer,
                 "extracted_answer": answer,
                 "expected": ground_truth,
                 "is_correct": correct,
            }
        return {
            "score": 0,
            "prediction": model_answer,
            "extracted_answer": None,
            "expected": ground_truth,
            "is_correct": False,
        }