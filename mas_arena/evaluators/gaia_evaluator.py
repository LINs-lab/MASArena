"""
GAIA Evaluator
"""

from typing import Dict, Any, Optional, TypedDict
import sys
import re
import os
import json
import asyncio
from functools import lru_cache
from typing_extensions import override
from openai import AsyncOpenAI
from dotenv import load_dotenv

from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark

# Load environment variables
load_dotenv(override = True)


def _get_string_value(data: Any) -> str:
    """Best-effort extraction of a human-readable string from heterogeneous objects."""
    if isinstance(data, str):
        return data
    if hasattr(data, "content") and isinstance(getattr(data, "content"), str):  # e.g. SDK objects
        return getattr(data, "content")
    if isinstance(data, dict):
        for key in ("text", "answer", "content", "final_answer", "output"):
            val = data.get(key)
            if isinstance(val, str):
                return val
        # Fallback: stringify but warn once per distinct type to avoid noise
        print(
            "GaiaEvaluator Warning: Dict without known string keys; falling back to str(data)",
            file=sys.stderr,
        )
        return str(data)
    if data is None:
        return ""
    print(
        f"GaiaEvaluator Warning: Unexpected data type {type(data)!r}; coercing to string.",
        file=sys.stderr,
    )
    return str(data)


@lru_cache(maxsize=4)
def _compile_answer_tag_pattern() -> re.Pattern:
    return re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)


class OpenAIResult(TypedDict, total=False):
    is_correct: bool
    confidence: float
    reasoning: str
    extracted_answer: Optional[str]
    raw_response: str


@register_benchmark(
    name="gaia",
    normalization_keys={
        "id": "task_id",
        "problem": "Question",
        "solution": "Final answer",
        "files": "file_name",
        "level": "Level",
    },
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

    async def _evaluate_with_openai(
        self, problem: str, solution: str, answer: str, model_name: str = "gpt-4o-mini"
    ) -> OpenAIResult:
        """
        Evaluate whether an answer is correct for a given problem using OpenAI API.

        Args:
            problem: The problem statement
            solution: The standard/correct answer
            answer: The answer to evaluate (may be LLM output with answer embedded)
            model_name: OpenAI model to use for evaluation

        Returns:
            Dictionary containing:
            - is_correct: Boolean indicating if the answer is correct
            - confidence: Float between 0-1 indicating confidence in the evaluation
            - reasoning: String explaining the evaluation reasoning
            - extracted_answer: The key answer extracted from the response (if any)
        """
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

        # Create evaluation prompt
        evaluation_prompt = f"""You are an expert evaluator. Your task is to determine whether a given answer is correct for a problem, even if the format differs from the standard solution.

Problem: {problem}

Standard Solution: {solution}

Answer to Evaluate: {answer}

Please evaluate whether the answer is correct. The answer might be:
- Embedded in a longer response (extract the key answer)
- In a different format but mathematically/logically equivalent
- More detailed than the standard solution
- Less detailed but still correct

Respond with a JSON object containing:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your evaluation",
    "extracted_answer": "the key answer you extracted (if any)"
}}

Be generous in your evaluation - if the core answer is correct despite formatting differences, mark it as correct.

If llm fails to provide a valid answer based on existing knowledge, mark it as incorrect.

If the answer is strictly required to be a specific word, then it must match that word exactly. Mark synonyms or similar words as incorrect.
"""

        def _extract_json_block(text: str) -> str:
            """Extract the first plausible JSON object substring from a model response."""
            text = text.strip()
            # Remove fenced code markers if present
            if text.startswith("```"):
                # Split off the first line (``` or ```json)
                parts = text.split("\n", 1)
                text = parts[1] if len(parts) > 1 else ""
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
            # Heuristic: find first '{' and last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return text[start : end + 1]
            return text  # fallback

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert answer evaluator. Always respond with valid JSON.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if not content:
                return OpenAIResult(
                    is_correct=False,
                    confidence=0.0,
                    reasoning="Empty response content from OpenAI",
                    extracted_answer=None,
                )
            raw = content.strip()
            json_candidate = _extract_json_block(raw)
            try:
                parsed: OpenAIResult = json.loads(json_candidate)  # type: ignore
            except json.JSONDecodeError as e:
                return OpenAIResult(
                    is_correct=False,
                    confidence=0.0,
                    reasoning=f"Failed to parse JSON: {e}",
                    extracted_answer=None,
                    raw_response=raw,
                )

            # Normalize fields
            parsed.setdefault("is_correct", False)
            conf = parsed.get("confidence", 0.0) or 0.0
            try:
                conf_f = float(conf)
            except (TypeError, ValueError):
                conf_f = 0.0
            parsed["confidence"] = max(0.0, min(1.0, conf_f))
            parsed.setdefault("reasoning", "")
            return parsed
        except Exception as e:  # network / auth / other
            return OpenAIResult(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Error during OpenAI evaluation: {e}",
                extracted_answer=None,
            )

    async def _evaluate_with_openai_sync(
        self, problem: str, solution: str, answer: str, model_name: str = "gpt-4o-mini"
    ) -> OpenAIResult:
        """Sync wrapper; reuses existing loop if running outside async context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():  # create a dedicated loop to avoid RuntimeError
            temp_loop = asyncio.new_event_loop()
            try:
                return temp_loop.run_until_complete(self._evaluate_with_openai(problem, solution, answer, model_name))
            finally:
                temp_loop.close()
        else:
            return loop.run_until_complete(self._evaluate_with_openai(problem, solution, answer, model_name))

    async def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Get run_result from kwargs, fallback to problem if not provided
        run_result = kwargs.get("run_result", problem)

        model_answer = _get_string_value(run_result.get("final_answer"))
        ground_truth = _get_string_value(problem.get("solution"))

        # Extract <answer>...</answer> (case-insensitive, cached regex)
        match = _compile_answer_tag_pattern().search(model_answer)
        answer = match.group(1) if match else model_answer

        # Get problem statement for OpenAI evaluation
        problem_statement = _get_string_value(problem.get("problem", problem.get("Question", "")))

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "score": 0.0,
                "prediction": answer,
                "extracted_answer": answer,
                "expected": ground_truth,
                "is_correct": False,
                "confidence": 0.0,
                "reasoning": "Missing OPENAI_API_KEY; evaluation aborted.",
                "evaluation_method": "openai_failed",
            }

        model_name = self.config.get("model_name", "gpt-4o-mini")

        attempts = 0
        last_result: OpenAIResult | None = None
        while attempts < 2:  # first attempt + one retry
            attempts += 1
            result = await self._evaluate_with_openai_sync(
                problem=problem_statement,
                solution=ground_truth,
                answer=answer,
                model_name=model_name,
            )
            last_result = result
            # Decide if retry needed: only retry on transport / parse errors (confidence==0 and not is_correct)
            if result.get("is_correct") is True:
                break
            reasoning = (result.get("reasoning") or "").lower()
            if not (("error" in reasoning) or ("fail" in reasoning) or ("parse" in reasoning)):
                # It's a deliberate negative evaluation, no need to retry.
                break

        # Compose final output
        is_correct = bool(last_result.get("is_correct")) if last_result else False
        extracted_answer = (last_result.get("extracted_answer") if last_result else None) or answer
        return {
            "score": 1.0 if is_correct else 0.0,
            "prediction": extracted_answer,
            "extracted_answer": extracted_answer,
            "expected": ground_truth,
            "is_correct": is_correct,
            "confidence": last_result.get("confidence") if last_result else 0.0,
            "reasoning": last_result.get("reasoning") if last_result else "No result",
            "evaluation_method": (
                "openai" if is_correct else ("openai_retry_failed" if attempts == 2 and not is_correct else "openai")
            ),
            "attempts": attempts,
        }
