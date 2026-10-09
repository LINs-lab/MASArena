"""Shared paper-run settings and task-only inputs for workflow adapters."""
import os
from typing import Any

from mas_arena.utils.env import DEFAULT_MODEL_NAME, get_model_name, get_openai_api_base
from mas_arena.utils.openai_compat import normalize_openai_api_base


def public_task(problem: dict[str, Any]) -> dict[str, Any]:
    """Only task inputs, never labels or evaluator metadata, enter inference."""
    return {key: problem[key] for key in ("problem", "id", "files") if key in problem}


def task_text(problem: dict[str, Any]) -> str:
    text = problem["problem"]
    files = problem.get("files") or []
    if isinstance(files, str):
        files = [files]
    if files:
        text += "\n\nAttached files (use the available file tools):\n" + "\n".join(str(path) for path in files)
    return text


def model_name(config: dict[str, Any]) -> str:
    return config.get("model_name") or get_model_name(DEFAULT_MODEL_NAME)


def client_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_key": config.get("api_key") or os.getenv("OPENAI_API_KEY"),
        "base_url": normalize_openai_api_base(config.get("api_base") or get_openai_api_base(),
                                               "https://api.openai.com/v1"),
        "timeout": int(config.get("timeout", os.getenv("OPENAI_API_TIMEOUT", "300"))),
    }


def request_settings(config: dict[str, Any], *, temperature: float | None = None) -> dict[str, Any]:
    max_tokens = config.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = config.get("max_tokens")
    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_TOKEN_SIZE", "8192"))
    settings = {
        "temperature": config.get("temperature", 0.2) if temperature is None else temperature,
        "top_p": config.get("top_p", 1.0),
        "max_tokens": max_tokens,
    }
    # A framework RNG seed does not imply that the backend supports an API seed.
    if config.get("model_seed") is not None:
        settings["seed"] = config["model_seed"]
    return settings


def bench_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Retain the benchmark/format and all backend limits in child executors."""
    return {
        "config": dict(config),
        "model": model_name(config),
        "manager_tools": config.get("manager_tools"),
        "search_tools": config.get("search_tools"),
        "memory": config.get("memory"),
        "max_steps": config.get("max_steps", 15),
        "search_max_steps": config.get("search_max_steps", 10),
    }


def usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        incoming = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        outgoing = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    else:
        incoming = getattr(usage, "prompt_tokens", 0) or 0
        outgoing = getattr(usage, "completion_tokens", 0) or 0
    return {"input_tokens": incoming, "output_tokens": outgoing, "total_tokens": incoming + outgoing}
