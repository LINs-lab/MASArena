import os
import warnings


DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


def get_model_name(default: str = DEFAULT_MODEL_NAME) -> str:
    return os.getenv("MODEL_NAME", default)


def get_openai_api_base(default: str | None = DEFAULT_OPENAI_API_BASE) -> str | None:
    api_base = os.getenv("OPENAI_API_BASE")
    legacy_api_base = os.getenv("OPENAI_BASE_URL")
    if api_base:
        return api_base
    if legacy_api_base:
        warnings.warn(
            "OPENAI_BASE_URL is deprecated; use OPENAI_API_BASE instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy_api_base
    return default


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")


def get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default
