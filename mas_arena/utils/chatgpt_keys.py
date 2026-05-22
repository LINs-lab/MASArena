import os
import threading
from typing import Mapping, Optional


CHATGPT_API_KEY_ENV_NAMES = (
    "CHATGPT_API_KEY",
    "CHATGPT_API_KEY_2",
    "CHATGPT_API_KEY_3",
    "CHATGPT_API_KEY_4",
    "CHATGPT_API_KEY_5",
)
CHATGPT_KEY_ROTATION_ENV_NAME = "CHATGPT_KEY_ROTATION_ENABLED"
CHATGPT_KEY_ROTATION_TRUE_VALUES = {"1", "true", "yes", "on"}

CHATGPT_RATE_LIMIT_STATUS_CODES = {429}
CHATGPT_RATE_LIMIT_TOKENS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
)

_next_chatgpt_key_index = 0
_chatgpt_key_lock = threading.Lock()


def _normalize_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().strip("\"'")
    return normalized or None


def is_chatgpt_key_rotation_enabled(env: Mapping[str, str] = os.environ) -> bool:
    value = _normalize_api_key(env.get(CHATGPT_KEY_ROTATION_ENV_NAME))
    return bool(value and value.lower() in CHATGPT_KEY_ROTATION_TRUE_VALUES)


def get_chatgpt_api_keys(
    primary_key: Optional[str] = None,
    env: Mapping[str, str] = os.environ,
) -> list[str]:
    """Return active ChatGPT keys in failover order."""
    keys: list[str] = []

    env_names = CHATGPT_API_KEY_ENV_NAMES if is_chatgpt_key_rotation_enabled(env) else (CHATGPT_API_KEY_ENV_NAMES[0],)
    for key in (primary_key, *(env.get(name) for name in env_names)):
        normalized = _normalize_api_key(key)
        if normalized and normalized not in keys:
            keys.append(normalized)

    return keys


def get_next_chatgpt_api_key(
    primary_key: Optional[str] = None,
    env: Mapping[str, str] = os.environ,
) -> Optional[str]:
    """Return the next configured key using process-local round-robin."""
    keys = get_chatgpt_api_keys(primary_key=primary_key, env=env)
    if not keys:
        return None

    global _next_chatgpt_key_index
    with _chatgpt_key_lock:
        key = keys[_next_chatgpt_key_index % len(keys)]
        _next_chatgpt_key_index += 1
        return key


def is_chatgpt_rate_limit_error(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    response_status_code = getattr(response, "status_code", None)

    if status_code in CHATGPT_RATE_LIMIT_STATUS_CODES:
        return True
    if response_status_code in CHATGPT_RATE_LIMIT_STATUS_CODES:
        return True

    text = str(error).lower()
    return any(token in text for token in CHATGPT_RATE_LIMIT_TOKENS)


def mask_api_key(api_key: Optional[str]) -> str:
    normalized = _normalize_api_key(api_key)
    if not normalized:
        return "<missing>"
    if len(normalized) <= 8:
        return "***"
    return f"{normalized[:4]}...{normalized[-4:]}"
