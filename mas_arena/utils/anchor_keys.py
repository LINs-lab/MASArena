import os
from typing import Mapping, Optional


ANCHOR_API_KEY_ENV_NAMES = (
    "ANCHOR_API_KEY",
    "ANCHOR_API_KEY_2",
    "ANCHOR_API_KEY_3",
    "ANCHOR_API_KEY_4",
    "ANCHOR_API_KEY_5",
)
ANCHOR_QUOTA_STATUS_CODES = {402, 429}
ANCHOR_QUOTA_ERROR_TOKENS = (
    "quota",
    "credit",
    "credits",
    "insufficient",
    "payment required",
    "rate limit",
    "rate_limit",
    "too many requests",
    "额度",
    "余额",
)


def _normalize_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = value.strip().strip("\"'")
    return normalized or None


def get_anchor_api_keys(
    primary_key: Optional[str] = None,
    env: Mapping[str, str] = os.environ,
) -> list[str]:
    """Return configured Anchor keys in failover order."""
    keys: list[str] = []

    for key in (primary_key, *(env.get(name) for name in ANCHOR_API_KEY_ENV_NAMES)):
        normalized = _normalize_api_key(key)
        if normalized and normalized not in keys:
            keys.append(normalized)

    return keys


def is_anchor_quota_error(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in ANCHOR_QUOTA_STATUS_CODES:
        return True

    text = str(error).lower()
    try:
        response_text = (getattr(response, "text", "") or "").lower()
    except Exception:
        response_text = ""

    return any(token in text or token in response_text for token in ANCHOR_QUOTA_ERROR_TOKENS)
