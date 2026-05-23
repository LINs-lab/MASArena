import os
from typing import Mapping, Optional

import requests


JINA_API_KEY_ENV_NAMES = (
    "JINA_API_KEY",
    "INA_API_KEY",
    *(f"JINA_API_KEY_{i}" for i in range(2, 13)),
)
JINA_QUOTA_STATUS_CODES = {402, 429}
JINA_QUOTA_ERROR_TOKENS = (
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


def get_jina_api_keys(
    primary_key: Optional[str] = None,
    env: Mapping[str, str] = os.environ,
) -> list[str]:
    """Return configured Jina keys in failover order."""
    keys: list[str] = []

    for key in (primary_key, *(env.get(name) for name in JINA_API_KEY_ENV_NAMES)):
        normalized = _normalize_api_key(key)
        if normalized and normalized not in keys:
            keys.append(normalized)

    return keys


def is_jina_quota_error(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in JINA_QUOTA_STATUS_CODES:
        return True

    text = str(error).lower()
    try:
        response_text = (getattr(response, "text", "") or "").lower()
    except Exception:
        response_text = ""

    return any(token in text or token in response_text for token in JINA_QUOTA_ERROR_TOKENS)


def raise_for_jina_status(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response is None:
            exc.response = resp
        raise
