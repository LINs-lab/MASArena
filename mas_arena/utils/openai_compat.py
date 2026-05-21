from urllib.parse import urlparse, urlunparse


def normalize_openai_api_base(api_base: str | None, default: str) -> str:
    """Normalize OpenAI-compatible base URLs."""
    normalized = (api_base or default or "").strip()
    if not normalized:
        return ""

    parsed = urlparse(normalized)
    path = (parsed.path or "").rstrip("/")
    if path in {"", "/"}:
        path = "/v1"

    return urlunparse(parsed._replace(path=path))
