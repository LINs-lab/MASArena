import requests


def _quota_response(status_code: int = 429) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = b'{"error":"quota exceeded"}'
    response.url = "https://s.jina.ai/?q=test"
    return response


def test_jina_searcher_uses_next_key_when_quota_exhausted(monkeypatch):
    from mas_arena.tools.search_tool import JinaSearcher

    monkeypatch.setenv("JINA_API_KEY", "")
    monkeypatch.setenv("INA_API_KEY", "")
    monkeypatch.setenv("JINA_API_KEY_2", "second-key")

    used_authorizations = []

    def fake_get(url, headers, timeout):
        used_authorizations.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer first-key":
            return _quota_response()

        response = requests.Response()
        response.status_code = 200
        response._content = b"ok from backup key"
        response.url = url
        return response

    monkeypatch.setattr("mas_arena.tools.search_tool.requests.get", fake_get)

    searcher = JinaSearcher(api_key="first-key")

    assert searcher.search("test") == "ok from backup key"
    assert used_authorizations == ["Bearer first-key", "Bearer second-key"]


def test_jina_keys_accept_legacy_ina_api_key_name(monkeypatch):
    from mas_arena.tools.jina_keys import JINA_API_KEY_ENV_NAMES, get_jina_api_keys

    for name in JINA_API_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INA_API_KEY", "legacy-primary")
    monkeypatch.setenv("JINA_API_KEY_2", "second-key")

    assert get_jina_api_keys() == ["legacy-primary", "second-key"]


def test_jina_keys_include_extended_env_names(monkeypatch):
    from mas_arena.tools.jina_keys import JINA_API_KEY_ENV_NAMES, get_jina_api_keys

    assert "JINA_API_KEY_12" in JINA_API_KEY_ENV_NAMES
    assert "JINA_API_KEY_13" not in JINA_API_KEY_ENV_NAMES

    for name in JINA_API_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JINA_API_KEY_10", "tenth-key")

    assert get_jina_api_keys() == ["tenth-key"]
