import importlib
import sys
import types

import requests


def _quota_response(status_code: int = 429) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = b'{"error":"insufficient credits"}'
    response.url = "https://api.anchorbrowser.io/v1/sessions"
    return response


def test_anchor_keys_include_secondary_env_key(monkeypatch):
    from mas_arena.utils.anchor_keys import get_anchor_api_keys

    monkeypatch.setenv("ANCHOR_API_KEY", "primary-key")
    monkeypatch.setenv("ANCHOR_API_KEY_2", "secondary-key")

    assert get_anchor_api_keys() == ["primary-key", "secondary-key"]


def test_browser_session_uses_next_anchor_key_when_quota_exhausted(monkeypatch):
    fake_browser_use = types.SimpleNamespace(
        Agent=object,
        Browser=object,
        ChatOpenAI=object,
        Controller=object,
        Tools=object,
    )
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    browser_module = importlib.import_module("mas_arena.tools_old.external_tools.web.browser")
    browser_module = importlib.reload(browser_module)

    monkeypatch.setenv("ANCHOR_API_KEY", "primary-key")
    monkeypatch.setenv("ANCHOR_API_KEY_2", "secondary-key")

    used_keys = []

    def fake_post(url, headers):
        used_keys.append(headers["anchor-api-key"])
        if headers["anchor-api-key"] == "primary-key":
            return _quota_response()

        response = requests.Response()
        response.status_code = 200
        response._content = b'{"data":{"cdp_url":"wss://backup-session"}}'
        response.url = url
        return response

    monkeypatch.setattr(browser_module.requests, "post", fake_post)

    tool = object.__new__(browser_module.BrowserTool)

    assert tool._create_remote_browser_session() == "wss://backup-session"
    assert used_keys == ["primary-key", "secondary-key"]
