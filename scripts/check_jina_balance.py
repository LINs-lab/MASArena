#!/usr/bin/env python3
"""
一键检查 .env 中所有 Jina API Key 的剩余 token 余额。

用法（项目根目录）:
  uv run python scripts/check_jina_balance.py
  bash scripts/check_jina_balance.sh

原理：对 https://r.jina.ai 发带 Authorization 的 GET（不跟具体 URL），
响应正文含 [Balance left] <tokens>，见 https://github.com/jina-ai/reader/issues/64
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mas_arena.tools.jina_keys import (  # noqa: E402
    JINA_API_KEY_ENV_NAMES,
    get_jina_api_keys,
)

BALANCE_URL = "https://r.jina.ai"
BALANCE_RE = re.compile(r"\[Balance left\]\s*(\d+)")
AUTH_RE = re.compile(r"\[Authenticated as\]\s*(.+)", re.MULTILINE)


@dataclass
class KeyBalance:
    env_name: str
    key_prefix: str
    status: str
    balance: Optional[int] = None
    account: Optional[str] = None
    error: Optional[str] = None


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return key[:4] + "..."
    return key[:12] + "..."


def _key_env_map(env: dict[str, str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in JINA_API_KEY_ENV_NAMES:
        raw = env.get(name)
        if not raw:
            continue
        normalized = raw.strip().strip("\"'")
        if normalized and normalized not in mapping:
            mapping[normalized] = name
    return mapping


def _parse_error_message(text: str) -> str:
    text = text.strip()
    if not text.startswith("{"):
        return text[:200]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:200]
    return (
        payload.get("readableMessage")
        or payload.get("message")
        or payload.get("name")
        or text[:200]
    )


def fetch_balance(api_key: str, timeout: float) -> KeyBalance:
    prefix = _mask_key(api_key)
    try:
        resp = requests.get(
            BALANCE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return KeyBalance(
            env_name="",
            key_prefix=prefix,
            status="error",
            error=str(exc),
        )

    text = resp.text or ""
    auth_match = AUTH_RE.search(text)
    account = auth_match.group(1).strip() if auth_match else None
    balance_match = BALANCE_RE.search(text)

    if balance_match is not None:
        balance = int(balance_match.group(1))
        if balance <= 0:
            status = "empty"
        else:
            status = "ok"
        return KeyBalance(
            env_name="",
            key_prefix=prefix,
            status=status,
            balance=balance,
            account=account,
        )

    if resp.status_code == 401:
        return KeyBalance(
            env_name="",
            key_prefix=prefix,
            status="invalid",
            error=_parse_error_message(text),
        )
    if resp.status_code == 402:
        return KeyBalance(
            env_name="",
            key_prefix=prefix,
            status="empty",
            balance=0,
            error=_parse_error_message(text),
        )

    return KeyBalance(
        env_name="",
        key_prefix=prefix,
        status="error",
        error=f"HTTP {resp.status_code}: {_parse_error_message(text)}",
    )


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:,} ({n / 1_000_000:.2f}M)"
    if n >= 1_000:
        return f"{n:,} ({n / 1_000:.1f}K)"
    return f"{n:,}"


def main() -> int:
    parser = argparse.ArgumentParser(description="检查所有 Jina API Key 剩余 token")
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="单次请求超时（秒），默认 20",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果",
    )
    args = parser.parse_args()

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    env = dict(os.environ)
    env_map = _key_env_map(env)
    keys = get_jina_api_keys()

    if not keys:
        print("未找到 Jina API Key。请在 .env 中设置 JINA_API_KEY 或 JINA_API_KEY_2 等。")
        return 1

    rows: list[KeyBalance] = []
    for key in keys:
        row = fetch_balance(key, timeout=args.timeout)
        row.env_name = env_map.get(key, "?")
        rows.append(row)

    if args.json:
        import json as json_mod

        print(
            json_mod.dumps(
                [
                    {
                        "env": r.env_name,
                        "key_prefix": r.key_prefix,
                        "status": r.status,
                        "balance": r.balance,
                        "account": r.account,
                        "error": r.error,
                    }
                    for r in rows
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("Jina API Key 余额（GET r.jina.ai，不消耗余额）\n")
        print(f"{'环境变量':<18} {'Key':<16} {'状态':<8} {'剩余 tokens':<22} 备注")
        print("-" * 90)
        for r in rows:
            if r.balance is not None:
                bal_str = _format_tokens(r.balance)
            else:
                bal_str = "-"
            note = (r.error or r.account or "")[:40]
            print(
                f"{r.env_name:<18} {r.key_prefix:<16} {r.status:<8} {bal_str:<22} {note}"
            )

        ok = [r for r in rows if r.status == "ok"]
        empty = [r for r in rows if r.status == "empty"]
        invalid = [r for r in rows if r.status == "invalid"]
        errors = [r for r in rows if r.status == "error"]
        total_balance = sum(r.balance or 0 for r in ok)

        print()
        print(
            f"汇总: 共 {len(rows)} 个 key | "
            f"可用 {len(ok)} | 余额为 0 {len(empty)} | 无效 {len(invalid)} | 请求失败 {len(errors)}"
        )
        if ok:
            print(f"可用 key 合计余额: {_format_tokens(total_balance)} tokens")
        print("Dashboard: https://jina.ai/api-dashboard/key-manager")

    has_problem = any(r.status in ("empty", "invalid", "error") for r in rows)
    return 1 if has_problem else 0


if __name__ == "__main__":
    sys.exit(main())
