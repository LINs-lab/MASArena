#!/usr/bin/env python3
"""测试 MiniMax OpenAI 兼容接口是否连通。

在项目根目录运行:
  uv run python scripts/test_minimax_connectivity.py

也支持环境变量覆盖:
  MINIMAX_MODEL / MINIMAX_API_KEY / MINIMAX_BASE_URL
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO_ROOT, ".env"))

MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7").strip()
API_KEY = os.getenv("MINIMAX_API_KEY", "").strip()
BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1").strip().rstrip("/")
TIMEOUT_SECONDS = 30


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> int:
    print("Testing MiniMax connectivity")
    print(f"  MINIMAX_MODEL={MODEL}")
    print(f"  MINIMAX_BASE_URL={BASE_URL}")
    print(f"  MINIMAX_API_KEY={'set' if API_KEY else 'missing'}")

    if not API_KEY:
        return fail("MINIMAX_API_KEY 未设置")

    try:
        import httpx
        from openai import OpenAI
    except ImportError as e:
        return fail(f"缺少依赖: {e}")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=TIMEOUT_SECONDS,
        http_client=httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=True),
    )

    print("\n[1/2] GET /v1/models")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        ok(f"返回 {len(model_ids)} 个模型: {', '.join(model_ids)}")
    except Exception as e:
        return fail(f"/v1/models 请求失败: {e}")

    print(f"\n[2/2] POST /v1/chat/completions (model={MODEL})")
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=32,
        )
        reply = (completion.choices[0].message.content or "").strip()
        ok(f"响应: {reply!r}")
    except Exception as e:
        return fail(f"chat/completions 请求失败: {e}")

    print("\nRESULT: MiniMax 配置可连通")
    return 0


if __name__ == "__main__":
    sys.exit(main())
