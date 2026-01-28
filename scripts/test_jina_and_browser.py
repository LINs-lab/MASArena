#!/usr/bin/env python3
"""
测试 Jina 搜索 与 Browser 是否可用。

- Jina: JinaSearcher（tools_old/external_tools/web/search_tool.py），
  调用 s.jina.ai 做网页搜索，需环境变量 JINA_API_KEY。
- Browser: BrowserTool（mas_arena/tools/browser_tool.py），
  Playwright navigate -> get_content -> close。

运行（在项目根目录）:
  uv run python scripts/test_jina_and_browser.py
"""
import os
import sys
import asyncio

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 从项目根目录加载 .env，否则 os.getenv("JINA_API_KEY") 等读不到
from dotenv import load_dotenv
load_dotenv(os.path.join(REPO_ROOT, ".env"))


def test_jina():
    """测试 JinaSearcher（s.jina.ai 搜索）。"""
    print("\n========== 1. Jina (JinaSearcher, s.jina.ai) ==========")
    key = os.getenv("JINA_API_KEY")
    if not key:
        print("  JINA_API_KEY 未设置，无法测试 Jina 搜索。")
        return False
    print("  JINA_API_KEY 已设置，请求 s.jina.ai 搜索。")

    try:
        from mas_arena.tools_old.external_tools.web.search_tool import JinaSearcher

        searcher = JinaSearcher(api_key=key, max_results=3)
        out = searcher.search("Python programming")
        if not out or not str(out).strip():
            print("  结果: 失败（无返回内容）")
            return False
        print("  结果: 成功，返回长度:", len(str(out)))
        return True
    except Exception as e:
        print("  结果: 失败")
        print("  异常:", e)
        return False


async def _test_browser_async():
    """异步执行：navigate -> get_content -> close。"""
    from mas_arena.tools.browser_tool import BrowserTool

    tool = BrowserTool()
    if not getattr(tool, "available", True) or not getattr(tool, "browser_instance", None):
        return False, "BrowserTool 初始化失败或不可用"
    try:
        out_nav = await tool.forward(action="navigate", url="https://example.com")
        if "Error" in str(out_nav):
            return False, f"navigate 报错: {out_nav}"
        out_content = await tool.forward(action="get_content")
        if "Error" in str(out_content):
            return False, f"get_content 报错: {out_content}"
        await tool.forward(action="close")
        if "example" in (out_content or "").lower() or len(out_content or "") > 50:
            return True, "navigate + get_content + close 正常"
        return True, "执行完成，内容长度: " + str(len(out_content or ""))
    except Exception as e:
        try:
            await tool.forward(action="close")
        except Exception:
            pass
        return False, str(e)


def test_browser():
    """测试 BrowserTool（mas_arena/tools/browser_tool.py，Playwright）。"""
    print("\n========== 2. Browser (BrowserTool, Playwright) ==========")
    try:
        ok, msg = asyncio.run(_test_browser_async())
        if ok:
            print("  结果: 成功")
            print("  说明:", msg)
        else:
            print("  结果: 失败")
            print("  说明:", msg)
        return ok
    except Exception as e:
        print("  结果: 失败")
        print("  异常:", e)
        return False


def main():
    print("Testing JinaSearcher + BrowserTool (run from repo root: uv run python scripts/test_jina_and_browser.py)")
    jina_ok = test_jina()
    browser_ok = test_browser()
    print("\n========== 汇总 ==========")
    print("  Jina (JinaSearcher):   ", "OK" if jina_ok else "FAIL")
    print("  Browser (BrowserTool): ", "OK" if browser_ok else "FAIL")
    return 0 if (jina_ok and browser_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
