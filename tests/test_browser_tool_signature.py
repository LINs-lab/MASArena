import sys
import os

# Ensure project root is importable when running pytest from repo root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_browser_tool_can_be_instantiated_and_called():
    """
    Regression test for smolagents.Tool signature validation:
    BrowserTool.forward() parameters must exactly match BrowserTool.inputs keys.
    """
    from mas_arena.tools.browser_tool import BrowserTool

    tool = BrowserTool()
    assert tool.name == "browser"
    # Should not raise even if the underlying browser implementation isn't available.
    out = tool.forward(action="get_url", url=None)
    assert isinstance(out, str)


