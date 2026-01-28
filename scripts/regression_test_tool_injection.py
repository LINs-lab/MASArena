"""
回归测试：确保 CodeAgent 的 PythonInterpreterTool 环境里可以直接调用其他工具函数。

运行方式（推荐 uv）：
  uv run python scripts/regression_test_tool_injection.py
"""

import os
import sys

# 确保从仓库根目录导入（当脚本从 scripts/ 运行时，sys.path[0] 不是 repo root）
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mas_arena.agents.agent_core.agents import CodeAgent
from mas_arena.tools.python_interpreter import PythonInterpreterTool
from mas_arena.tools.visual_inspector import VisualInspectorTool


class FakeModel:
    """最小模型桩：第一轮返回代码块，第二轮返回最终答案。"""

    def __init__(self):
        self.calls = 0
        self.monitor = None

    def __call__(self, messages):
        self.calls += 1
        if self.calls == 1:
            return """```python
print('inspect_file_as_image' in globals())
print(callable(inspect_file_as_image))
```
"""
        return "<answer>ok</answer>"


def main() -> int:
    agent = CodeAgent(
        tools=[PythonInterpreterTool(), VisualInspectorTool()],
        model=FakeModel(),
        max_steps=3,
        verbosity_level=0,
    )
    out = agent.run("test")
    print("FINAL:", out)
    return 0 if out == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())


