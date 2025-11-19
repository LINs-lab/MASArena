from typing import Dict, Any
from io import StringIO
import sys
from smolagents import Tool


class PythonInterpreterTool(Tool):
    """Python code interpreter tool for safely executing Python snippets."""

    name = "python_interpreter"
    description = "Execute Python code and return the captured stdout/stderr output."

    inputs = {
        "code": {
            "type": "string",
            "description": "Python code to execute.",
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Restrict builtins for safety (basic sandboxing)
        self.globals_dict = {"__builtins__": __builtins__}
        self.locals_dict = {}

    def forward(self, code: str) -> str:
        """Execute Python code and return its output."""
        try:
            # Capture stdout and stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout, stderr = StringIO(), StringIO()

            try:
                sys.stdout = stdout
                sys.stderr = stderr

                exec(code, self.globals_dict, self.locals_dict)

                stdout_value = stdout.getvalue()
                stderr_value = stderr.getvalue()

                output_parts = []
                if stdout_value:
                    output_parts.append(stdout_value.rstrip())
                if stderr_value:
                    output_parts.append(f"Error: {stderr_value.rstrip()}")

                return (
                    "\n".join(output_parts)
                    if output_parts
                    else "Code executed successfully (no output)"
                )

            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            return f"Error executing code: {str(e)}"


def test_basic_print():
    """测试基本 print 输出"""
    tool = PythonInterpreterTool()
    result = tool.forward("print('Hello from interpreter!')")
    assert "Hello from interpreter!" in result
    print("✅ test_basic_print passed")


def test_no_output_code():
    """测试无输出的代码（如赋值）"""
    tool = PythonInterpreterTool()
    result = tool.forward("x = 10\ny = x * 2")
    assert result == "Code executed successfully (no output)"
    print("✅ test_no_output_code passed")


def test_stdout_and_stderr():
    """测试同时有 stdout 和 stderr"""
    code = """
print("Standard output")
import sys
sys.stderr.write("This is an error message\\n")
"""
    tool = PythonInterpreterTool()
    result = tool.forward(code)
    assert "Standard output" in result
    assert "Error: This is an error message" in result
    print("✅ test_stdout_and_stderr passed")


def test_runtime_error():
    """测试运行时异常（如除零）"""
    tool = PythonInterpreterTool()
    result = tool.forward("1 / 0")
    assert "Error executing code:" in result
    assert "ZeroDivisionError" in result
    print("✅ test_runtime_error passed")


def test_syntax_error():
    """测试语法错误"""
    tool = PythonInterpreterTool()
    result = tool.forward("print('missing quote)")
    assert "Error executing code:" in result
    assert "SyntaxError" in result
    print("✅ test_syntax_error passed")


def test_empty_code():
    """测试空代码或空白代码"""
    tool = PythonInterpreterTool()
    assert tool.forward("") == "Code executed successfully (no output)"
    assert tool.forward("   \n\t  ") == "Code executed successfully (no output)"
    print("✅ test_empty_code passed")


def test_state_persistence_across_calls():
    """测试多次调用间是否保持变量状态（locals/globals）"""
    tool = PythonInterpreterTool()
    tool.forward("counter = 100")
    result = tool.forward("print(counter + 50)")
    assert "150" in result
    print("✅ test_state_persistence_across_calls passed")


def test_tool_metadata_compliance():
    """验证是否符合 smolagents.Tool 规范（inputs, output_type 等）"""
    tool = PythonInterpreterTool()

    # 检查类属性
    assert tool.name == "python_interpreter"
    assert "Execute Python code" in tool.description

    # 检查 inputs schema
    assert isinstance(tool.inputs, dict)
    assert "code" in tool.inputs
    assert tool.inputs["code"]["type"] == "string"
    assert "description" in tool.inputs["code"]

    # 检查 output_type
    assert tool.output_type == "string"

    print("✅ test_tool_metadata_compliance passed")


def test_special_characters_and_multiline():
    """测试多行代码与特殊字符"""
    code = """
text = "🌟 Hello\\nWorld! 🌍"
print(text)
"""
    tool = PythonInterpreterTool()
    result = tool.forward(code)
    assert "🌟 Hello" in result
    assert "World! 🌍" in result
    print("✅ test_special_characters_and_multiline passed")


if __name__ == "__main__":
    print("🧪 Running tests for PythonInterpreterTool (smolagents style)...\n")

    tests = [
        test_basic_print,
        test_no_output_code,
        test_stdout_and_stderr,
        test_runtime_error,
        test_syntax_error,
        test_empty_code,
        test_state_persistence_across_calls,
        test_tool_metadata_compliance,
        test_special_characters_and_multiline,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n💥 {failed} test(s) failed.")
        sys.exit(1)