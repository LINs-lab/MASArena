from typing import Dict, Any, List, Optional
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

    def __init__(self, authorized_imports: Optional[List[str]] = None, additional_globals: Optional[Dict[str, Any]] = None):
        """
        Initialize the Python interpreter tool.
        
        Args:
            authorized_imports: List of module names that are allowed to be imported.
            additional_globals: Dictionary of additional global variables/functions to inject.
        """
        super().__init__()
        self.authorized_imports = authorized_imports or []
        self.additional_globals = additional_globals or {}
        # Don't initialize globals_dict here with __builtins__ as it might trigger validation issues
        # We'll initialize execution state in forward

    def forward(self, code: str) -> str:
        """Execute Python code and return its output."""
        # Import necessary modules inside the method to avoid static analysis issues
        import sys
        from io import StringIO
        import builtins
        
        try:
            # Capture stdout and stderr
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout, stderr = StringIO(), StringIO()
            
            # Prepare execution environment
            # We construct a fresh dict for each execution to avoid state pollution if desired,
            # but here we keep it simple.
            
            # Safe builtins
            safe_builtins = {
                name: getattr(builtins, name)
                for name in dir(builtins)
                if name not in ["open", "exit", "quit"]
            }
            
            # Add authorized imports to globals
            globals_dict = {"__builtins__": safe_builtins}
            
            # Add additional globals
            if self.additional_globals:
                globals_dict.update(self.additional_globals)
            
            # Pre-import authorized modules
            for module_name in self.authorized_imports:
                try:
                    module = __import__(module_name)
                    globals_dict[module_name] = module
                except ImportError:
                    pass
            
            # Also allow direct import of basic IO/Sys for output capture if needed by the code
            # (though we capture it externally, some code explicitly imports sys)
            if "sys" not in globals_dict:
                globals_dict["sys"] = sys
            if "io" not in globals_dict:
                import io
                globals_dict["io"] = io

            locals_dict = {}

            try:
                sys.stdout = stdout
                sys.stderr = stderr

                exec(code, globals_dict, locals_dict)

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


def test_state_persistence_across_calls():
    """测试状态持久化（这里实际上每次调用都是隔离的，如果需要持久化需要修改类）"""
    # 注意：当前的实现每次 forward 都是新的 globals/locals，所以不持久化。
    # 这与 smolagents 的默认行为一致（无状态）。
    tool = PythonInterpreterTool()
    tool.forward("x = 42")
    result = tool.forward("print(x if 'x' in locals() else 'x not found')")
    # 因为我们每次都重置 locals，所以 x 应该找不到，或者是 'x not found'
    # 如果要支持持久化，需要把 locals_dict 提升为实例属性
    assert "Error" in result or "x not found" in result or "name 'x' is not defined" in result
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
for i in range(3):
    print(f"Loop {i}")
"""
    tool = PythonInterpreterTool()
    result = tool.forward(code)
    assert "Loop 0" in result
    assert "Loop 1" in result
    assert "Loop 2" in result
    print("✅ test_special_characters_and_multiline passed")


if __name__ == "__main__":
    test_basic_print()
    test_no_output_code()
    test_stdout_and_stderr()
    test_runtime_error()
    test_state_persistence_across_calls()
    test_tool_metadata_compliance()
    test_special_characters_and_multiline()
