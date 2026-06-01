from mas_arena.tools.base import ToolFactory
from langchain_core.tools import tool as langchain_tool
import subprocess
import sys
from typing import List

def run_python_code(code: str) -> str:
    """
    Executes a string of Python code in a separate process and captures its output.
    """
    try:
        # Use the same Python interpreter that's running this script
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30  # Add a timeout to prevent hanging
        )
        if process.returncode == 0:
            return process.stdout or "Code executed successfully with no output."
        else:
            return f"Error during execution:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30 seconds."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@langchain_tool
def run_in_python_repl(code: str) -> str:
    """
    Runs Python code in a REPL-like environment and returns the output.
    This is not a real REPL, but it executes the code and captures stdout/stderr.
    The code should be a complete script, not a single expression to be evaluated.
    Example:
    ```python
    print("Hello, World!")
    ```
    """
    return run_python_code(code)

@ToolFactory.register(name="python_repl", desc="A Python REPL that can execute python code.", category="Code")
class PythonReplTool:
    def get_tools(self) -> List:
        return [run_in_python_repl] 