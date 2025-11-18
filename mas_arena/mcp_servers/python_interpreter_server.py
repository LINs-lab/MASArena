import sys
import json
from io import StringIO
import contextlib
import ast
import multiprocessing

# 安全内置函数白名单
safe_builtins = {
    'print': print,
    'range': range,
    'len': len,
    'str': str,
    'int': int,
    'float': float,
    'list': list,
    'dict': dict,
    'sum': sum,
    'abs': abs,
    'round': round,
    'all': all,
    'any': any,
}

@contextlib.contextmanager
def stdout_io():
    old_stdout = sys.stdout
    new_stdout = StringIO()
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout

def execute_python_code(code: str):
    with stdout_io() as s:
        try:
            exec(code, {"__builtins__": safe_builtins})
            return {"result": s.getvalue(), "stdout": s.getvalue(), "stderr": ""}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)}", "stdout": "", "stderr": str(e)}

def run_in_subprocess(code_queue, code):
    try:
        result = execute_python_code(code)
        code_queue.put(result)
    except Exception as e:
        code_queue.put(json.dumps({"error": str(e)}))

def safe_execute_with_timeout(code, timeout=5):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_in_subprocess, args=(queue, code))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        return json.dumps({"error": "Execution timed out"})
    return queue.get()

def validate_code(code):
    # 简单正则校验
    import re
    if not re.match(r'^[a-zA-Z0-9\s\+\-\*/\=\(\)\{\}\[\]\:\.\,\'\"]+$', code):
        return False
    return True

if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        input_data = json.loads(raw_input)

        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})

        if tool_name == "python":
            code = tool_input.get("code")
            if not code:
                print(json.dumps({"error": "Missing 'code' in tool_input"}))
            elif not validate_code(code):
                print(json.dumps({"error": "Invalid code format"}))
            else:
                print(safe_execute_with_timeout(code))
        else:
            print(json.dumps({"error": f"Unknown tool_name: {tool_name}"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))