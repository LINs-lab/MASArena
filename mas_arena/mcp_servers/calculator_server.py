import sys
import json
from asteval import Interpreter

def safe_eval(expr):
    """
    安全地计算数学表达式。
    使用 asteval 库避免 eval() 的安全隐患。
    """
    # 创建一个安全的表达式解析器
    aeval = Interpreter()
    try:
        # 尝试解析并计算表达式
        result = aeval(expr)
        if aeval.error:  # 检查是否有错误
            return json.dumps({"error": str(aeval.error[0].exc_value)})
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})

def validate_expression(expr):
    """
    验证表达式是否为合法的数学表达式。
    可以根据需求扩展验证规则。
    """
    if not isinstance(expr, str):
        return False
    # 简单的正则验证（可根据需要调整）
    import re
    pattern = r'^[\d\s\+\-\*/\(\)\.\^e\%\s]+$'  # 允许数字、运算符和括号
    return bool(re.match(pattern, expr))

if __name__ == "__main__":
    try:
        # 从标准输入读取 JSON 数据
        raw_input = sys.stdin.read()
        input_data = json.loads(raw_input)

        # 提取 tool_name 和 tool_input
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})

        # 处理 calculator 工具
        if tool_name == "calculator":
            expr = tool_input.get("expression")
            if not expr:
                print(json.dumps({"error": "Missing 'expression' in tool_input"}))
            elif not validate_expression(expr):
                print(json.dumps({"error": "Invalid expression format"}))
            else:
                print(safe_eval(expr))
        else:
            print(json.dumps({"error": f"Unknown tool_name: {tool_name}"}))
    except json.JSONDecodeError:
        print(json.dumps({"error": "Invalid JSON input"}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))