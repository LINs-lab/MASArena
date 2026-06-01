"""
本地工具系统 - 为代理提供各种功能工具

这个模块定义了工具的基类和一些基础工具实现。
"""

import subprocess
import sys
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from io import StringIO
import contextlib
import logging

logger = logging.getLogger(__name__)


class Tool(ABC):
    """工具基类"""
    
    name: str = ""
    description: str = ""
    
    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__.lower().replace("tool", "")
        if not self.description:
            self.description = f"A tool for {self.name}"
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> str:
        """执行工具功能"""
        pass
    
    def __call__(self, *args, **kwargs) -> str:
        """使工具可调用"""
        return self.forward(*args, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为OpenAI函数调用格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema()
            }
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """获取参数模式（子类可重写）"""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class FinalAnswerTool(Tool):
    """最终答案工具"""
    
    name = "final_answer"
    description = "Use this tool to provide the final answer to the user's question"
    
    def forward(self, answer: str) -> str:
        """提供最终答案"""
        return f"Final answer: {answer}"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to provide to the user"
                }
            },
            "required": ["answer"]
        }


class PythonInterpreterTool(Tool):
    """Python代码解释器工具"""
    
    name = "python_interpreter"
    description = "Execute Python code and return the output"
    
    def __init__(self):
        super().__init__()
        self.globals_dict = {"__builtins__": __builtins__}
        self.locals_dict = {}
    
    def forward(self, code: str) -> str:
        """执行Python代码"""
        try:
            # 创建输出捕获
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout = StringIO()
            stderr = StringIO()
            
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                
                # 执行代码
                exec(code, self.globals_dict, self.locals_dict)
                
                # 获取输出
                stdout_value = stdout.getvalue()
                stderr_value = stderr.getvalue()
                
                # 组合输出
                output = ""
                if stdout_value:
                    output += stdout_value
                if stderr_value:
                    if output:
                        output += "\n"
                    output += f"Error: {stderr_value}"
                
                return output if output else "Code executed successfully (no output)"
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }


class WikipediaSearchTool(Tool):
    """维基百科搜索工具"""
    
    name = "wikipedia_search"
    description = "Search Wikipedia for information on a given topic"
    
    def __init__(self):
        super().__init__()
        self._wikipedia_available = False
        self._wikipedia = None
        
        # 尝试导入wikipedia包
        try:
            import wikipedia
            self._wikipedia = wikipedia
            self._wikipedia_available = True
            # 设置Wikipedia语言
            wikipedia.set_lang("en")
        except ImportError:
            logger.warning("wikipedia package not installed. WikipediaSearchTool will return error messages.")
    
    def forward(self, query: str, max_sentences: int = 3) -> str:
        """搜索维基百科"""
        if not self._wikipedia_available:
            return "Wikipedia search unavailable: 'wikipedia' package not installed. Please install it with: pip install wikipedia"
        
        try:
            # 搜索页面
            search_results = self._wikipedia.search(query, results=5)
            
            if not search_results:
                return f"No Wikipedia articles found for '{query}'"
            
            # 尝试获取第一个结果的摘要
            for title in search_results:
                try:
                    summary = self._wikipedia.summary(title, sentences=max_sentences)
                    return f"Wikipedia article: {title}\n\n{summary}"
                except self._wikipedia.exceptions.DisambiguationError as e:
                    # 如果有歧义，尝试第一个选项
                    try:
                        summary = self._wikipedia.summary(e.options[0], sentences=max_sentences)
                        return f"Wikipedia article: {e.options[0]}\n\n{summary}"
                    except:
                        continue
                except self._wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    logger.warning(f"Error getting summary for {title}: {e}")
                    continue
            
            return f"Found Wikipedia articles for '{query}' but couldn't retrieve content: {', '.join(search_results)}"
            
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for Wikipedia"
                },
                "max_sentences": {
                    "type": "integer",
                    "description": "Maximum number of sentences to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }


class WebSearchTool(Tool):
    """网络搜索工具（简化版）"""
    
    name = "web_search"
    description = "Search the web for information"
    
    def forward(self, query: str, num_results: int = 5) -> str:
        """执行网络搜索"""
        # 这里是一个简化的实现，实际项目中可以集成真正的搜索API
        return f"Web search results for '{query}':\n\nNote: This is a placeholder implementation. Please integrate with a real search API like Google Search API, Bing API, or DuckDuckGo API for actual web search functionality."
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }


class CalculatorTool(Tool):
    """计算器工具"""
    
    name = "calculator"
    description = "Perform mathematical calculations"
    
    def forward(self, expression: str) -> str:
        """执行数学计算"""
        try:
            # 简单的安全检查
            if any(dangerous in expression for dangerous in ['import', 'exec', 'eval', '__']):
                return "Error: Potentially dangerous expression detected"
            
            # 只允许数学运算符和函数
            allowed_names = {
                "abs", "round", "min", "max", "sum", "pow",
                "sin", "cos", "tan", "asin", "acos", "atan",
                "sinh", "cosh", "tanh", "log", "log10", "exp",
                "sqrt", "pi", "e"
            }
            
            # 创建安全的命名空间
            import math
            safe_dict = {name: getattr(math, name) for name in allowed_names if hasattr(math, name)}
            safe_dict.update({"__builtins__": {}})
            
            result = eval(expression, safe_dict)
            return str(result)
            
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }


class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """获取所有工具"""
        return list(self.tools.values())
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取所有工具的OpenAI函数调用格式"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    def execute_tool(self, name: str, *args, **kwargs) -> str:
        """执行工具"""
        tool = self.get_tool(name)
        if not tool:
            return f"Tool '{name}' not found"
        
        try:
            return tool.forward(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {str(e)}"
