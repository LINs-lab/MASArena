from typing import Any, Dict, List, Optional
import logging
from smolagents import Tool

logger = logging.getLogger(__name__)


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
        schemas = []
        for tool in self.tools.values():
            raw_schema = tool.to_dict()
            
            # 兼容性处理：确保符合 OpenAI Tools API 格式
            if "type" in raw_schema and raw_schema["type"] == "function" and "function" in raw_schema:
                # 已经是标准格式
                schemas.append(raw_schema)
            else:
                # 假设是 Function 定义，进行包装
                # 注意：smolagents 的 to_dict 可能包含 type 字段但不是 'function' (例如可能是 output type)
                # 我们需要清理不应该出现在 function definition 中的字段
                
                function_def = raw_schema.copy()
                # 移除可能导致冲突的顶层字段
                function_def.pop("type", None) 
                
                schemas.append({
                    "type": "function",
                    "function": function_def
                })
        return schemas

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