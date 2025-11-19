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