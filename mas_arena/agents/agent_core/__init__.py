"""
Agent核心框架 - 本地Agent系统实现

这个模块提供了Agent系统的核心组件，包括步骤管理、代理实现和模型集成。
完全替代了对外部smolagents库的依赖。
"""

from .steps import ActionStep, PlanningStep, TaskStep
from .agents import MultiStepAgent, CodeAgent, ToolCallingAgent
from .models import OpenAIServerModel

# 从tools模块导入工具类
from mas_arena.tools.local_tools import (
    Tool, FinalAnswerTool, PythonInterpreterTool, WikipediaSearchTool, 
    CalculatorTool, WebSearchTool, ToolManager
)

__all__ = [
    # 步骤类
    "ActionStep",
    "PlanningStep", 
    "TaskStep",
    
    # 代理类
    "MultiStepAgent",
    "CodeAgent",
    "ToolCallingAgent",
    
    # 模型类
    "OpenAIServerModel",
    
    # 工具类
    "Tool",
    "FinalAnswerTool",
    "PythonInterpreterTool",
    "WikipediaSearchTool",
    "CalculatorTool",
    "WebSearchTool",
    "ToolManager",
]
