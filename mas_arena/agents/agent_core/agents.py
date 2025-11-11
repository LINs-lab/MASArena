"""
代理实现 - 多步骤代理系统

这个模块实现了各种类型的代理，包括基础的多步骤代理、代码代理和工具调用代理。
"""

import json
import re
import ast
import sys
from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import logging

from .steps import ActionStep, PlanningStep, TaskStep, StepMemory
from .models import OpenAIServerModel
from mas_arena.tools.local_tools import Tool, ToolManager, FinalAnswerTool, PythonInterpreterTool

logger = logging.getLogger(__name__)


class MultiStepAgent(ABC):
    """多步骤代理基类"""
    
    def __init__(
        self,
        tools: List[Tool],
        model: OpenAIServerModel,
        max_steps: int = 10,
        verbosity_level: int = 1,
        planning_interval: Optional[int] = None,
        provide_run_summary: bool = False,
        step_callbacks: Optional[List[Callable]] = None,
    ):
        """
        初始化多步骤代理
        
        Args:
            tools: 可用工具列表
            model: 语言模型
            max_steps: 最大步数
            verbosity_level: 详细程度
            planning_interval: 计划间隔
            provide_run_summary: 是否提供运行摘要
            step_callbacks: 步骤回调函数列表
        """
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.verbosity_level = verbosity_level
        self.planning_interval = planning_interval
        self.provide_run_summary = provide_run_summary
        self.step_callbacks = step_callbacks or []
        
        # 初始化工具管理器
        self.tool_manager = ToolManager()
        for tool in tools:
            self.tool_manager.register_tool(tool)
        
        # 步骤内存
        self.memory = StepMemory()
        
        # 代理属性
        self.name = getattr(self, 'name', self.__class__.__name__)
        self.description = getattr(self, 'description', '')
        
        # 提示模板
        self.prompt_templates = {
            "system_prompt": self._get_default_system_prompt(),
            "managed_agent": {
                "task": ""
            }
        }
        
        # 监控器（兼容smolagents接口）
        self.monitor = model.monitor
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        return f"""You are {self.name}, a helpful AI assistant.
{self.description}

You have access to the following tools:
{self._format_tools_description()}

Always use tools when needed to complete tasks effectively."""
    
    def _format_tools_description(self) -> str:
        """格式化工具描述"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    @abstractmethod
    def run(self, task: str, additional_args: Optional[Dict[str, Any]] = None) -> Any:
        """运行代理"""
        pass
    
    def _call_step_callbacks(self, step: Any) -> None:
        """调用步骤回调"""
        for callback in self.step_callbacks:
            try:
                callback(step, self)
            except Exception as e:
                logger.warning(f"Step callback error: {e}")


class ToolCallingAgent(MultiStepAgent):
    """工具调用代理"""
    
    def run(self, task: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """运行工具调用代理"""
        self.memory.clear()
        additional_args = additional_args or {}
        
        # 构建系统消息
        system_prompt = self.prompt_templates["system_prompt"]
        if additional_args.get("additional_knowledge"):
            system_prompt += f"\n\nAdditional knowledge:\n{additional_args['additional_knowledge']}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        
        for step_num in range(self.max_steps):
            if self.verbosity_level > 0:
                print(f"Step {step_num + 1}/{self.max_steps}")
            
            try:
                # 调用模型生成工具调用
                tools_schema = self.tool_manager.get_tools_schema()
                response = self.model.generate_with_tools(messages, tools_schema)
                
                assistant_message = response.choices[0].message
                messages.append(assistant_message.dict())
                
                # 检查是否有工具调用
                if assistant_message.tool_calls:
                    tool_calls = []
                    observations = []
                    
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if self.verbosity_level > 1:
                            print(f"Calling tool: {function_name} with args: {function_args}")
                        
                        # 执行工具调用
                        result = self.tool_manager.execute_tool(function_name, **function_args)
                        
                        tool_calls.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                        observations.append(result)
                        
                        # 添加工具结果到消息历史
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                        
                        # 检查是否是最终答案
                        if function_name == "final_answer":
                            # 记录步骤
                            action_step = ActionStep(
                                tool_calls=tool_calls,
                                observations=observations,
                                model_output=result,
                                agent_name=self.name
                            )
                            self.memory.add_step(action_step)
                            self._call_step_callbacks(action_step)
                            return result
                    
                    # 记录动作步骤
                    action_step = ActionStep(
                        tool_calls=tool_calls,
                        observations=observations,
                        model_output=assistant_message.content,
                        agent_name=self.name
                    )
                    self.memory.add_step(action_step)
                    self._call_step_callbacks(action_step)
                    
                else:
                    # 没有工具调用，直接返回回应
                    content = assistant_message.content or ""
                    if self.verbosity_level > 1:
                        print(f"Agent response: {content}")
                    return content
                    
            except Exception as e:
                error_msg = f"Error in step {step_num + 1}: {str(e)}"
                logger.error(error_msg)
                
                # 记录错误步骤
                error_step = ActionStep(
                    error=error_msg,
                    agent_name=self.name
                )
                self.memory.add_step(error_step)
                self._call_step_callbacks(error_step)
                
                return f"Agent encountered an error: {error_msg}"
        
        return f"Agent reached maximum steps ({self.max_steps}) without completing the task"
    
    def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """处理工具调用（兼容smolagents接口）"""
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.get("function", "")
            arguments = tool_call.get("arguments", {})
            result = self.tool_manager.execute_tool(function_name, **arguments)
            results.append(result)
        return results


class CodeAgent(MultiStepAgent):
    """代码代理 - 可以编写和执行Python代码"""
    
    def __init__(
        self,
        tools: List[Tool],
        model: OpenAIServerModel,
        max_steps: int = 10,
        managed_agents: Optional[List[MultiStepAgent]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        instructions: Optional[str] = None,
        stream_outputs: bool = False,
        **kwargs
    ):
        """
        初始化代码代理
        
        Args:
            managed_agents: 管理的子代理
            additional_authorized_imports: 额外授权的导入模块
            instructions: 额外指令
            stream_outputs: 是否流式输出
        """
        # 在调用父类初始化之前设置属性，因为父类初始化会调用_get_default_system_prompt
        self.managed_agents = managed_agents or []
        self.additional_authorized_imports = additional_authorized_imports or []
        self.instructions = instructions or ""
        self.stream_outputs = stream_outputs
        
        super().__init__(tools, model, max_steps, **kwargs)
        
        # 确保有Python解释器工具
        if not any(isinstance(tool, PythonInterpreterTool) for tool in self.tools):
            self.tools.append(PythonInterpreterTool())
            self.tool_manager.register_tool(PythonInterpreterTool())
        
        # 管理的代理字典
        self.managed_agents_dict = {agent.name: agent for agent in self.managed_agents}
    
    def _get_default_system_prompt(self) -> str:
        """获取代码代理的默认系统提示"""
        base_prompt = f"""You are {self.name}, an AI coding assistant that can write and execute Python code.

{self.description}

You have access to the following tools:
{self._format_tools_description()}

You can also delegate tasks to the following managed agents:
{self._format_managed_agents_description()}

{self.instructions}

Always break down complex problems into smaller steps and use code to solve them systematically."""
        
        return base_prompt
    
    def _format_managed_agents_description(self) -> str:
        """格式化管理代理描述"""
        if not self.managed_agents:
            return "None"
        
        descriptions = []
        for agent in self.managed_agents:
            descriptions.append(f"- {agent.name}: {agent.description}")
        return "\n".join(descriptions)
    
    def run(self, task: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """运行代码代理"""
        self.memory.clear()
        additional_args = additional_args or {}
        
        # 构建系统消息
        system_prompt = self.prompt_templates["system_prompt"]
        if additional_args.get("additional_knowledge"):
            system_prompt += f"\n\nAdditional knowledge:\n{additional_args['additional_knowledge']}"
        
        # 添加管理代理相关指令
        if additional_args.get("manager_suggestion"):
            system_prompt += f"\n\nManager suggestion: {additional_args['manager_suggestion']}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task}
        ]
        
        for step_num in range(self.max_steps):
            if self.verbosity_level > 0:
                print(f"Code Agent Step {step_num + 1}/{self.max_steps}")
            
            try:
                # 生成响应
                response_text = self.model(messages)
                messages.append({"role": "assistant", "content": response_text})
                
                if self.verbosity_level > 1:
                    print(f"Agent thinking: {response_text}")
                
                # 检查是否有代码块需要执行
                code_blocks = self._extract_code_blocks(response_text)
                
                if code_blocks:
                    # 执行代码块
                    execution_results = []
                    for code in code_blocks:
                        if self.verbosity_level > 1:
                            print(f"Executing code:\n{code}")
                        
                        # 使用Python解释器工具执行代码
                        python_tool = next(
                            (tool for tool in self.tools if isinstance(tool, PythonInterpreterTool)),
                            None
                        )
                        
                        if python_tool:
                            result = python_tool.forward(code)
                            execution_results.append(result)
                            
                            if self.verbosity_level > 1:
                                print(f"Execution result: {result}")
                        else:
                            execution_results.append("Error: Python interpreter not available")
                    
                    # 将执行结果添加到消息历史
                    if execution_results:
                        results_text = "\n\n".join(execution_results)
                        messages.append({"role": "user", "content": f"Code execution results:\n{results_text}"})
                        
                        # 记录动作步骤
                        action_step = ActionStep(
                            tool_calls=[{"function": "python_interpreter", "arguments": {"code": code}} for code in code_blocks],
                            observations=execution_results,
                            model_output=response_text,
                            agent_name=self.name
                        )
                        self.memory.add_step(action_step)
                        self._call_step_callbacks(action_step)
                
                # 检查是否提到了管理的代理
                agent_delegation = self._check_agent_delegation(response_text, additional_args)
                if agent_delegation:
                    agent_name, delegated_task = agent_delegation
                    if agent_name in self.managed_agents_dict:
                        if self.verbosity_level > 1:
                            print(f"Delegating to {agent_name}: {delegated_task}")
                        
                        # 委托任务给子代理
                        sub_agent = self.managed_agents_dict[agent_name]
                        delegation_args = {}
                        
                        # 传递搜索建议给搜索代理
                        if agent_name == "search_agent" and additional_args.get("search_suggestion"):
                            delegation_args["search_suggestion"] = additional_args["search_suggestion"]
                        
                        result = sub_agent.run(delegated_task, delegation_args)
                        
                        # 将结果添加到消息历史
                        messages.append({"role": "user", "content": f"Result from {agent_name}: {result}"})
                        
                        # 记录任务步骤
                        task_step = TaskStep(
                            task=delegated_task,
                            assigned_agent=agent_name,
                            result=result,
                            model_output=response_text,
                            agent_name=self.name
                        )
                        self.memory.add_step(task_step)
                        self._call_step_callbacks(task_step)
                
                # 检查是否有最终答案
                final_answer = self._extract_final_answer(response_text)
                if final_answer:
                    return final_answer
                
                # 如果没有代码执行或代理委托，继续下一步
                if not code_blocks and not agent_delegation:
                    # 添加一个提示让代理继续
                    messages.append({"role": "user", "content": "Please continue or provide your final answer."})
                    
            except Exception as e:
                error_msg = f"Error in code agent step {step_num + 1}: {str(e)}"
                logger.error(error_msg)
                
                # 记录错误步骤
                error_step = ActionStep(
                    error=error_msg,
                    agent_name=self.name
                )
                self.memory.add_step(error_step)
                self._call_step_callbacks(error_step)
                
                return f"Code agent encountered an error: {error_msg}"
        
        # 如果达到最大步数，尝试获取最后的响应作为答案
        if messages and messages[-1]["role"] == "assistant":
            return messages[-1]["content"]
        
        return f"Code agent reached maximum steps ({self.max_steps}) without completing the task"
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """提取代码块"""
        # 匹配```python ... ```格式的代码块
        pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]
    
    def _extract_final_answer(self, text: str) -> Optional[str]:
        """提取最终答案"""
        # 查找各种最终答案标记
        patterns = [
            r'Final answer[:\s]*(.+)',
            r'The answer is[:\s]*(.+)',
            r'Answer[:\s]*(.+)',
            r'Result[:\s]*(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _check_agent_delegation(self, text: str, additional_args: Dict[str, Any]) -> Optional[tuple]:
        """检查是否需要委托给其他代理"""
        # 检查是否提到了搜索代理
        if "search_agent" in text.lower() or "search for" in text.lower():
            # 提取搜索任务
            search_patterns = [
                r'search_agent[:\s]*(.+)',
                r'search for[:\s]*(.+)',
                r'find information about[:\s]*(.+)'
            ]
            
            for pattern in search_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    task = match.group(1).strip()
                    return ("search_agent", task)
            
            # 如果没有找到具体任务，使用整个文本作为搜索查询
            return ("search_agent", "Search for relevant information")
        
        return None
