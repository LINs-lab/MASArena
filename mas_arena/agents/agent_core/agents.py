"""
代理实现 - 多步骤代理系统

这个模块实现了各种类型的代理，包括基础的多步骤代理、代码代理和工具调用代理。
"""

import asyncio
import json
import re
import ast
import sys
from io import StringIO
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import logging
import multiprocessing
import threading

from .steps import ActionStep, PlanningStep, TaskStep, StepMemory
from .models import OpenAIServerModel
from smolagents import Tool
from mas_arena.tools.tool_manager import ToolManager
from mas_arena.tools.python_interpreter import PythonInterpreterTool
from func_timeout import func_timeout, FunctionTimedOut

logger = logging.getLogger(__name__)


def _run_coroutine_sync(coro):
    """Run an async tool/agent from synchronous CodeAgent execution."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def runner():
        try:
            result_box["result"] = asyncio.run(coro)
        except BaseException as exc:
            error_box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("result")


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
        self.name = getattr(self, "name", self.__class__.__name__)
        self.description = getattr(self, "description", "")

        # 提示模板
        self.prompt_templates = {
            "system_prompt": self._get_default_system_prompt(),
            "managed_agent": {"task": ""},
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

    def _safe_parse_tool_arguments(self, raw_args: Union[str, Dict[str, Any], None]) -> Dict[str, Any]:
        """
        Tool-call arguments should be strict JSON strings, but models sometimes emit malformed JSON.
        This parser is best-effort and should never raise; it returns {} on failure.
        """
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if not isinstance(raw_args, str):
            return {}

        # 1) Fast path: strict JSON
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        # 2) Extract likely JSON object substring
        s = raw_args.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]

        # 3) Remove trailing commas before closing braces/brackets
        try:
            s2 = re.sub(r",\s*([}\]])", r"\1", s)
            parsed = json.loads(s2)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        # 4) Python-literal fallback (handles single quotes / True/False/None)
        try:
            s3 = s
            s3 = re.sub(r"\bnull\b", "None", s3)
            s3 = re.sub(r"\btrue\b", "True", s3, flags=re.IGNORECASE)
            s3 = re.sub(r"\bfalse\b", "False", s3, flags=re.IGNORECASE)
            parsed = ast.literal_eval(s3)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def run(self, task: str, additional_args: Optional[Dict[str, Any]] = None) -> str:
        """运行工具调用代理"""
        self.memory.clear()
        additional_args = additional_args or {}

        # 构建系统消息
        system_prompt = self.prompt_templates["system_prompt"]
        if additional_args.get("additional_knowledge"):
            system_prompt += (
                f"\n\nAdditional knowledge:\n{additional_args['additional_knowledge']}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for step_num in range(self.max_steps):
            if self.verbosity_level > 0:
                print(f"Step {step_num + 1}/{self.max_steps}")

            try:
                # 调用模型生成工具调用
                tools_schema = self.tool_manager.get_tools_schema()
                
                # Attempt to use async generation if available
                if hasattr(self.model, "agenerate_with_tools"):
                    response = await self.model.agenerate_with_tools(messages, tools_schema)
                else:
                    # Fallback to generate_with_tools
                    response = self.model.generate_with_tools(messages, tools_schema)
                    # If it happens to return a coroutine (e.g. wrapped async method), await it
                    if asyncio.iscoroutine(response):
                        response = await response

                assistant_message = response.choices[0].message
                messages.append(assistant_message.dict())

                # 检查是否有工具调用
                if assistant_message.tool_calls:
                    tool_calls = []
                    observations = []

                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = self._safe_parse_tool_arguments(
                            getattr(tool_call.function, "arguments", None)
                        )

                        logger.info(
                            "AGENT_TOOL_CALL agent=%s step=%d name=%s args=%s",
                            self.name,
                            step_num + 1,
                            function_name,
                            function_args,
                        )

                        if self.verbosity_level > 1:
                            print(
                                f"Calling tool: {function_name} with args: {function_args}"
                            )

                        # 如果参数解析失败，避免直接炸整个 step；把错误作为 observation 喂回去让模型自我修复
                        if not function_args and getattr(tool_call.function, "arguments", None):
                            result = (
                                "Error: Failed to parse tool arguments as JSON. "
                                f"tool={function_name}, raw_arguments={tool_call.function.arguments!r}"
                            )
                        else:
                            # 执行工具调用
                            result = self.tool_manager.execute_tool(
                                function_name, **function_args
                            )
                        
                        if asyncio.iscoroutine(result):  # 检查是否是异步函数
                            result = await result
                        result_str = str(result)

                        logger.info(
                            "AGENT_TOOL_RESULT agent=%s step=%d name=%s len=%d",
                            self.name,
                            step_num + 1,
                            function_name,
                            len(result_str),
                        )
                        
                        tool_calls.append(
                            {
                                "function": function_name,
                                "arguments": function_args,
                                "result": result_str,
                            }
                        )
                        observations.append(result)

                        # 添加工具结果到消息历史
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            }
                        )

                        # 检查是否是最终答案
                        if function_name == "final_answer":
                            # 记录步骤
                            action_step = ActionStep(
                                tool_calls=tool_calls,
                                observations=observations,
                                model_output=result,
                                agent_name=self.name,
                            )
                            self.memory.add_step(action_step)
                            self._call_step_callbacks(action_step)
                            return result

                    # 记录动作步骤
                    action_step = ActionStep(
                        tool_calls=tool_calls,
                        observations=observations,
                        model_output=assistant_message.content,
                        agent_name=self.name,
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
                error_step = ActionStep(error=error_msg, agent_name=self.name)
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
        **kwargs,
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

        # Prepare globals for Python interpreter:
        # - tool wrappers (e.g. `inspect_file_as_image(...)`)
        # - managed agent wrappers (e.g. `search_agent(...)`)
        additional_globals: Dict[str, Any] = {}

        def _is_valid_python_identifier(name: str) -> bool:
            return bool(re.match(r"^[A-Za-z_]\w*$", name or ""))

        def create_tool_wrapper(tool_name: str):
            def tool_wrapper(*args, **kwargs):
                result = self.tool_manager.execute_tool(tool_name, *args, **kwargs)
                if asyncio.iscoroutine(result):
                    return _run_coroutine_sync(result)
                return result

            return tool_wrapper

        # Expose tools to the python interpreter environment.
        # Note: avoid exposing `python_interpreter` itself (recursion) and `final_answer` (prompt rule).
        for t in self.tools:
            t_name = getattr(t, "name", None)
            if not t_name or not isinstance(t_name, str):
                continue
            if t_name in ("python_interpreter", "final_answer"):
                continue
            if not _is_valid_python_identifier(t_name):
                continue
            if t_name in additional_globals:
                continue
            additional_globals[t_name] = create_tool_wrapper(t_name)

        if self.managed_agents:
            for agent in self.managed_agents:
                # Create a wrapper function that calls agent.run
                def create_wrapper(agent_instance):
                    def agent_wrapper(task: str, **kwargs):
                        # print(f"Calling managed agent '{agent_instance.name}' with task: {task}")
                        import asyncio
                        try:
                            # If we are in a thread without a loop (asyncio.to_thread), use asyncio.run
                            return asyncio.run(agent_instance.run(task, additional_args=kwargs))
                        except RuntimeError:
                            # Fallback if somehow there is a loop or other issue
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            return loop.run_until_complete(
                                agent_instance.run(task, additional_args=kwargs)
                            )
                    return agent_wrapper
                
                additional_globals[agent.name] = create_wrapper(agent)

        # 确保有Python解释器工具
        python_tool_found = False
        for tool in self.tools:
            if isinstance(tool, PythonInterpreterTool):
                python_tool_found = True
                # Inject globals into existing tool
                if hasattr(tool, "additional_globals"):
                    tool.additional_globals.update(additional_globals)
                else:
                    tool.additional_globals = additional_globals
                # Ensure imports are set
                if hasattr(tool, "authorized_imports"):
                    # Union of existing and new imports
                    tool.authorized_imports = list(
                        set(tool.authorized_imports + self.additional_authorized_imports)
                    )
                break
        
        if not python_tool_found:
            self.tools.append(PythonInterpreterTool(
                authorized_imports=self.additional_authorized_imports,
                additional_globals=additional_globals
            ))
            self.tool_manager.register_tool(self.tools[-1])

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

Always break down complex problems into smaller steps and use code to solve them systematically.

IMPORTANT: FORMATTING RULES
1. When you want to use a tool or perform a calculation, you MUST write Python code inside a code block.
2. The format MUST be exactly:
```python
# Your code here
result = tool_name(arg1="value", arg2=123)
print(result)
```
3. DO NOT use plain text actions like "Action: tool_name" or "I will use tool_name". These will be ignored.
4. You MUST use print() to output the final result of your calculation or tool usage.
5. Separation of Concerns: Never use a final answer tool (e.g., final_answer()) inside a Python code block.
"""

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
            system_prompt += (
                f"\n\nAdditional knowledge:\n{additional_args['additional_knowledge']}"
            )

        # 添加管理代理相关指令
        if additional_args.get("manager_suggestion"):
            system_prompt += (
                f"\n\nManager suggestion: {additional_args['manager_suggestion']}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
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
                # 检查FinalAnswer 工具调用    
                tool_final_answer = self._extract_final_answer_tool_call(response_text)
                
                if tool_final_answer:
                    if self.verbosity_level > 0:
                        print(f"CodeAgent returning final answer from tool call: {tool_final_answer}")
                    # 记录工具调用步骤（可选，但推荐）
                    action_step = ActionStep(
                        tool_calls=[
                            {
                                "function": "final_answer",
                                "arguments": {"answer": tool_final_answer},
                            }
                        ],
                        observations=[f"Final answer provided: {tool_final_answer}"],
                        model_output=response_text,
                        agent_name=self.name,
                    )
                    self.memory.add_step(action_step)
                    self._call_step_callbacks(action_step)
                    
                    return tool_final_answer # 立即终止循环并返回答案

                # 关键：如果已经显式给出 <answer>...</answer>（或 \\boxed{...}），
                # 直接返回，避免其中包含的 ```python``` 被误当作“待执行代码块”。
                strict_final = self._extract_final_answer(response_text, strict=True)
                if strict_final:
                    if self.verbosity_level > 0:
                        print(
                            f"CodeAgent returning strict final answer found in text: {strict_final}"
                        )
                    return strict_final
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
                            (
                                tool
                                for tool in self.tools
                                if isinstance(tool, PythonInterpreterTool)
                            ),
                            None,
                        )

                        if python_tool:
                            timeout = 30
                            try:
                                # 直接调用，设置超时时间
                                result = func_timeout(timeout, python_tool.forward, args=(code,))
                            except FunctionTimedOut:
                                result = f"Error: Execution timed out after {timeout} seconds. Your code took too long to run. Please optimize it."
                            except Exception as e:
                                result = f"Error executing code: {str(e)}\nPlease check your code logic and syntax."
                            # with multiprocessing.Pool(processes=1) as pool:
                            #     result_obj = pool.apply_async(python_tool.forward, (code,))
                            #     try:
                            #         # get() 会阻塞直到结果返回或超时
                            #         result = result_obj.get(timeout=timeout)
                            #     except multiprocessing.TimeoutError:
                            #         result = f"Error: Execution timed out after {timeout} seconds."
                            #     except Exception as e:
                            #         result = f"Error: {str(e)}"
                            execution_results.append(result)

                            if self.verbosity_level > 1:
                                print(f"Execution result: {result}")
                        else:
                            execution_results.append(
                                "Error: Python interpreter not available"
                            )

                    # 将执行结果添加到消息历史
                    if execution_results:
                        results_text = "\n\n".join(str(result) for result in execution_results)
                        error_feedback = self._build_error_feedback(execution_results)
                        feedback_suffix = f"\n\n{error_feedback}" if error_feedback else ""
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Code execution results:\n{results_text}{feedback_suffix}",
                            }
                        )

                        # 记录动作步骤
                        action_step = ActionStep(
                            tool_calls=[
                                {
                                    "function": "python_interpreter",
                                    "arguments": {"code": code},
                                }
                                for code in code_blocks
                            ],
                            observations=execution_results,
                            model_output=response_text,
                            agent_name=self.name,
                        )
                        self.memory.add_step(action_step)
                        self._call_step_callbacks(action_step)
                        continue

                # 检查是否提到了管理的代理
                agent_delegation = self._check_agent_delegation(
                    response_text, additional_args
                )
                if agent_delegation:
                    agent_name, delegated_task = agent_delegation
                    if agent_name in self.managed_agents_dict:
                        if self.verbosity_level > 1:
                            print(f"Delegating to {agent_name}: {delegated_task}")

                        # 委托任务给子代理
                        sub_agent = self.managed_agents_dict[agent_name]
                        delegation_args = {}

                        # 传递搜索建议给搜索代理
                        if agent_name == "search_agent" and additional_args.get(
                            "search_suggestion"
                        ):
                            delegation_args["search_suggestion"] = additional_args[
                                "search_suggestion"
                            ]

                        result = sub_agent.run(delegated_task, delegation_args)
                        if asyncio.iscoroutine(result):
                            result = _run_coroutine_sync(result)

                        # 将结果添加到消息历史
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Result from {agent_name}: {result}",
                            }
                        )

                        # 记录任务步骤
                        task_step = TaskStep(
                            task=delegated_task,
                            assigned_agent=agent_name,
                            result=result,
                            model_output=response_text,
                            agent_name=self.name,
                        )
                        self.memory.add_step(task_step)
                        self._call_step_callbacks(task_step)
                        continue

                # 检查是否有最终答案
                final_answer = self._extract_final_answer(response_text)
                if final_answer:
                    if self.verbosity_level > 0:
                        print(f"CodeAgent returning final answer found in text: {final_answer}")
                    return final_answer

                # 如果没有代码执行或代理委托，继续下一步
                if not code_blocks and not agent_delegation:
                    # 检查是否有潜在的工具调用意图但格式错误
                    # 匹配常见的 ReAct 模式或其他明显的工具调用意图
                    potential_tool_call = re.search(
                        r"(?:Action|Call|Use|Using)\s*:?\s*[`'\"]?(\w+)[`'\"]?", 
                        response_text, 
                        re.IGNORECASE
                    )
                    
                    # 检查是否提到了具体的工具名称
                    mentioned_tools = [tool.name for tool in self.tools if tool.name in response_text]
                    
                    tool_hint = ""
                    should_provide_feedback = False

                    if mentioned_tools:
                        tool_hint = f" regarding '{mentioned_tools[0]}'"
                        should_provide_feedback = True
                    elif potential_tool_call:
                        matched_word = potential_tool_call.group(1)
                        # 过滤常用停用词
                        if matched_word.lower() not in ["the", "a", "an", "this", "that", "it", "to", "python"]:
                            tool_hint = f" regarding '{matched_word}'"
                            should_provide_feedback = True
                    
                    if should_provide_feedback:
                        feedback_msg = (
                            f"I detected a potential intent to use a tool{tool_hint}, but I did not find any executable code block. "
                            "You MUST write Python code inside a ```python``` block to use tools. "
                            "Do not use 'Action:' or plain text descriptions. "
                            "Example:\n```python\nprint(tool_name(arg='value'))\n```\n"
                            "Please rewrite your response with the correct code block format."
                        )
                        if self.verbosity_level > 1:
                            print(f"Providing format feedback: {feedback_msg}")
                            
                        messages.append({"role": "user", "content": feedback_msg})
                    else:
                        # 通用提示，但也强调代码格式
                        messages.append(
                            {
                                "role": "user",
                                "content": "I did not detect any executable code or tool calls. If you intended to use a tool, please use a ```python code block. Otherwise, please continue or provide your final answer.",
                            }
                        )

            except Exception as e:
                error_msg = f"Error in code agent step {step_num + 1}: {str(e)}"
                logger.error(error_msg)

                # 记录错误步骤
                error_step = ActionStep(error=error_msg, agent_name=self.name)
                self.memory.add_step(error_step)
                self._call_step_callbacks(error_step)

                return f"Code agent encountered an error: {error_msg}"

        return self._force_final_answer(messages)

    def _force_final_answer(self, messages: List[Dict[str, Any]]) -> str:
        """
        If the normal loop exhausts max_steps, do one final no-tool model call.
        This mirrors the intended benchmark behavior: return the best answer
        available instead of a generic step-limit failure.
        """
        fallback = ""
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                fallback = str(message["content"])
                break

        final_prompt = (
            f"You have reached the maximum step budget ({self.max_steps}). "
            "Do not call any tools and do not write more Python code. "
            "Use only the evidence already present in the conversation, make the best possible guess, "
            "and return exactly one concise final answer wrapped as <answer>...</answer>."
        )

        try:
            response_text = self.model(messages + [{"role": "user", "content": final_prompt}])
            forced_answer = (
                self._extract_final_answer_tool_call(response_text)
                or self._extract_final_answer(response_text, strict=True)
                or self._extract_final_answer(response_text)
                or response_text.strip()
            )
            action_step = ActionStep(
                tool_calls=[],
                observations=[f"Forced final answer after max steps: {forced_answer}"],
                model_output=response_text,
                agent_name=self.name,
            )
            self.memory.add_step(action_step)
            self._call_step_callbacks(action_step)
            logger.info("MAX_STEPS_FORCED_FINAL agent=%s answer=%s", self.name, forced_answer)
            return forced_answer
        except Exception as e:
            error_msg = f"Error forcing final answer after max steps: {str(e)}"
            logger.error(error_msg)
            error_step = ActionStep(error=error_msg, agent_name=self.name)
            self.memory.add_step(error_step)
            self._call_step_callbacks(error_step)
            return fallback or f"Code agent reached maximum steps ({self.max_steps}) without completing the task"

    def _build_error_feedback(self, execution_results: List[str]) -> str:
        """Tell the model when it is repeating a failing action."""
        current_errors = [
            str(result).strip()
            for result in execution_results
            if result is not None
            and (
                str(result).strip().lower().startswith("error")
                or "timed out" in str(result).lower()
                or "coroutine object" in str(result).lower()
            )
        ]
        if not current_errors:
            return ""

        prior_observations = []
        for step in self.memory.get_steps(ActionStep):
            prior_observations.extend(str(obs).strip() for obs in getattr(step, "observations", []) or [])

        repeated = False
        for error in current_errors:
            error_key = error[:240]
            if any(obs[:240] == error_key for obs in prior_observations):
                repeated = True
                break

        if not repeated:
            return ""

        return (
            "The same error has already occurred before. Do NOT retry the same code, tool, URL, or query. "
            "Use a different strategy, reduce the scope of the request, rely on evidence already gathered, "
            "or provide your best final answer if external tools keep failing."
        )

    def _extract_code_blocks(self, text: str) -> List[str]:
        """提取代码块"""
        # 匹配```python ... ```格式的代码块
        pattern = r"```(?:python)?\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _extract_final_answer_tool_call(self, text: str) -> Optional[str]:
        """
        专门用于从模型输出中解析 final_answer 工具调用，
        并提取其 'answer' 参数。
        
        匹配模式示例：
        1. final_answer("The result is 42.")
        2. final_answer(answer="The result is 42.")
        3. Final_Answer('The result is 42.')
        """
        
        # 匹配 final_answer(answer="...") 或 final_answer("...") 
        # 使用非贪婪匹配 (.*?) 来防止匹配到多个答案
        # re.I 使匹配不区分大小写 (e.g., Final_Answer)
        
        # 模式1: final_answer("...") 或 final_answer(answer="...")
        pattern_quotes = r'final_answer\s*\((?:answer\s*=\s*)?["\'](.*?)["\']\s*\)'
        
        # 模式2: final_answer(answer=valid) - 匹配不带引号的简单答案（虽然不推荐，但为兼容性考虑）
        # 由于不带引号的答案容易出错，我们主要依赖带引号的模式1
        
        match = re.search(pattern_quotes, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            answer = match.group(1).strip()
            # 若 agent 传入的是 final_answer("<answer>X</answer>")，只保留 X
            tag_inner = re.search(r"<answer>\s*(.*?)\s*</answer>", answer, re.IGNORECASE | re.DOTALL)
            if tag_inner:
                answer = tag_inner.group(1).strip()
            if self.verbosity_level > 1:
                print(f"DEBUG: Extracted final answer via tool call pattern: {answer}")
            return answer
            
        return None
    
    def _extract_final_answer(self, text: str, strict: bool = False) -> Optional[str]:
        """提取最终答案

        strict=True 时：仅允许显式格式（如 <answer>...</answer> 或 \\boxed{...}），
        不使用“Final answer:”等弱匹配，以避免在代码生成/验证阶段误判。
        """
        # 优先捕捉常见的格式化答案
        boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", text, re.DOTALL)
        if boxed_matches:
            if self.verbosity_level > 1:
                print(
                    f"DEBUG: Extracted final answer via \\boxed: {boxed_matches[-1].strip()}"
                )
            return boxed_matches[-1].strip()

        tag_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL
        )
        if tag_match:
            if self.verbosity_level > 1:
                print(
                    f"DEBUG: Extracted final answer via <answer> tag: {tag_match.group(1).strip()}"
                )
            return tag_match.group(1).strip()

        if strict:
            return None

        # 弱匹配模式：只取冒号后第一行或第一句，避免把整段推理当答案（如 "Answer: must be a single number. No loopholes..."）
        # 使用 [^\n]+ 限制为单行，避免 DOTALL 下 (.+) 吃掉全文
        patterns_single_line = [
            r"Final answer[:\s]*([^\n]+)",
            r"The answer is[:\s]*([^\n]+)",
            r"\bAnswer\b[:\s]*([^\n]+)",
            r"\bResult\b[:\s]*([^\n]+)",
        ]
        # 明显是推理/下一步的短语，不应作为最终答案
        reasoning_markers = (
            "next step",
            "i will",
            "let me",
            "therefore",
            "analysis",
            "did not return",
            "no loopholes",
            "depends entirely",
            "i would",
            "we need to",
            "must be a",
            "must be the",
        )

        for pattern in patterns_single_line:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group(1).strip()
                if not raw or len(raw) > 500:
                    continue
                lower = raw.lower()
                if any(m in lower for m in reasoning_markers):
                    continue
                if self.verbosity_level > 1:
                    print(
                        f"DEBUG: Extracted final answer via pattern '{pattern}': {raw}"
                    )
                return raw

        return None

    def _check_agent_delegation(
        self, text: str, additional_args: Dict[str, Any]
    ) -> Optional[tuple]:
        """检查是否需要委托给其他代理"""
        # 检查是否提到了搜索代理
        if "search_agent" in text.lower() or "search for" in text.lower():
            # 提取搜索任务
            search_patterns = [
                r"search_agent[:\s]*(.+)",
                r"search for[:\s]*(.+)",
                r"find information about[:\s]*(.+)",
            ]

            for pattern in search_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    task = match.group(1).strip()
                    return ("search_agent", task)

            # 如果没有找到具体任务，使用整个文本作为搜索查询
            return ("search_agent", "Search for relevant information")

        return None
