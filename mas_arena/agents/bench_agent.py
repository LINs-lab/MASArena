"""
BenchAgent - 高效易用的评测智能体

这个模块实现了一个简化的、易于使用的智能体接口，支持插拔式工具配置、可选内存系统，以及统一的基准测试接口。

Examples:
    >>> agent = BenchAgent(
    ...     model="gpt-4o-mini",
    ...     manager_tools=["python_interpreter", "final_answer"],
    ...     search_tools=["search", "browser"],
    ...     memory="melo_memory",
    ...     api_key="your-api-key"
    ... )
    >>> result = await agent.evaluate({"problem": "What is 2+2?", "solution": "4"}, evaluator_name="math")
    >>> print(result.final_answer)
"""

import os
import time
import re
from typing import Dict, Any, Optional, List, Union
import contextlib
from dataclasses import dataclass
from string import Template
from typing_extensions import override
from openai.types.completion_usage import CompletionUsage
import yaml
import asyncio

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.reformulator import prepare_response, truncate_observation
from mas_arena.utils.openai_compat import normalize_openai_api_base
from mas_arena.utils.llm_utils import RetryWrapper
from mas_arena.utils.env import (
    DEFAULT_MODEL_NAME,
    get_env_float,
    get_env_int,
    get_model_name,
    get_openai_api_base,
)

# 导入外部工具（统一管理，避免重复导入）
from mas_arena.tools import (
    # 媒体工具
    AudioInspectorTool, VisualInspectorTool,
    # 网络工具  
    SearchTool, SimpleCrawler, CrawlerArchiveSearchTool, CrawlerReadTool, WikipediaSearchTool,
    # 文档工具
    CSVExtractorTool, MarkdownConverterTool, SheetExtractorTool, 
    TextExtractorTool, ZipExtractorTool,
    # masarena工具
    FinalAnswerTool, 
    # 本地工具
    PythonInterpreterTool,
    # 工具集合
    ALL_EXTERNAL_TOOLS,
)

# 使用本地实现替代smolagents
from mas_arena.agents.agent_core import (
    ActionStep,
    CodeAgent,
    MultiStepAgent,
    PlanningStep,
    TaskStep,
    ToolCallingAgent,
    OpenAIServerModel,
    Tool,
)

from mas_arena.agents.base import logger
from mas_arena.agents.workflow_protocol import request_settings

# 加载prompts
prompts_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
    "agents_prompts.yaml",
)
with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.load(f, Loader=yaml.FullLoader)

# 授权的Python导入
AUTHORIZED_IMPORTS = [
    "requests", "zipfile", "os", "pandas", "numpy", "sympy", "json", "bs4", "xml",
    "pydub", "io", "PIL", "PyPDF2", "pptx", "datetime", "fractions", "csv", 
    "random", "re", "sys", "shutil",
]


@dataclass
class BenchWorker:
    """包装smolagents agents的工作器"""
    name: str
    agent: MultiStepAgent
    llm: OpenAIServerModel


class BenchAgent(AgentSystem):
    """
    高效易用的基准测试智能体
    
    这个类提供了一个简化的接口来创建和使用多智能体系统进行基准测试。
    用户可以通过构造函数直接配置所有组件，无需复杂的配置文件。
    """
    
    # 预定义的工具映射 - 整合本地工具和smolagents工具
    AVAILABLE_TOOLS = {
        # 核心工具（本地实现）
        "python_interpreter": PythonInterpreterTool,
        "final_answer": FinalAnswerTool,
        "wikipedia": WikipediaSearchTool,
        
        # 外部工具（通过工具字典自动添加）
        **ALL_EXTERNAL_TOOLS,
    }
    
    def __init__(
        self,
        model: Optional[str] = None,
        manager_tools: Optional[List[Union[str, Tool]]] = None,
        search_tools: Optional[List[Union[str, Tool]]] = None,
        memory: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_steps: Optional[int] = None,
        search_max_steps: Optional[int] = None,
        verbosity_level: Optional[int] = None,
        additional_instructions: Optional[str] = None,
        name: str = "bench_agent",
        **kwargs
    ):
        """
        初始化BenchAgent
        
        Args:
            model: 语言模型名称，默认为"gpt-4o-mini"
            manager_tools: 管理代理使用的工具列表，可以是工具名称字符串或Tool对象
            search_tools: 搜索代理使用的工具列表，可以是工具名称字符串或Tool对象  
            memory: 内存类型，如"melo_memory", "memory_bank"等
            api_key: API密钥，如果不提供则从环境变量获取
            api_base: API基础URL，如果不提供则使用默认值
            max_steps: 管理代理最大步数
            search_max_steps: 搜索代理最大步数
            verbosity_level: 详细程度级别
            additional_instructions: 额外的指令
            name: 代理名称
            **kwargs: 其他配置参数
        """
        # Explicit arguments override registry values; omitted defaults do not.
        registry_config = dict(kwargs.pop("config", {}) or {})
        final_config = {
            "model_name": get_model_name(DEFAULT_MODEL_NAME), "max_steps": 15,
            "search_max_steps": 10, "verbosity_level": 2,
            **registry_config, **kwargs,
        }
        explicit = {
            "model_name": model, "api_key": api_key, "api_base": api_base,
            "max_steps": max_steps, "search_max_steps": search_max_steps,
            "verbosity_level": verbosity_level, "additional_instructions": additional_instructions,
        }
        final_config.update({key: value for key, value in explicit.items() if value is not None})
        self.benchmark_name = final_config.get("evaluator")
        super().__init__(name, final_config)
        if manager_tools is None:
            manager_tools = final_config.get("manager_tools")
        if search_tools is None:
            search_tools = final_config.get("search_tools")

        self.manager_tools_config = manager_tools if manager_tools is not None else ["python_interpreter"]
        if search_tools is None:
            if self.benchmark_name in {"hotpotqa", "gaia"}:
                search_tools = [
                    "audio_inspector", "visual_inspector", "search", "browser", "wikipedia",
                    "crawler_read", "crawler_archive_search", "csv_extractor", "markdown_converter",
                    "sheet_extractor", "text_extractor", "zip_extractor",
                ]
            else:
                search_tools = []
        self.search_tools_config = search_tools
        self.memory_type = memory if memory is not None else final_config.get("memory")
        
        # 初始化组件
        self._initialize_model()
        self._initialize_tools()
        
        # 创建代理层次结构
        agent_components = self._create_agents()
        self.workers = agent_components["workers"]
        
        # 获取主要代理引用
        self.agent = next((w.agent for w in self.workers if isinstance(w.agent, CodeAgent)), None)
        self.search_agent = next(
            (w.agent for w in self.workers if isinstance(w.agent, ToolCallingAgent)), None
        )
        
        if not self.agent:
            raise ValueError("Could not find CodeAgent (manager) in created workers")
        if not self.search_agent:
            raise ValueError("Could not find ToolCallingAgent (search) in created workers")
        
        # 初始化内存（如果指定）
        if self.memory_type:
            self._initialize_memory()
        
        # 执行历史
        self.conversation_history = []
        self.execution_log = []

    def _initialize_model(self):
        """初始化语言模型"""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        api_base = normalize_openai_api_base(
            self.config.get("api_base") or get_openai_api_base(),
            "https://api.openai.com/v1",
        )
        model_name = self.config.get("model_name") or get_model_name(DEFAULT_MODEL_NAME)
        
        if not api_key:
            raise ValueError(
                "API key is required. Provide it in constructor or set OPENAI_API_KEY environment variable."
            )
        
        parameters = request_settings(self.config)
        parameters["max_completion_tokens"] = parameters.pop("max_tokens")
        self.llm = RetryWrapper(OpenAIServerModel(
            model_id=model_name,
            custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
            api_base=api_base,
            api_key=api_key,
            timeout=int(self.config.get("timeout", os.getenv("OPENAI_API_TIMEOUT", "300"))),
            **parameters,
        ))

    def _initialize_tools(self):
        """根据配置初始化工具"""
        # 初始化管理工具
        self.manager_tools = self._build_tool_list(self.manager_tools_config, is_manager=True)
        
        # 初始化搜索工具
        self.search_tools = self._build_tool_list(self.search_tools_config, is_manager=False)
        
        # 确保必要的工具存在 - FinalAnswerTool 是必须的
        if not any(isinstance(tool, FinalAnswerTool) for tool in self.manager_tools):
            self.manager_tools.append(FinalAnswerTool())
        
        if not any(isinstance(tool, FinalAnswerTool) for tool in self.search_tools):
            self.search_tools.append(FinalAnswerTool())

    def _build_tool_list(self, tool_configs: List[Union[str, Tool]], is_manager: bool = True) -> List[Tool]:
        """根据配置构建工具列表"""
        tools = []
        crawler = SimpleCrawler()  # 为爬虫工具创建共享实例
        
        # 处理 "ALL" 关键字，将其展开为所有可用工具名称
        processed_configs = []
        if isinstance(tool_configs, list):
            for cfg in tool_configs:
                if isinstance(cfg, str) and cfg.upper() == "ALL":
                    processed_configs.extend(self.AVAILABLE_TOOLS.keys())
                else:
                    processed_configs.append(cfg)
        else:
            processed_configs = tool_configs or []

        for tool_config in processed_configs:
            if isinstance(tool_config, Tool):
                # 直接使用Tool对象
                tools.append(tool_config)
            elif isinstance(tool_config, str):
                # 根据字符串名称创建工具
                tool = self._create_tool_by_name(tool_config, crawler, is_manager)
                if tool:
                    tools.append(tool)
                else:
                    logger.warning(f"Unknown tool: {tool_config}")
            else:
                logger.warning(f"Invalid tool configuration: {tool_config}")
        
        return tools

    def _create_tool_by_name(self, tool_name: str, crawler, is_manager: bool = True) -> Optional[Tool]:
        """根据名称创建工具实例"""
        if tool_name not in self.AVAILABLE_TOOLS:
            return None
        
        tool_class = self.AVAILABLE_TOOLS[tool_name]
        
        # 根据工具类型创建实例
        try:
            if tool_name == "crawler_read":
                return CrawlerReadTool(crawler)
            elif tool_name == "crawler_archive_search":
                return CrawlerArchiveSearchTool(crawler)
            elif tool_name in ["arxiv", "markdown_converter", "terminal", "download"]:
                return tool_class(model=self.llm, text_limit=1000)
            elif tool_name == "browser":
                return tool_class() # Browser wrapper doesn't need model arg usually, or has default
            elif tool_name == "text_inspector":
                return tool_class(self.llm, text_limit=100000)
            elif tool_name in ["csv_extractor", "zip_extractor"]:
                return tool_class(model=self.llm, text_limit=1000)
            elif tool_name in ["audio_inspector", "video_inspector", "visual_inspector"]:
                return tool_class(model=self.llm, text_limit=1000)
            elif tool_name == "sheet_extractor":
                return tool_class(model=self.llm)
            elif tool_name == "text_extractor":
                return tool_class(text_limit=1000)
            elif tool_name == "search":
                return SearchTool(reflection=False)
            elif tool_name == "wikipedia":
                return WikipediaSearchTool()
            elif tool_name == "python_interpreter":
                return tool_class(authorized_imports=AUTHORIZED_IMPORTS)
            else:
                # 对于不需要特殊参数的工具
                return tool_class()
        except Exception as e:
            logger.error(f"Failed to create tool {tool_name}: {e}")
            return None

    def _initialize_memory(self):
        """初始化内存系统"""
        try:
            from mas_arena.memory.memory_registry import memory_registry
            self.meta_memory = memory_registry.get(self.memory_type)
            if self.meta_memory:
                logger.info(f"Successfully initialized memory: {self.memory_type}")
            else:
                logger.warning(f"Failed to initialize memory: {self.memory_type}")
        except Exception as e:
            logger.error(f"Error initializing memory {self.memory_type}: {e}")
            self.meta_memory = None

    def _create_agents(self) -> Dict[str, List[BenchWorker]]:
        """创建代理层次结构"""
        # 创建搜索代理
        search_planning_interval_str = os.environ.get("SEARCH_AGENT_PLANNING_INTERVAL")
        search_planning_interval = (
            int(search_planning_interval_str)
            if search_planning_interval_str and search_planning_interval_str.isdigit()
            else None
        )
        
        search_agent = ToolCallingAgent(
            tools=self.search_tools,
            model=self.llm,
            max_steps=self.config.get("search_max_steps", 10),
            verbosity_level=self.config.get("verbosity_level", 1),
            provide_run_summary=True,
            planning_interval=search_planning_interval,
        )
        
        # 兼容性修复 - 本地实现已包含此方法
        # if not hasattr(search_agent, "process_single_tool_call"):
        #     def process_single_tool_call(tool_call):
        #         return search_agent.process_tool_calls([tool_call])
        #     search_agent.process_single_tool_call = process_single_tool_call
        
        search_agent.name = "search_agent"
        search_agent.description = """Specialized web search agent. Handles all web browsing and online information gathering.
Use full sentences for requests, provide context including timeframes when needed."""
        
        # 增强搜索代理的提示模板
        search_agent.prompt_templates["system_prompt"] += """
Your response MUST be a JSON object with the following structure, and nothing else:
{
  "tool": "<name_of_the_tool>",
  "arguments": {
    "<parameter_name>": "<parameter_value>",
    ...
  }
}"""
        search_agent.prompt_templates["managed_agent"]["task"] += prompts["search_agent_prompt_managed_task"]
        
        # 创建管理代理
        manager_system_prompt = prompts["base_manager_prompt"]
        if self.config.get("additional_instructions"):
            manager_system_prompt += "\n\n" + self.config["additional_instructions"]
        
        manager_planning_interval_str = os.environ.get("MANAGE_AGENT_PLANNING_INTERVAL")
        manager_planning_interval = (
            int(manager_planning_interval_str)
            if manager_planning_interval_str and manager_planning_interval_str.isdigit()
            else None
        )
        
        manager_instructions = (
            "You are a manager agent responsible for solving a given task. "
            "You can use the available tools and delegate tasks to a search agent if needed. "
            "When you receive a 'search_suggestion' in your arguments, you should consider it carefully. "
            "If the suggestion is relevant to the search task, you must pass this suggestion to the search_agent via its 'additional_args' parameter."
        )
        
        manager_agent = CodeAgent(
            tools=self.manager_tools,
            model=self.llm,
            max_steps=self.config.get("max_steps", 15),
            verbosity_level=self.config.get("verbosity_level", 1),
            stream_outputs=False,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            managed_agents=[search_agent],
            planning_interval=manager_planning_interval,
            instructions=manager_instructions,
        )
        
        manager_agent.name = "manager_agent"
        manager_agent.description = """Manager agent for computational tasks and information synthesis.
Handles complex multi-step problems, writes Python code, and processes data."""
        manager_agent.prompt_templates["system_prompt"] += manager_system_prompt
        
        # 创建工作器
        search_worker = BenchWorker(name="search_agent", agent=search_agent, llm=self.llm)
        manager_worker = BenchWorker(name="manager_agent", agent=manager_agent, llm=self.llm)
        
        return {"workers": [search_worker, manager_worker]}

    async def aclose(self):
        """关闭资源"""
        if hasattr(self.llm, "aclose"):
            await self.llm.aclose()
        else:
            if hasattr(self.llm, "client"):
                self.llm.client.close()
            if hasattr(self.llm, "async_client"):
                await self.llm.async_client.close()
        for tools_list in (getattr(self, "search_tools", []), getattr(self, "manager_tools", [])):
            for tool in tools_list:
                bi = getattr(tool, "browser_instance", None)
                if bi and hasattr(bi, "close"):
                    try:
                        await asyncio.wait_for(bi.close(), timeout=5)
                    except Exception:
                        pass

    @override
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """运行代理系统，这是从原始smolagents.py复制过来的核心逻辑"""
        usage_start = self._usage_totals()
        search_keywords = ""
        
        # 创建增强的问题描述
        augmented_question = """Answer this question correctly. You have all the tools needed to find the right answer.

Use search_agent for any web research or current information.

Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!


Task:
""" + problem["problem"]
        
        # 处理文件附件（如果存在）
        if "files" in problem and problem["files"]:
            print("Detected files in problem, preparing file descriptions...")
            files = problem["files"]
            if isinstance(files, str):
                files = [files]
            
            if len(files) == 1:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
                prompt_use_files += self._get_file_description(files[0], problem["problem"])
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:"
                for file_path in files:
                    prompt_use_files += self._get_file_description(file_path, problem["problem"])
            
            augmented_question += prompt_use_files
        
        if "context" in kwargs:
            augmented_question += f"\n\nContext: {kwargs['context']}"
        
        return await self.run_agent_step(
            augmented_question, {"id": problem.get("id", "")}, usage_start=usage_start
        )

    async def run_agent_step(self, augmented_question: str, additional_args: Dict[str, Any], *, usage_start=None) -> Dict[str, Any]:
        """
        简化版单步运行（类似 llm.invoke）
        - 输入完整的 prompt（augmented_question）
        - 保留工具调用、记忆检索等能力
        - 不做答案校验/重试，直接返回一次运行结果
        """
        self._usage_start = self._usage_totals() if usage_start is None else usage_start
        self.execution_log = []
        search_keywords = ""
        additional_args = {key: value for key, value in (additional_args or {}).items()
                           if key not in {"solution", "expected_answer", "answer", "reference_answer", "ground_truth"}}
        additional_knowledge = ""

        # 在单步模式也附加格式提示，确保最终答案符合基准要求
        if self.format_prompt:
            augmented_question = f"{augmented_question.rstrip()}\n\n{self.format_prompt.strip()}"

        # 如启用记忆，先生成搜索关键词并检索相关记忆
        if self.meta_memory is not None:
            try:
                search_keywords = await self.llm.acall([{
                    "role": "user", "content": "Generate concise memory-retrieval keywords for this task:\n" + augmented_question,
                }])

                successful_trajectories, _, insights = await self.meta_memory.retrieve_memory(
                    task_search_keywords=search_keywords,
                    task_question=augmented_question,
                    successful_topk=get_env_int("MAS_RL_SUCCESSFUL_TOPK", 2),
                    failed_topk=get_env_int("MAS_RL_FAILED_TOPK", 1),
                    insight_topk=get_env_int("MAS_RL_INSIGHTS_TOPK", 3),
                    threshold=get_env_float("MAS_RL_SIMILARITY_THRESHOLD", 0.3),
                )
                additional_knowledge = "\n\n".join([insight for insight in insights])
                logger.info(
                    f"Retrieved {len(successful_trajectories)} successful trajectories and {len(insights)} insights"
                )
            except Exception as e:
                logger.error(f"Error retrieving memory in run_agent_step: {str(e)}")
                additional_knowledge = None

        # 合并附加知识传递给代理
        additional_args["additional_knowledge"] = additional_knowledge

        # 同步运行代理（包含所有工具调用等）
        try:
            result = await asyncio.to_thread(self._run_agent_sync, augmented_question, additional_args)
            final_answer = self.extract_final_answer(result)

            # 构建对话与步骤摘要
            conversation_messages = self._extract_conversation_history(augmented_question, final_answer, problem_id=additional_args.get("id", ""))
            self.conversation_history.extend(conversation_messages)
            manager_agent_steps, search_agent_steps = self._extract_agent_steps()

            return {
                "messages": conversation_messages,
                "final_answer": final_answer,
                "manager_agent_steps": manager_agent_steps,
                "search_agent_steps": search_agent_steps,
                "search_keywords": search_keywords,
            }
        except Exception as e:
            error_message = f"Error running BenchAgent (step): {str(e)}"
            error_ai_message = {
                "content": error_message,
                "name": "bench_agent_error",
                "role": "assistant",
                "message_type": "error_response",
                "usage_metadata": self._extract_token_usage_from_agent(),
            }
            return {
                "messages": [error_ai_message],
                "final_answer": error_message,
                "error": str(e),
            }

    def _run_agent_sync(self, augmented_question: str, additional_args: Dict[str, Any]) -> Any:
        """同步运行代理"""
        return self.agent.run(augmented_question, additional_args=additional_args)

    def _extract_agent_steps(self) -> tuple:
        """提取代理步骤信息"""
        manager_agent_steps = []
        search_agent_steps = []
        
        # 提取管理代理步骤
        for memory_step in self.agent.memory.steps:
            memory_step.model_input_messages = None
            step_dict = memory_step.dict()
            truncate_observation(step_dict)
            
            if isinstance(memory_step, ActionStep):
                step_dict["step_type"] = "action"
                step_dict.pop("model_output_message", None)
            elif isinstance(memory_step, TaskStep):
                step_dict["step_type"] = "task"
            elif isinstance(memory_step, PlanningStep):
                step_dict["step_type"] = "planning"
                step_dict.pop("model_output_message_facts", None)
                step_dict.pop("model_output_message_plan", None)
            else:
                step_dict["step_type"] = "unknown"
            
            manager_agent_steps.append(step_dict)
        
        # 提取搜索代理步骤
        for memory_step in self.search_agent.memory.steps:
            memory_step.model_input_messages = None
            step_dict = memory_step.dict()
            truncate_observation(step_dict)
            
            if isinstance(memory_step, ActionStep):
                step_dict["step_type"] = "action"
                step_dict.pop("model_output_message", None)
            elif isinstance(memory_step, TaskStep):
                step_dict["step_type"] = "task"
            elif isinstance(memory_step, PlanningStep):
                step_dict["step_type"] = "planning"
                step_dict.pop("model_output_message_facts", None)
                step_dict.pop("model_output_message_plan", None)
            else:
                step_dict["step_type"] = "unknown"
            
            search_agent_steps.append(step_dict)
        
        return manager_agent_steps, search_agent_steps

    def _parse_suggestions(self, suggestion_text: str) -> tuple[str, str]:
        """解析建议文本，分离管理代理和搜索代理的建议"""
        manager_header = "Manager Agent Suggestions:"
        search_header = "Search Agent Suggestions:"
        
        manager_text = ""
        search_text = ""
        
        # 使用正则表达式解析
        manager_match = re.search(
            f"{re.escape(manager_header)}(.*?)(?={re.escape(search_header)}|$)",
            suggestion_text,
            re.DOTALL,
        )
        if manager_match:
            manager_text = manager_match.group(1).strip()
        
        search_match = re.search(f"{re.escape(search_header)}(.*)", suggestion_text, re.DOTALL)
        if search_match:
            search_text = search_match.group(1).strip()
        
        # 如果没有找到标题但有文本，则作为管理建议
        if not manager_text and not search_text and suggestion_text:
            return suggestion_text.strip(), ""
        
        return manager_text, search_text

    def extract_final_answer(self, result: Any) -> str:
        """从结果中提取最终答案"""
        final_answer: Optional[str] = None
        
        # 基本类型
        if isinstance(result, (str, int, float)):
            final_answer = str(result)
        else:
            # 对象属性
            candidate_attr = (
                getattr(result, "final_answer", None)
                or getattr(result, "answer", None)
                or getattr(result, "output", None)
                or getattr(result, "model_output", None)
            )
            if candidate_attr is not None:
                final_answer = str(candidate_attr)
            else:
                # 可迭代对象
                try:
                    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
                        for step in result:
                            candidate = (
                                getattr(step, "answer", None)
                                or getattr(step, "final_answer", None)
                                or getattr(step, "output", None)
                                or getattr(step, "model_output", None)
                            )
                            if candidate:
                                final_answer = str(candidate)
                except TypeError:
                    pass
        
        # 从执行日志中获取fallback
        if final_answer is None and self.execution_log:
            for step_info in reversed(self.execution_log):
                candidate = step_info.get("model_output") or step_info.get("observations") or step_info.get("error")
                if candidate:
                    final_answer = str(candidate)
                    break
        
        if final_answer is None:
            final_answer = "No answer generated"
        
        # 标准化编码
        with contextlib.suppress(UnicodeDecodeError):
            final_answer = str(final_answer).encode("utf-8").decode("utf-8-sig")
        
        return final_answer

    def _get_file_description(self, file_path: str, question: str) -> str:
        """获取文件描述（复制自原始smolagents实现）"""
        try:
            file_path_lower = file_path.lower()
            
            # 视觉文件
            if any(file_path_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
                visual_tool = VisualInspectorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the image."""
                description = visual_tool.forward(file_path, prompt)
                return f"\n[Image file: {file_path}]\nDescription: {description}\n"
            
            # 音频文件
            elif any(file_path_lower.endswith(ext) for ext in [".mp3", ".m4a", ".wav"]):
                audio_tool = AudioInspectorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this audio. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the audio."""
                description = audio_tool.forward(file_path, prompt)
                return f"\n[Audio file: {file_path}]\nDescription: {description}\n"
            
            # PDF和Office文档
            elif any(file_path_lower.endswith(ext) for ext in [".pdf", ".pptx", ".docx"]):
                markdown_tool = MarkdownConverterTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the document."""
                description = markdown_tool.forward(file_path, prompt)
                return f"\n[Document file: {file_path}]\nDescription: {description}\n"
            
            # CSV文件
            elif file_path_lower.endswith(".csv"):
                csv_tool = CSVExtractorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this CSV file. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the CSV file."""
                description = csv_tool.forward(file_path, prompt)
                return f"\n[CSV file: {file_path}]\nDescription: {description}\n"
            
            # 压缩文件
            elif file_path_lower.endswith(".zip"):
                zip_tool = ZipExtractorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this ZIP archive. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the archive."""
                description = zip_tool.forward(file_path, prompt)
                return f"\n[ZIP archive: {file_path}]\nDescription: {description}\n"
            
            # 文本文件
            elif any(
                file_path_lower.endswith(ext)
                for ext in [".txt", ".md", ".json", ".xml", ".yaml", ".py", ".js", ".html", ".css", ".log"]
            ):
                text_tool = TextExtractorTool(1000)
                prompt = f"""Write a caption of 5 sentences for this text file. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the text file."""
                description = text_tool.forward(file_path, prompt)
                return f"\n[Text file: {file_path}]\nDescription: {description}\n"
            
            # 表格文件
            elif any(file_path_lower.endswith(ext) for ext in [".xlsx", ".xls"]):
                sheet_tool = SheetExtractorTool(self.llm)
                prompt = f"""Write a caption of 5 sentences for this spreadsheet. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the spreadsheet."""
                description = sheet_tool.forward(file_path, prompt)
                return f"\n[Spreadsheet file: {file_path}]\nDescription: {description}\n"
            
            # 默认文本提取
            else:
                text_tool = TextExtractorTool(1000)
                prompt = f"""Write a caption of 5 sentences for this file. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the file."""
                description = text_tool.forward(file_path, prompt)
                return f"\n[File: {file_path}]\nDescription: {description}\n"
                
        except Exception as e:
            return f"\n[File: {file_path}]\n Error processing file: {str(e)}\n"

    def _usage_totals(self) -> tuple[int, int]:
        # Manager and search share one model monitor; count each monitor once.
        monitors = {id(worker.agent.monitor): worker.agent.monitor for worker in self.workers}
        return (sum(m.total_input_token_count for m in monitors.values()),
                sum(m.total_output_token_count for m in monitors.values()))

    def _extract_token_usage_from_agent(self, problem_id: str = "") -> Optional[CompletionUsage]:
        """Recorded input/output tokens for this invocation, including tool calls."""
        try:
            incoming, outgoing = self._usage_totals()
            start_in, start_out = getattr(self, "_usage_start", (0, 0))
            incoming, outgoing = incoming - start_in, outgoing - start_out
            return CompletionUsage(prompt_tokens=incoming, completion_tokens=outgoing,
                                   total_tokens=incoming + outgoing)
        except (AttributeError, TypeError):
            return None

    def _extract_conversation_history(self, query: str, final_answer: Any, problem_id: str = "") -> List[Dict[str, Any]]:
        """构建对话历史"""
        conversation = [
            {
                "role": "user",
                "content": query,
                "name": "user",
                "message_type": "user_query",
                "usage_metadata": None,
            }
        ]
        
        for step_info in self.execution_log:
            content = step_info.get("model_output") or step_info.get("observations") or step_info.get("error") or ""
            
            conversation.append(
                {
                    "role": "assistant",
                    "content": f"[{step_info['step_type']}] {content}",
                    "name": f"{step_info.get('agent_name', 'bench_agent_steps')}",
                    "message_type": "execution_step",
                    "usage_metadata": None,
                }
            )
        
        final_answer_str = str(final_answer) if final_answer is not None else "No answer generated"
        usage_metadata = self._extract_token_usage_from_agent(problem_id=problem_id)
        
        conversation.append(
            {
                "role": "assistant",
                "content": final_answer_str,
                "name": "bench_agent_final",
                "message_type": "final_answer",
                "usage_metadata": usage_metadata,
            }
        )
        return conversation

    def get_agent_info(self) -> Dict[str, Any]:
        """获取代理信息"""
        base_info = super().get_agent_info()
        base_info.update(
            {
                "model_name": self.config.get("model_name"),
                "manager_tools": [tool.__class__.__name__ for tool in getattr(self, "manager_tools", [])],
                "search_tools": [tool.__class__.__name__ for tool in getattr(self, "search_tools", [])],
                "max_steps": self.config.get("max_steps", 15),
                "search_max_steps": self.config.get("search_max_steps", 10),
                "agent_hierarchy": "hierarchical",
                "agents": [{"name": w.name, "type": w.agent.__class__.__name__} for w in self.workers],
                "memory_type": self.memory_type,
                "bench_agent_available": True,
            }
        )
        return base_info


# 注册BenchAgent到系统
AgentSystemRegistry.register(
    "bench_agent",
    BenchAgent,
    max_steps=15,
    search_max_steps=10,
    verbosity_level=1,
    description="High-performance, easy-to-use benchmark agent with pluggable tools and memory support"
)
