# """
# Smolagents Agent System

# This module implements an agent system that uses the smolagents library
# to solve problems with tool-enabled LLM agents.
# """

# import os
# import time
# import re
# from typing import Dict, Any, Optional, List
# import contextlib
# from dataclasses import dataclass
# from string import Template
# from typing_extensions import override
# from openai.types.completion_usage import CompletionUsage
# import yaml

# from mas_arena.utils.llm_utils import call_model
# from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
# from mas_arena.agents.reformulator import prepare_response, truncate_observation
# # 导入外部工具
# from mas_arena.tools.external_tools import (
#     # 媒体工具
#     AudioInspectorTool, VideoInspectorTool, VisualInspectorTool,
#     # 网络工具
#     BrowserTool, DownloadTool, SearchTool, TextInspectorTool, ArxivTool,
#     SimpleCrawler, CrawlerArchiveSearchTool, CrawlerReadTool,
#     # 文档工具
#     CSVExtractorTool, MarkdownConverterTool, SheetExtractorTool,
#     TextExtractorTool, ZipExtractorTool,
#     # 系统工具
#     TerminalTool,
# )

# from smolagents import (
#     ActionStep,
#     CodeAgent,
#     MultiStepAgent,
#     PlanningStep,
#     TaskStep,
#     ToolCallingAgent,
#     OpenAIServerModel,
#     Tool,
#     FinalAnswerTool,
#     PythonInterpreterTool,
#     WikipediaSearchTool as SmolWikipediaSearchTool,
# )
# from mas_arena.agents.base import logger
# from mas_arena.utils.score import question_scorer
# from mas_arena.utils.llm_utils import RetryWrapper

# import asyncio

# current_dir = os.path.dirname(os.path.abspath(__file__))
# with open(prompts_path, "r", encoding="utf-8") as f:
#     prompts = yaml.load(f, Loader=yaml.FullLoader)
# AUTHORIZED_IMPORTS = [
#     "requests",
#     "zipfile",
#     "os",
#     "pandas",
#     "numpy",
#     "sympy",
#     "json",
#     "bs4",
#     "xml",
#     "pydub",
#     "io",
#     "PIL",
#     "PyPDF2",
#     "pptx",
#     "datetime",
#     "fractions",
#     "csv",
#     "random",
#     "re",
#     "sys",
#     "shutil",
# ]


# @dataclass
# class SmolagentsWorker:
#     """Wrapper for smolagents agents (CodeAgent or ToolCallingAgent)"""

#     name: str
#     agent: MultiStepAgent
#     llm: OpenAIServerModel


# class SmolagentsAgent(AgentSystem):
#     """Smolagents Agent System using tool-enabled LLM agents."""

#     def __init__(self, name: str = "smolagents", config: Optional[Dict[str, Any]] = None):
#         """Initialize the Smolagents Agent System"""
#         super().__init__(name, config or {})
#         self.config = config or {}

#         self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
#         # Note: self.format_prompt is set automatically by the base class
#         # based on self.evaluator_name from config

#         self._initialize_model()
#         self._initialize_tools()

#         agent_components = self._create_agents()
#         self.workers = agent_components["workers"]

#         # Find the manager agent (CodeAgent) as the main agent
#         self.agent = next((w.agent for w in self.workers if isinstance(w.agent, CodeAgent)), None)
#         # Find the search agent (ToolCallingAgent) for reference
#         self.search_agent = next(
#             (w.agent for w in self.workers if isinstance(w.agent, ToolCallingAgent)),
#             None,
#         )

#         if not self.agent:
#             raise ValueError("Could not find CodeAgent (manager) in created workers")
#         if not self.search_agent:
#             raise ValueError("Could not find ToolCallingAgent (search) in created workers")

#         self.conversation_history = []
#         self.execution_log = []

#     async def aclose(self):
#         """Close any open connections."""
#         if hasattr(self.llm, "aclose"):
#             await self.llm.aclose()

#     def _create_agents(self) -> Dict[str, List[SmolagentsWorker]]:
#         """Create hierarchical agent system with specialized search agent and manager agent."""

#         # Create search agent with search-specific tools
#         search_planning_interval_str = os.environ.get("SEARCH_AGENT_PLANNING_INTERVAL")
#         search_planning_interval = (
#             int(search_planning_interval_str)
#             if search_planning_interval_str and search_planning_interval_str.isdigit()
#             else None
#         )
#         search_agent = ToolCallingAgent(
#             tools=self.search_tools,
#             model=self.llm,
#             max_steps=self.config.get("search_max_steps", 10),
#             verbosity_level=self.config.get("verbosity_level", 1),
#             # step_callbacks=[self._on_step],
#             provide_run_summary=True,
#             planning_interval=search_planning_interval,
#         )
#         # Monkey-patch for smolagents version mismatch
#         if not hasattr(search_agent, "process_single_tool_call"):
#             def process_single_tool_call(tool_call):
#                 # Encapsulate the single tool call in a list to match process_tool_calls
#                 return search_agent.process_tool_calls([tool_call])
            
#             search_agent.process_single_tool_call = process_single_tool_call

#         search_agent.name = "search_agent"
#         search_agent.description = """Specialized web search agent. Handles all web browsing and online information gathering.
# Use full sentences for requests, provide context including timeframes when needed."""

#         # Enhanced prompt template for search agent
#         search_agent.prompt_templates["system_prompt"] += """
# Your response MUST be a JSON object with the following structure, and nothing else:
# {
#   "tool": "<name_of_the_tool>",
#   "arguments": {
#     "<parameter_name>": "<parameter_value>",
#     ...
#   }
# }"""
#         search_agent.prompt_templates["managed_agent"]["task"] += prompts["search_agent_prompt_managed_task"]

#         # Create enhanced manager agent prompt that explicitly encourages using the search agent

#         manager_system_prompt = prompts["base_manager_prompt"]
#         if self.config.get("additional_instructions"):
#             manager_system_prompt += "\n\n" + self.config["additional_instructions"]

#         manage_planning_interval_str = os.environ.get("MANAGE_AGENT_PLANNING_INTERVAL")
#         manage_planning_interval = (
#             int(manage_planning_interval_str)
#             if manage_planning_interval_str and manage_planning_interval_str.isdigit()
#             else None
#         )
#         manager_instructions = (
#             "You are a manager agent responsible for solving a given task. "
#             "You can use the available tools and delegate tasks to a search agent if needed. "
#             "When you receive a 'search_suggestion' in your arguments, you should consider it carefully. "
#             "If the suggestion is relevant to the search task, you must pass this suggestion to the search_agent via its 'additional_args' parameter."
#         )
#         manager_agent = CodeAgent(
#             tools=self.manager_tools,
#             model=self.llm,
#             max_steps=self.config.get("max_steps", 10),
#             verbosity_level=self.config.get("verbosity_level", 1),
#             # step_callbacks=[self._on_step],
#             stream_outputs=False,
#             additional_authorized_imports=AUTHORIZED_IMPORTS,
#             managed_agents=[search_agent],
#             planning_interval=manage_planning_interval,
#             instructions=manager_instructions,
#         )
#         manager_agent.name = "manager_agent"
#         manager_agent.description = """Manager agent for computational tasks and information synthesis.
# Handles complex multi-step problems, writes Python code, and processes data."""
#         manager_agent.prompt_templates["system_prompt"] += manager_system_prompt

#         # Create workers
#         search_worker = SmolagentsWorker(name="search_agent", agent=search_agent, llm=self.llm)
#         manager_worker = SmolagentsWorker(name="manager_agent", agent=manager_agent, llm=self.llm)

#         return {"workers": [search_worker, manager_worker]}

#     def _initialize_model(self):
#         """Initialize the OpenAI language model for smolagents"""
#         api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
#         api_base = self.config.get("api_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

#         if not api_key:
#             raise ValueError(
#                 "OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config."
#             )

#         llm_model = OpenAIServerModel(
#             model_id=self.model_name,
#             api_base=api_base,
#             api_key=api_key,
#             timeout=int(os.getenv("OPENAI_API_TIMEOUT", "300")),
#         )
#         self.llm = OpenAIServerModel(
#             model_id=self.model_name,
#             custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
#             max_completion_tokens=8192,
#             api_base=api_base,
#             api_key=api_key,
#         )

#         # Wrap the LLM model with our retry logic
#         self.llm = RetryWrapper(llm_model, max_retries=3)

#     def _initialize_tools(self):
#         """Initialize tools for the agent hierarchy"""
#         crawler = SimpleCrawler()

#         search_web_tools = [SearchTool(reflection=False)]

#         class CustomWikipediaSearchTool(SmolWikipediaSearchTool):
#             def forward(self, query: str) -> str:
#                 # The custom session logic caused an AttributeError with the current
#                 # version of wikipedia-api, so we revert to the parent's implementation.
#                 return super().forward(query)

#         crawler_tools = [
#             CrawlerReadTool(crawler),
#             CrawlerArchiveSearchTool(crawler),
#             CustomWikipediaSearchTool(),
#         ]

#         search_browser_tools = [
#             BrowserTool(model=self.llm, text_limit=1000),
#             TextInspectorTool(self.llm, text_limit=100000),
#         ]

#         self.search_tools: List[Tool] = [
#             *search_web_tools,
#             *crawler_tools,
#             *search_browser_tools,
#             ArxivTool(model=self.llm, text_limit=1000),
#             FinalAnswerTool(),
#             MarkdownConverterTool(model=self.llm, text_limit=1000),
#         ]

#         # Manager tools for computational and analytical tasks
#         web_tools = [
#             DownloadTool(model=self.llm, workspace="./download_files"),
#         ]

#         media_tools = [
#             AudioInspectorTool(model=self.llm, text_limit=1000),
#             VideoInspectorTool(model=self.llm, text_limit=1000),
#             VisualInspectorTool(model=self.llm, text_limit=1000),
#         ]

#         system_tools = [
#             FinalAnswerTool(),
#             PythonInterpreterTool(),
#             TerminalTool(model=self.llm, text_limit=1000),
#         ]

#         document_tools = [
#             CSVExtractorTool(model=self.llm, text_limit=1000),
#             MarkdownConverterTool(model=self.llm, text_limit=1000),
#             SheetExtractorTool(model=self.llm),
#             TextExtractorTool(text_limit=1000),
#             ZipExtractorTool(model=self.llm, text_limit=1000),
#         ]

#         self.manager_tools: List[Tool] = []
#         self.manager_tools: List[Tool] = [
#             *web_tools,
#             *media_tools,
#             *system_tools,
#             *document_tools,
#         ]

#         # Combined tools for backward compatibility
#         self.tools: List[Tool] = self.search_tools + self.manager_tools

#     def _on_step(self, memory_step, agent):
#         """Step callback to log each step during execution."""
#         step_data = {
#             "step_number": getattr(memory_step, "step_number", None),
#             "step_type": memory_step.__class__.__name__,
#             "model_output": getattr(memory_step, "model_output", None),
#             "observations": getattr(memory_step, "observations", None),
#             "tool_calls": getattr(memory_step, "tool_calls", None),
#             "error": getattr(memory_step, "error", None),
#             "timestamp": time.time(),
#             "raw": memory_step.__dict__,
#             "agent_name": getattr(agent, "name", "smolagents_agent"),
#             "agent_description": getattr(agent, "description", None),
#             "max_steps": getattr(agent, "max_steps", None),
#             "is_final_answer": getattr(memory_step, "is_final_answer", False),
#         }
#         self.execution_log.append(step_data)

#     def _extract_token_usage_from_agent(self) -> Optional[CompletionUsage]:
#         """Estimate token usage."""
#         try:
#             prompt_tokens = 0
#             completion_tokens = 0
#             for worker in self.workers:
#                 monitor = worker.agent.monitor
#                 prompt_tokens += monitor.total_input_token_count
#                 completion_tokens += monitor.total_output_token_count

#             logger.info(
#                 f"Token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {prompt_tokens + completion_tokens}"
#             )
#             return CompletionUsage(
#                 prompt_tokens=prompt_tokens,
#                 completion_tokens=completion_tokens,
#                 total_tokens=prompt_tokens + completion_tokens,
#             )
#         except Exception:
#             return None

#     def _extract_conversation_history(self, query: str, final_answer: Any) -> List[Dict[str, Any]]:
#         """Build conversation history from execution log."""
#         conversation = [
#             {
#                 "role": "user",
#                 "content": query,
#                 "name": "user",
#                 "message_type": "user_query",
#                 "usage_metadata": None,
#             }
#         ]

#         for step_info in self.execution_log:
#             content = step_info.get("model_output") or step_info.get("observations") or step_info.get("error") or ""

#             conversation.append(
#                 {
#                     "role": "assistant",
#                     "content": f"[{step_info['step_type']}] {content}",
#                     "name": f"{step_info.get('agent_name', 'smolagents_steps')}",
#                     "message_type": "execution_step",
#                     "usage_metadata": None,
#                 }
#             )

#         final_answer_str = str(final_answer) if final_answer is not None else "No answer generated"

#         usage_metadata = self._extract_token_usage_from_agent()

#         conversation.append(
#             {
#                 "role": "assistant",
#                 "content": final_answer_str,
#                 "name": "smolagents_final",
#                 "message_type": "final_answer",
#                 "usage_metadata": usage_metadata,
#             }
#         )
#         return conversation

#     def extract_final_answer(self, result: Any) -> str:
#         """Extract a final answer string from a smolagents run result.

#         This centralizes the logic for pulling an answer from different possible
#         result shapes (primitive, object with attributes, iterable of steps, or
#         falling back to the execution_log).
#         """
#         final_answer: Optional[str] = None

#         # Primitive types
#         if isinstance(result, (str, int, float)):
#             final_answer = str(result)
#         else:
#             # Objects with common answer attributes
#             candidate_attr = (
#                 getattr(result, "final_answer", None)
#                 or getattr(result, "answer", None)
#                 or getattr(result, "output", None)
#                 or getattr(result, "model_output", None)
#             )
#             if candidate_attr is not None:
#                 final_answer = str(candidate_attr)
#             else:
#                 # Iterables of steps
#                 try:
#                     if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
#                         for step in result:  # type: ignore
#                             candidate = (
#                                 getattr(step, "answer", None)
#                                 or getattr(step, "final_answer", None)
#                                 or getattr(step, "output", None)
#                                 or getattr(step, "model_output", None)
#                             )
#                             if candidate:
#                                 final_answer = str(candidate)
#                 except TypeError:
#                     pass

#         # Fallback to last non-empty entry in execution log
#         if final_answer is None and self.execution_log:
#             for step_info in reversed(self.execution_log):
#                 candidate = step_info.get("model_output") or step_info.get("observations") or step_info.get("error")
#                 if candidate:
#                     final_answer = str(candidate)
#                     break

#         if final_answer is None:
#             final_answer = "No answer generated"

#         # Normalize potential BOM/encoding issues
#         with contextlib.suppress(UnicodeDecodeError):
#             final_answer = str(final_answer).encode("utf-8").decode("utf-8-sig")

#         return final_answer

#     def _get_file_description(self, file_path: str, question: str) -> str:
#         """Get description of a single file based on its type."""
#         try:
#             file_path_lower = file_path.lower()

#             # Visual files
#             if any(file_path_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
#                 visual_tool = VisualInspectorTool(self.llm, 1000)
#                 prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the image."""
#                 description = visual_tool.forward(file_path, prompt)
#                 return f"\n[Image file: {file_path}]\nDescription: {description}\n"

#             # Audio files
#             elif any(file_path_lower.endswith(ext) for ext in [".mp3", ".m4a", ".wav"]):
#                 audio_tool = AudioInspectorTool(self.llm, 1000)
#                 prompt = f"""Write a caption of 5 sentences for this audio. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the audio."""
#                 description = audio_tool.forward(file_path, prompt)
#                 return f"\n[Audio file: {file_path}]\nDescription: {description}\n"

#             # PDF and Office documents (convert to markdown first)
#             elif any(file_path_lower.endswith(ext) for ext in [".pdf", ".pptx", ".docx"]):
#                 markdown_tool = MarkdownConverterTool(self.llm, 1000)
#                 prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the document."""
#                 description = markdown_tool.forward(file_path, prompt)
#                 return f"\n[Document file: {file_path}]\nDescription: {description}\n"

#             # CSV files (handle before general text files)
#             elif file_path_lower.endswith(".csv"):
#                 csv_tool = CSVExtractorTool(self.llm, 1000)
#                 prompt = f"""Write a caption of 5 sentences for this CSV file. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the CSV file."""
#                 description = csv_tool.forward(file_path, prompt)
#                 return f"\n[CSV file: {file_path}]\nDescription: {description}\n"

#             # Archive files
#             elif file_path_lower.endswith(".zip"):
#                 zip_tool = ZipExtractorTool(self.llm, 1000)
#                 prompt = f"""Write a caption of 5 sentences for this ZIP archive. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the archive."""
#                 description = zip_tool.forward(file_path, prompt)
#                 return f"\n[ZIP archive: {file_path}]\nDescription: {description}\n"

#             # Text/Document files
#             elif any(
#                 file_path_lower.endswith(ext)
#                 for ext in [
#                     ".txt",
#                     ".md",
#                     ".json",
#                     ".xml",
#                     ".yaml",
#                     ".py",
#                     ".js",
#                     ".html",
#                     ".css",
#                     ".log",
#                 ]
#             ):
#                 text_tool = TextExtractorTool(1000)
#                 prompt = f"""Write a caption of 5 sentences for this text file. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the text file."""
#                 description = text_tool.forward(file_path, prompt)
#                 return f"\n[Text file: {file_path}]\nDescription: {description}\n"

#             # Spreadsheet files
#             elif any(file_path_lower.endswith(ext) for ext in [".xlsx", ".xls"]):
#                 sheet_tool = SheetExtractorTool(self.llm)
#                 prompt = f"""Write a caption of 5 sentences for this spreadsheet. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the spreadsheet."""
#                 description = sheet_tool.forward(file_path, prompt)
#                 return f"\n[Spreadsheet file: {file_path}]\nDescription: {description}\n"

#             # Default to text extraction for unknown types
#             else:
#                 text_tool = TextExtractorTool(1000)
#                 prompt = f"""Write a caption of 5 sentences for this file. Pay special attention to any details that might be useful for someone answering the following question:
# {question}. But do not try to answer the question directly!
# Do not add any information that is not present in the file."""
#                 description = text_tool.forward(file_path, prompt)
#                 return f"\n[File: {file_path}]\nDescription: {description}\n"

#         except Exception as e:
#             return f"\n[File: {file_path}]\n Error processing file: {str(e)}\n"

#     def _parse_suggestions(self, suggestion_text: str) -> tuple[str, str]:
#         """Parses the suggestion text to separate suggestions for manager and search agents."""
#         manager_header = "Manager Agent Suggestions:"
#         search_header = "Search Agent Suggestions:"

#         manager_text = ""
#         search_text = ""

#         # Use regex to be more robust, assuming manager suggestions come before search suggestions
#         manager_match = re.search(
#             f"{re.escape(manager_header)}(.*?)(?={re.escape(search_header)}|$)",
#             suggestion_text,
#             re.DOTALL,
#         )
#         if manager_match:
#             manager_text = manager_match.group(1).strip()

#         search_match = re.search(f"{re.escape(search_header)}(.*)", suggestion_text, re.DOTALL)
#         if search_match:
#             search_text = search_match.group(1).strip()

#         # Fallback if no headers are found but there is text
#         if not manager_text and not search_text and suggestion_text:
#             return suggestion_text.strip(), ""

#         return manager_text, search_text

#     def _run_agent_sync(self, augmented_question: str, additional_args: Dict[str, Any]) -> Any:
#         """Synchronous part of the agent execution."""
#         return self.agent.run(augmented_question, additional_args=additional_args)

#     async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
#         """Run the agent system and capture all steps (non-streaming by default)."""
#         self.execution_log = []
#         search_keywords = ""

#         # Create enhanced augmented question that explicitly guides the agent to use search_agent
#         augmented_question = """Answer this question correctly. You have all the tools needed to find the right answer.

# Use search_agent for any web research or current information.

# Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
# Run verification steps if that's needed, you must make sure you find the correct answer!


# Task:
# """ + problem["problem"]
#         # Handle file attachments if present
#         if "files" in problem and problem["files"]:
#             print("Detected files in problem, preparing file descriptions...")
#             files = problem["files"]
#             if isinstance(files, str):
#                 # Convert single file to list
#                 files = [files]

#             if len(files) == 1:
#                 prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:"
#                 prompt_use_files += self._get_file_description(files[0], problem["problem"])
#             else:
#                 prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:"
#                 for file_path in files:
#                     prompt_use_files += self._get_file_description(file_path, problem["problem"])

#             augmented_question += prompt_use_files

#         if "context" in kwargs:
#             augmented_question += f"\n\nContext: {kwargs['context']}"

#         try:
#             if not self.agent:
#                 raise RuntimeError("Agent not properly initialized")

#             additional_knowledge = ""
#             if self.meta_memory is not None:
#                 template_str = prompts["build_search_keywords_prompt"]
#                 template = Template(template_str)

#                 build_search_keywords_prompt = template.substitute(
#                     question=problem["problem"], true_answer=problem["solution"]
#                 )

#                 search_keywords = await asyncio.to_thread(
#                     call_model, build_search_keywords_prompt, self.model_name
#                 )
#                 try:
#                     successful_trajectories, _, insights = (
#                         await self.meta_memory.retrieve_memory(
#                             task_search_keywords=search_keywords,
#                             task_question=problem["problem"],
#                         )
#                     )
#                     logger.info(
#                         f"the number of successful trajectories: {len(successful_trajectories)}"
#                     )
#                     logger.info(f"the number of successful trajectories: {len(successful_trajectories)}")
#                     logger.info(f"the number of insights: {len(insights)}")

#                     additional_knowledge += "\n\n".join(
#                         [insight for insight in insights]
#                     )
#                 except Exception as e:
#                     additional_knowledge = None
#                     logger.error(f"Error retrieving memory: {str(e)}")
#             additional_args = {"additional_knowledge": additional_knowledge}
#             # result = self.agent.run(augmented_question, additional_args=additional_args)
#             result = await asyncio.to_thread(self._run_agent_sync, augmented_question, additional_args)

#             final_answer = self.extract_final_answer(result)
#             semantic_match_prompt = prompts["semantic_match_prompt"].format(
#                 question=problem["problem"],
#                 prediction=final_answer,
#                 true_answer=problem["solution"],
#             )
#             semantic_check = await asyncio.to_thread(
#                 call_model,
#                 query=semantic_match_prompt, model_name=self.model_name
#             )

#             if (not question_scorer(final_answer, problem["solution"])) or (semantic_check == "false"):
#                 manager_agent_steps = []
#                 search_agent_steps = []

#                 for memory_step in self.agent.memory.steps:
#                     memory_step.model_input_messages = None
#                     step_dict = memory_step.dict()
#                     truncate_observation(step_dict)
#                     if isinstance(memory_step, ActionStep):
#                         step_dict["step_type"] = "action"
#                         step_dict.pop("model_output_message", None)
#                     elif isinstance(memory_step, TaskStep):
#                         step_dict["step_type"] = "task"
#                     elif isinstance(memory_step, PlanningStep):
#                         step_dict["step_type"] = "planning"
#                         step_dict.pop("model_output_message_facts", None)
#                         step_dict.pop("model_output_message_plan", None)
#                     else:
#                         step_dict["step_type"] = "unknown"
#                     manager_agent_steps.append(step_dict)

#                 for memory_step in self.search_agent.memory.steps:
#                     memory_step.model_input_messages = None
#                     step_dict = memory_step.dict()
#                     truncate_observation(step_dict)
#                     if isinstance(memory_step, ActionStep):
#                         step_dict["step_type"] = "action"
#                         step_dict.pop("model_output_message", None)
#                     elif isinstance(memory_step, TaskStep):
#                         step_dict["step_type"] = "task"
#                     elif isinstance(memory_step, PlanningStep):
#                         step_dict["step_type"] = "planning"
#                         step_dict.pop("model_output_message_facts", None)
#                         step_dict.pop("model_output_message_plan", None)
#                     else:
#                         step_dict["step_type"] = "unknown"
#                     search_agent_steps.append(step_dict)

#                 annotated_example = {
#                     "question": problem["problem"],
#                     "prediction": final_answer,
#                     "ground_truth": problem["solution"],
#                     "manager_agent_steps": manager_agent_steps,
#                     "search_agent_steps": search_agent_steps,
#                 }

#                 suggestion_prompt = prompts["failure_attribution_and_suggestion_prompt"].format(
#                     knowledge=additional_knowledge, agent_log=str(annotated_example)
#                 )

#                 suggestion = await asyncio.to_thread(
#                     call_model,
#                     query=suggestion_prompt,
#                     model_name=self.model_name
#                 )
#                 logger.info(f"suggestion by memory: {suggestion}")
#                 manager_suggestion, search_suggestion = self._parse_suggestions(suggestion)
#                 logger.info(f"Parsed manager suggestion: {manager_suggestion}")
#                 logger.info(f"Parsed search suggestion: {search_suggestion}")
#                 additional_args = {}
#                 if manager_suggestion:
#                     additional_args["manager_suggestion"] = manager_suggestion
#                 if search_suggestion:
#                     additional_args["search_suggestion"] = search_suggestion

#                 # Fallback to the raw suggestion if parsing fails
#                 if not additional_args and suggestion:
#                     additional_args["suggestion"] = suggestion

#                 final_result = self.agent.run(augmented_question, additional_args=additional_args)
#                 final_answer = self.extract_final_answer(final_result)

#             conversation_messages = self._extract_conversation_history(augmented_question, final_answer)
#             self.conversation_history.extend(conversation_messages)

#             manager_agent_steps = []
#             search_agent_steps = []

#             for memory_step in self.agent.memory.steps:
#                 memory_step.model_input_messages = None
#                 step_dict = memory_step.dict()
#                 truncate_observation(step_dict)
#                 if isinstance(memory_step, ActionStep):
#                     step_dict["step_type"] = "action"
#                     step_dict.pop("model_output_message", None)
#                 elif isinstance(memory_step, TaskStep):
#                     step_dict["step_type"] = "task"
#                 elif isinstance(memory_step, PlanningStep):
#                     step_dict["step_type"] = "planning"
#                     step_dict.pop("model_output_message_facts", None)
#                     step_dict.pop("model_output_message_plan", None)
#                 else:
#                     step_dict["step_type"] = "unknown"
#                 manager_agent_steps.append(step_dict)

#             for memory_step in self.search_agent.memory.steps:
#                 memory_step.model_input_messages = None
#                 step_dict = memory_step.dict()
#                 truncate_observation(step_dict)
#                 if isinstance(memory_step, ActionStep):
#                     step_dict["step_type"] = "action"
#                     step_dict.pop("model_output_message", None)
#                 elif isinstance(memory_step, TaskStep):
#                     step_dict["step_type"] = "task"
#                 elif isinstance(memory_step, PlanningStep):
#                     step_dict["step_type"] = "planning"
#                     step_dict.pop("model_output_message_facts", None)
#                     step_dict.pop("model_output_message_plan", None)
#                 else:
#                     step_dict["step_type"] = "unknown"
#                 search_agent_steps.append(step_dict)

#             return {
#                 "messages": conversation_messages,
#                 "final_answer": final_answer,
#                 "manager_agent_steps": manager_agent_steps,
#                 "search_agent_steps": search_agent_steps,
#                 "search_keywords": search_keywords,
#             }

#         except Exception as e:
#             error_message = f"Error running smolagents: {str(e)}"
#             error_ai_message = {
#                 "content": error_message,
#                 "name": "smolagents_error",
#                 "role": "assistant",
#                 "message_type": "error_response",
#                 "usage_metadata": None,
#             }
#             return {
#                 "messages": [error_ai_message],
#                 "final_answer": error_message,
#                 "error": str(e),
#             }

#     def get_agent_info(self) -> Dict[str, Any]:
#         base_info = super().get_agent_info()
#         base_info.update(
#             {
#                 "model_name": self.model_name,
#                 "search_tools": [tool.__class__.__name__ for tool in getattr(self, "search_tools", [])],
#                 "manager_tools": [tool.__class__.__name__ for tool in getattr(self, "manager_tools", [])],
#                 "max_steps": self.config.get("max_steps", 10),
#                 "search_max_steps": self.config.get("search_max_steps", 15),
#                 "agent_hierarchy": "hierarchical",
#                 "agents": [{"name": w.name, "type": w.agent.__class__.__name__} for w in self.workers],
#                 "smolagents_available": True,
#             }
#         )
#         return base_info


# AgentSystemRegistry.register(
#     "smolagents",
#     SmolagentsAgent,
#     max_steps=15,
#     search_max_steps=10,
#     verbosity_level=1,
# )
