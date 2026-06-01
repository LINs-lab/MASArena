import os
import yaml
import contextlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

# 导入外部工具
from mas_arena.tools.external_tools import (
    # 媒体工具
    AudioInspectorTool, VideoInspectorTool, VisualInspectorTool,
    # 网络工具
    BrowserTool, DownloadTool, SearchTool, TextInspectorTool, ArxivTool,
    SimpleCrawler, CrawlerArchiveSearchTool, CrawlerReadTool,
    # 文档工具
    CSVExtractorTool, MarkdownConverterTool, SheetExtractorTool,
    TextExtractorTool, ZipExtractorTool,
    # 系统工具
    TerminalTool,
)

from smolagents import (
    CodeAgent,
    MultiStepAgent,
    ToolCallingAgent,
    OpenAIServerModel,
    Tool,
    FinalAnswerTool,
    PythonInterpreterTool,
    WikipediaSearchTool,
)

from .prompts import GAIA_FORMAT_PROMPT
from mas_arena.utils.env import (
    DEFAULT_MODEL_NAME,
    get_env_float,
    get_env_int,
    get_model_name,
    get_openai_api_base,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_path = Path(current_dir).parents[1] / "prompts" / "verl_prompts.yaml"
with open(prompts_path, "r", encoding="utf-8") as f:
    prompts = yaml.load(f, Loader=yaml.FullLoader)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "xml",
    "pydub",
    "io",
    "PIL",
    "PyPDF2",
    "pptx",
    "datetime",
    "fractions",
    "csv",
    "random",
    "re",
    "sys",
    "shutil",
]


@dataclass
class SmolagentsWorker:
    name: str
    agent: MultiStepAgent
    llm: OpenAIServerModel


class SmolagentsAgent:
    """独立版 Smolagents Agent System，不依赖 mas_arena"""

    llm: OpenAIServerModel
    agent: Optional[CodeAgent]
    search_agent: Optional[ToolCallingAgent]
    workers: List[SmolagentsWorker]
    execution_log: list

    def __init__(self, name: str = "smolagents", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.model_name = self.config.get("model_name") or get_model_name(DEFAULT_MODEL_NAME)
        self.execution_log = []
        self.llm = self._initialize_model()
        self._initialize_tools()
        agent_components = self._create_agents()
        self.workers = agent_components["workers"]
        self.agent = next((w.agent for w in self.workers if isinstance(w.agent, CodeAgent)), None)
        self.search_agent = next(
            (w.agent for w in self.workers if isinstance(w.agent, ToolCallingAgent)),
            None,
        )
        if not self.agent:
            raise ValueError("Could not find CodeAgent (manager) in created workers")
        if not self.search_agent:
            raise ValueError("Could not find ToolCallingAgent (search) in created workers")

    def _initialize_model(self) -> OpenAIServerModel:
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        api_base = self.config.get("api_base") or get_openai_api_base()
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config."
            )
        return OpenAIServerModel(
            model_id=self.model_name,
            custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
            max_completion_tokens=8192,
            api_base=api_base,
            api_key=api_key,
        )

    def _initialize_tools(self):
        crawler = SimpleCrawler()
        search_web_tools = [SearchTool(reflection=False)]
        crawler_tools = [
            CrawlerReadTool(crawler),
            CrawlerArchiveSearchTool(crawler),
            WikipediaSearchTool(),
        ]
        search_browser_tools = [
            BrowserTool(model=self.llm, text_limit=1000),
            TextInspectorTool(self.llm, text_limit=100000),
        ]
        self.search_tools: List[Tool] = [
            *search_web_tools,
            *crawler_tools,
            *search_browser_tools,
            ArxivTool(model=self.llm, text_limit=1000),
            FinalAnswerTool(),
            MarkdownConverterTool(model=self.llm, text_limit=1000),
        ]
        web_tools = [DownloadTool(model=self.llm, workspace="./download_files")]
        media_tools = [
            AudioInspectorTool(model=self.llm, text_limit=1000),
            VideoInspectorTool(model=self.llm, text_limit=1000),
            VisualInspectorTool(model=self.llm, text_limit=1000),
        ]
        system_tools = [
            FinalAnswerTool(),
            PythonInterpreterTool(),
            TerminalTool(model=self.llm, text_limit=1000),
        ]
        document_tools = [
            CSVExtractorTool(model=self.llm, text_limit=1000),
            MarkdownConverterTool(model=self.llm, text_limit=1000),
            SheetExtractorTool(model=self.llm),
            TextExtractorTool(text_limit=1000),
            ZipExtractorTool(model=self.llm, text_limit=1000),
        ]
        self.manager_tools: List[Tool] = [
            *web_tools,
            *media_tools,
            *system_tools,
            *document_tools,
        ]
        self.tools: List[Tool] = self.search_tools + self.manager_tools

    def _create_agents(self) -> Dict[str, List[SmolagentsWorker]]:
        search_planning_interval = int(os.environ.get("SEARCH_AGENT_PLANNING_INTERVAL", "0") or 0) or None
        search_agent = ToolCallingAgent(
            tools=self.search_tools,
            model=self.llm,
            max_steps=self.config.get("search_max_steps", 15),
            verbosity_level=self.config.get("verbosity_level", 1),
            provide_run_summary=True,
            planning_interval=search_planning_interval,
        )
        search_agent.name = "search_agent"
        search_agent.description = (
            """Specialized web search agent. Handles all web browsing and online information gathering."""
        )
        if prompts and "search_agent_prompt_managed_task" in prompts:
            search_agent.prompt_templates["managed_agent"]["task"] += prompts["search_agent_prompt_managed_task"]
        manager_system_prompt = prompts.get("base_manager_prompt", "")
        if self.config.get("additional_instructions"):
            manager_system_prompt += "\n\n" + self.config["additional_instructions"]
        manage_planning_interval = int(os.environ.get("MANAGE_AGENT_PLANNING_INTERVAL", "0") or 0) or None
        manager_agent = CodeAgent(
            tools=self.manager_tools,
            model=self.llm,
            max_steps=self.config.get("max_steps", 10),
            verbosity_level=self.config.get("verbosity_level", 1),
            stream_outputs=False,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            managed_agents=[search_agent],
            planning_interval=manage_planning_interval,
        )
        manager_agent.name = "manager_agent"
        manager_agent.description = "Manager agent for computational tasks and information synthesis. Handles complex multi-step problems, writes Python code, and processes data."
        if "system_prompt" in manager_agent.prompt_templates:
            manager_agent.prompt_templates["system_prompt"] += manager_system_prompt
        search_worker = SmolagentsWorker(name="search_agent", agent=search_agent, llm=self.llm)
        manager_worker = SmolagentsWorker(name="manager_agent", agent=manager_agent, llm=self.llm)
        return {"workers": [search_worker, manager_worker]}

    def _get_file_description(self, file_path: str, question: str) -> str:
        try:
            file_path_lower = file_path.lower()
            if any(file_path_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
                visual_tool = VisualInspectorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the image."""
                description = visual_tool.forward(file_path, prompt)
                return f"\n[Image file: {file_path}]\nDescription: {description}\n"
            elif any(file_path_lower.endswith(ext) for ext in [".mp3", ".m4a", ".wav"]):
                audio_tool = AudioInspectorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this audio. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the audio."""
                description = audio_tool.forward(file_path, prompt)
                return f"\n[Audio file: {file_path}]\nDescription: {description}\n"
            elif any(file_path_lower.endswith(ext) for ext in [".pdf", ".pptx", ".docx"]):
                markdown_tool = MarkdownConverterTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the document."""
                description = markdown_tool.forward(file_path, prompt)
                return f"\n[Document file: {file_path}]\nDescription: {description}\n"
            elif file_path_lower.endswith(".csv"):
                csv_tool = CSVExtractorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this CSV file. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the CSV file."""
                description = csv_tool.forward(file_path, prompt)
                return f"\n[CSV file: {file_path}]\nDescription: {description}\n"
            elif file_path_lower.endswith(".zip"):
                zip_tool = ZipExtractorTool(self.llm, 1000)
                prompt = f"""Write a caption of 5 sentences for this ZIP archive. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the archive."""
                description = zip_tool.forward(file_path, prompt)
                return f"\n[ZIP archive: {file_path}]\nDescription: {description}\n"
            elif any(
                file_path_lower.endswith(ext)
                for ext in [
                    ".txt",
                    ".md",
                    ".json",
                    ".xml",
                    ".yaml",
                    ".py",
                    ".js",
                    ".html",
                    ".css",
                    ".log",
                ]
            ):
                text_tool = TextExtractorTool(1000)
                prompt = f"""Write a caption of 5 sentences for this text file. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the text file."""
                description = text_tool.forward(file_path, prompt)
                return f"\n[Text file: {file_path}]\nDescription: {description}\n"
            elif any(file_path_lower.endswith(ext) for ext in [".xlsx", ".xls"]):
                sheet_tool = SheetExtractorTool(self.llm)
                prompt = f"""Write a caption of 5 sentences for this spreadsheet. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the spreadsheet."""
                description = sheet_tool.forward(file_path, prompt)
                return f"\n[Spreadsheet file: {file_path}]\nDescription: {description}\n"
            else:
                text_tool = TextExtractorTool(1000)
                prompt = f"""Write a caption of 5 sentences for this file. Pay special attention to any details that might be useful for someone answering the following question:\n{question}. But do not try to answer the question directly!\nDo not add any information that is not present in the file."""
                description = text_tool.forward(file_path, prompt)
                return f"\n[File: {file_path}]\nDescription: {description}\n"
        except Exception as e:
            return f"\n[File: {file_path}]\n Error processing file: {str(e)}\n"

    def extract_final_answer(self, result: Any) -> str:
        final_answer: Optional[str] = None
        if isinstance(result, (str, int, float)):
            final_answer = str(result)
        else:
            candidate_attr = (
                getattr(result, "final_answer", None)
                or getattr(result, "answer", None)
                or getattr(result, "output", None)
                or getattr(result, "model_output", None)
            )
            if candidate_attr is not None:
                final_answer = str(candidate_attr)
            else:
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
        if final_answer is None:
            final_answer = "No answer generated"
        with contextlib.suppress(UnicodeDecodeError):
            final_answer = str(final_answer).encode("utf-8").decode("utf-8-sig")
        return final_answer

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        import logging
        from mas_arena.utils.llm_utils import call_model
        from mas_arena.utils.score import question_scorer
        from smolagents import ActionStep, TaskStep, PlanningStep

        self.execution_log = []
        augmented_question = (
            GAIA_FORMAT_PROMPT
            + """Answer this question correctly. You have all the tools needed to find the right answer.\n\nUse search_agent for any web research or current information.\n\nFailure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.\nRun verification steps if that's needed, you must make sure you find the correct answer!\n\n\nTask:\n"""
            + problem["problem"]
        )
        if "files" in problem and problem["files"]:
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
        try:
            if not self.agent:
                raise RuntimeError("Agent not properly initialized")

            additional_knowledge = None
            if getattr(self, "meta_memory", None) is not None:
                build_search_keywords_prompt = prompts["build_search_keywords_prompt"].format(
                    user_query=problem["problem"]
                )
                search_keywords = call_model(
                    build_search_keywords_prompt,
                    "gpt-4.1"
                )
                try:
                    successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
                        task_search_keywords=search_keywords,
                        task_question=problem["problem"],
                        successful_topk=get_env_int("MAS_RL_SUCCESSFUL_TOPK", 4),
                        failed_topk=get_env_int("MAS_RL_FAILED_TOPK", 2),
                        insight_topk=get_env_int("MAS_RL_INSIGHTS_TOPK", 3),
                        threshold=get_env_float("MAS_RL_SIMILARITY_THRESHOLD", 0.3),
                    )
                    logging.info(f"the number of successful trajectories: {len(successful_trajectories)}")
                    logging.info(f"the number of insights: {len(insights)}")
                    additional_knowledge = "\n\n".join(
                        [
                            trajectory.task_question + "\n" + trajectory.task_trajectory
                            for trajectory in successful_trajectories
                        ]
                    )
                    additional_knowledge += "\n\n".join([insight for insight in insights])
                except Exception as e:
                    additional_knowledge = None
                    logging.error(f"Error retrieving memory: {str(e)}")
            else:
                search_keywords = None
            additional_args = {"additional_knowledge": additional_knowledge}
            result = self.agent.run(augmented_question, additional_args=additional_args)

            final_answer = self.extract_final_answer(result)
            semantic_match_prompt = prompts["semantic_match_prompt"].format(
                question=problem["problem"],
                prediction=final_answer,
                true_answer=problem.get("solution", ""),
            )
            semantic_check = call_model(
                query=semantic_match_prompt,
                model_name="gpt-4.1"
            )

            if (not question_scorer(final_answer, problem.get("solution", ""))) and (semantic_check == "false"):
                intermediate_steps = []
                log_plan = None
                for memory_step in getattr(self.agent, "memory", None).steps:
                    memory_step.model_input_messages = None
                    step_dict = memory_step.dict()
                    if isinstance(memory_step, ActionStep):
                        step_dict["step_type"] = "action"
                        step_dict.pop("model_output_message", None)
                    elif isinstance(memory_step, TaskStep):
                        step_dict["step_type"] = "task"
                    elif isinstance(memory_step, PlanningStep):
                        log_plan = step_dict.get("plan", None)
                        step_dict["step_type"] = "planning"
                        step_dict.pop("model_output_message_facts", None)
                        step_dict.pop("model_output_message_plan", None)
                    else:
                        step_dict["step_type"] = "unknown"
                    intermediate_steps.append(step_dict)

                annotated_example = {
                    "question": problem["problem"],
                    "prediction": final_answer,
                    "intermediate_steps": intermediate_steps,
                }

                suggestion_prompt = prompts["failure_attribution_and_suggestion_prompt"].format(
                    knowledge=additional_knowledge, agent_log=str(annotated_example)
                )

                suggestion = call_model(
                    query=suggestion_prompt,
                    model_name="gpt-4.1"
                )
                logging.info(f"suggestion by memory: {suggestion}")
                additional_args = {"suggestion": suggestion}

                final_result = self.agent.run(augmented_question, additional_args=additional_args)
                final_answer = self.extract_final_answer(final_result)

            conversation_messages = [{"role": "assistant", "content": final_answer}]
            intermediate_steps = []
            log_plan = None
            if getattr(self.agent, "memory", None) is not None:
                for memory_step in self.agent.memory.steps:
                    memory_step.model_input_messages = None
                    step_dict = memory_step.dict()
                    if isinstance(memory_step, ActionStep):
                        step_dict["step_type"] = "action"
                        step_dict.pop("model_output_message", None)
                    elif isinstance(memory_step, TaskStep):
                        step_dict["step_type"] = "task"
                    elif isinstance(memory_step, PlanningStep):
                        log_plan = step_dict.get("plan", None)
                        step_dict["step_type"] = "planning"
                        step_dict.pop("model_output_message_facts", None)
                        step_dict.pop("model_output_message_plan", None)
                    else:
                        step_dict["step_type"] = "unknown"
                    intermediate_steps.append(step_dict)

            return {
                "messages": conversation_messages,
                "final_answer": final_answer,
                "intermediate_steps": intermediate_steps,
                "log_plan": log_plan,
                "search_keywords": search_keywords,
            }
        except Exception as e:
            error_message = f"Error running smolagents: {str(e)}"
            return {
                "messages": [{"role": "assistant", "content": error_message}],
                "final_answer": error_message,
                "error": str(e),
            }
