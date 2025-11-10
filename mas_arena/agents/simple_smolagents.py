"""
Smolagents Agent System

This module implements an agent system that uses the smolagents library
to solve problems with tool-enabled LLM agents.
"""

import os
import time
import re
from typing import Dict, Any, Optional, List
import contextlib
from dataclasses import dataclass
from string import Template
from openai.types.completion_usage import CompletionUsage
import yaml

from mas_arena.utils.llm_utils import call_model
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.smolagents_tools.system.terminal_tool import TerminalTool
from smolagents import (
    ActionStep,
    CodeAgent,
    MultiStepAgent,
    PlanningStep,
    TaskStep,
    OpenAIServerModel,
    Tool,
    FinalAnswerTool,
    PythonInterpreterTool,
)
from mas_arena.agents.base import logger
from dotenv import load_dotenv
from mas_arena.utils.score import question_scorer
from mas_arena.utils.llm_utils import RetryWrapper

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
prompts_path = os.path.join(current_dir, "simple_smolagents_prompts.yaml")
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
class SimpleSmolagentsWorker:
    """Wrapper for smolagents agents (CodeAgent or ToolCallingAgent)"""

    name: str
    agent: MultiStepAgent
    llm: OpenAIServerModel


class SimpleSmolagentsAgent(AgentSystem):
    """Smolagents Agent System using tool-enabled LLM agents."""

    def __init__(self, name: str = "smolagents", config: Optional[Dict[str, Any]] = None):
        """Initialize the Smolagents Agent System"""
        super().__init__(name, config or {})
        self.config = config or {}

        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        # Note: self.format_prompt is set automatically by the base class
        # based on self.evaluator_name from config

        self._initialize_model()
        self._initialize_tools()

        agent_components = self._create_agents()
        self.workers: List[SimpleSmolagentsWorker] = agent_components["workers"]

        # Find the manager agent (CodeAgent) as the main agent
        self.agent = next((w.agent for w in self.workers if isinstance(w.agent, CodeAgent)), None)

        if not self.agent:
            raise ValueError("Could not find CodeAgent (manager) in created workers")

        self.conversation_history = []
        self.execution_log = []

    async def aclose(self):
        """Close any open connections."""
        if hasattr(self.llm, "aclose"):
            await self.llm.aclose()

    def _create_agents(self) -> Dict[str, List[SimpleSmolagentsWorker]]:
        """Create agent system with manager agent."""

        # Create enhanced manager agent prompt that explicitly encourages using the search agent

        manager_system_prompt = prompts["base_manager_prompt"]
        if self.config.get("additional_instructions"):
            manager_system_prompt += "\n\n" + self.config["additional_instructions"]

        manage_planning_interval_str = os.environ.get("MANAGE_AGENT_PLANNING_INTERVAL")
        manage_planning_interval = (
            int(manage_planning_interval_str)
            if manage_planning_interval_str and manage_planning_interval_str.isdigit()
            else None
        )
        manager_instructions = (
            "You are a manager agent responsible for solving a given task. "
            "You can use the available tools and delegate tasks to a search agent if needed. "
        )
        manager_agent = CodeAgent(
            tools=self.manager_tools,
            model=self.llm,
            max_steps=self.config.get("max_steps", 10),
            verbosity_level=self.config.get("verbosity_level", 1),
            stream_outputs=False,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            planning_interval=manage_planning_interval,
            instructions=manager_instructions,
        )
        manager_agent.name = "manager_agent"
        manager_agent.description = """Manager agent for computational tasks and information synthesis.
Handles complex multi-step problems, writes Python code, and processes data."""
        manager_agent.prompt_templates["system_prompt"] += manager_system_prompt
        # Create workers
        manager_worker = SimpleSmolagentsWorker(name="manager_agent", agent=manager_agent, llm=self.llm)

        return {"workers": [manager_worker]}

    def _initialize_model(self):
        """Initialize the OpenAI language model for smolagents"""
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        api_base = self.config.get("api_base") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config."
            )

        llm_model = OpenAIServerModel(
            model_id=self.model_name,
            api_base=api_base,
            api_key=api_key,
            timeout=int(os.getenv("OPENAI_API_TIMEOUT", "300")),
        )
        self.llm = OpenAIServerModel(
            model_id=self.model_name,
            custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
            max_completion_tokens=8192,
            api_base=api_base,
            api_key=api_key,
        )

        # Wrap the LLM model with our retry logic
        self.llm = RetryWrapper(llm_model, max_retries=3)

    def _initialize_tools(self):
        """Initialize tools for the agent hierarchy"""
        # Manager tools for computational and analytical tasks
        system_tools = [
            FinalAnswerTool(),
            PythonInterpreterTool(),
            TerminalTool(model=self.llm, text_limit=1000),
        ]

        self.manager_tools: List[Tool] = system_tools

        # Combined tools for backward compatibility
        self.tools: List[Tool] = self.manager_tools

    def _on_step(self, memory_step, agent):
        """Step callback to log each step during execution."""
        step_data = {
            "step_number": getattr(memory_step, "step_number", None),
            "step_type": memory_step.__class__.__name__,
            "model_output": getattr(memory_step, "model_output", None),
            "observations": getattr(memory_step, "observations", None),
            "tool_calls": getattr(memory_step, "tool_calls", None),
            "error": getattr(memory_step, "error", None),
            "timestamp": time.time(),
            "raw": memory_step.__dict__,
            "agent_name": getattr(agent, "name", "smolagents_agent"),
            "agent_description": getattr(agent, "description", None),
            "max_steps": getattr(agent, "max_steps", None),
            "is_final_answer": getattr(memory_step, "is_final_answer", False),
        }
        self.execution_log.append(step_data)

    def _extract_token_usage_from_agent(self) -> Optional[CompletionUsage]:
        """Estimate token usage."""
        try:
            prompt_tokens = 0
            completion_tokens = 0
            for worker in self.workers:
                monitor = worker.agent.monitor
                prompt_tokens += monitor.total_input_token_count
                completion_tokens += monitor.total_output_token_count

            logger.info(
                f"Token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {prompt_tokens + completion_tokens}"
            )
            return CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        except Exception:
            return None

    def _extract_conversation_history(self, query: str, final_answer: Any) -> List[Dict[str, Any]]:
        """Build conversation history from execution log."""
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
                    "name": f"{step_info.get('agent_name', 'smolagents_steps')}",
                    "message_type": "execution_step",
                    "usage_metadata": None,
                }
            )

        final_answer_str = str(final_answer) if final_answer is not None else "No answer generated"

        usage_metadata = self._extract_token_usage_from_agent()

        conversation.append(
            {
                "role": "assistant",
                "content": final_answer_str,
                "name": "smolagents_final",
                "message_type": "final_answer",
                "usage_metadata": usage_metadata,
            }
        )
        return conversation

    def extract_final_answer(self, result: Any) -> str:
        """Extract a final answer string from a smolagents run result.

        This centralizes the logic for pulling an answer from different possible
        result shapes (primitive, object with attributes, iterable of steps, or
        falling back to the execution_log).
        """
        final_answer: Optional[str] = None

        # Primitive types
        if isinstance(result, (str, int, float)):
            final_answer = str(result)
        else:
            # Objects with common answer attributes
            candidate_attr = (
                getattr(result, "final_answer", None)
                or getattr(result, "answer", None)
                or getattr(result, "output", None)
                or getattr(result, "model_output", None)
            )
            if candidate_attr is not None:
                final_answer = str(candidate_attr)
            else:
                # Iterables of steps
                try:
                    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
                        for step in result:  # type: ignore
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

        # Fallback to last non-empty entry in execution log
        if final_answer is None and self.execution_log:
            for step_info in reversed(self.execution_log):
                candidate = step_info.get("model_output") or step_info.get("observations") or step_info.get("error")
                if candidate:
                    final_answer = str(candidate)
                    break

        if final_answer is None:
            final_answer = "No answer generated"

        # Normalize potential BOM/encoding issues
        with contextlib.suppress(UnicodeDecodeError):
            final_answer = str(final_answer).encode("utf-8").decode("utf-8-sig")

        return final_answer

    def _parse_suggestions(self, suggestion_text: str) -> tuple[str, str]:
        """Parses the suggestion text to separate suggestions for manager and search agents."""
        manager_header = "Manager Agent Suggestions:"
        search_header = "Search Agent Suggestions:"

        manager_text = ""
        search_text = ""

        # Use regex to be more robust, assuming manager suggestions come before search suggestions
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

        # Fallback if no headers are found but there is text
        if not manager_text and not search_text and suggestion_text:
            return suggestion_text.strip(), ""

        return manager_text, search_text

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the agent system and capture all steps (non-streaming by default)."""
        self.execution_log = []
        search_keywords = ""

        augmented_question = """Answer this question correctly. You have all the tools needed to find the right answer.

Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer!


Task:
""" + problem["problem"]

        if "context" in kwargs:
            augmented_question += f"\n\nContext: {kwargs['context']}"

        try:
            if not self.agent:
                raise RuntimeError("Agent not properly initialized")

            additional_knowledge = ""
            if self.meta_memory is not None:
                template_str = prompts["build_search_keywords_prompt"]
                template = Template(template_str)

                build_search_keywords_prompt = template.substitute(
                    question=problem["problem"], true_answer=problem["solution"]
                )

                search_keywords = call_model(
                    build_search_keywords_prompt,
                    os.environ.get("MODEL_NAME", "gpt-4.1"),
                )
                try:
                    successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
                        task_search_keywords=search_keywords,
                        task_question=problem["problem"],
                        successful_topk=os.environ.get("SUCCESSFUL_TOPK", 2),
                        failed_topk=os.environ.get("FAILED_TOPK", 1),
                        insight_topk=os.environ.get("INSIGHTS_TOPK", 3),
                        threshold=os.environ.get("THRESHOLD", 0.3),
                    )
                    logger.info(f"the number of successful trajectories: {len(successful_trajectories)}")
                    logger.info(f"the number of insights: {len(insights)}")
                    additional_knowledge = "\n\n".join(
                        [
                            trajectory.task_question + "\n" + trajectory.task_trajectory
                            for trajectory in successful_trajectories
                        ]
                    )
                    additional_knowledge += "\n\n".join(list(insights))
                except Exception as e:
                    additional_knowledge = None
                    logger.error(f"Error retrieving memory: {str(e)}")
            additional_args = {"additional_knowledge": additional_knowledge}
            result = self.agent.run(augmented_question, additional_args=additional_args)

            final_answer = self.extract_final_answer(result)
            semantic_match_prompt = prompts["semantic_match_prompt"].format(
                question=problem["problem"],
                prediction=final_answer,
                true_answer=problem["solution"],
            )
            semantic_check = call_model(query=semantic_match_prompt, model_name="gpt-4.1")

            if (not question_scorer(final_answer, problem["solution"])) or (semantic_check == "false"):
                manager_agent_steps = []

                for memory_step in self.agent.memory.steps:
                    memory_step.model_input_messages = None
                    step_dict = memory_step.dict()
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

                annotated_example = {
                    "question": problem["problem"],
                    "prediction": final_answer,
                    "ground_truth": problem["solution"],
                    "manager_agent_steps": manager_agent_steps,
                }

                suggestion_prompt = prompts["failure_attribution_and_suggestion_prompt"].format(
                    knowledge=additional_knowledge, agent_log=str(annotated_example)
                )

                suggestion = call_model(query=suggestion_prompt, model_name="gpt-4.1")
                logger.info(f"suggestion by memory: {suggestion}")
                manager_suggestion, search_suggestion = self._parse_suggestions(suggestion)
                logger.info(f"Parsed manager suggestion: {manager_suggestion}")
                logger.info(f"Parsed search suggestion: {search_suggestion}")
                additional_args = {}
                if manager_suggestion:
                    additional_args["manager_suggestion"] = manager_suggestion
                if search_suggestion:
                    additional_args["search_suggestion"] = search_suggestion

                # Fallback to the raw suggestion if parsing fails
                if not additional_args and suggestion:
                    additional_args["suggestion"] = suggestion

                final_result = self.agent.run(augmented_question, additional_args=additional_args)
                final_answer = self.extract_final_answer(final_result)

            conversation_messages = self._extract_conversation_history(augmented_question, final_answer)
            self.conversation_history.extend(conversation_messages)

            manager_agent_steps = []

            for memory_step in self.agent.memory.steps:
                memory_step.model_input_messages = None
                step_dict = memory_step.dict()
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

            return {
                "messages": conversation_messages,
                "final_answer": final_answer,
                "manager_agent_steps": manager_agent_steps,
                "search_keywords": search_keywords,
            }

        except Exception as e:
            error_message = f"Error running smolagents: {str(e)}"
            error_ai_message = {
                "content": error_message,
                "name": "smolagents_error",
                "role": "assistant",
                "message_type": "error_response",
                "usage_metadata": None,
            }
            return {
                "messages": [error_ai_message],
                "final_answer": error_message,
                "error": str(e),
            }

    def get_agent_info(self) -> Dict[str, Any]:
        base_info = super().get_agent_info()
        base_info.update(
            {
                "model_name": self.model_name,
                "search_tools": [tool.__class__.__name__ for tool in getattr(self, "search_tools", [])],
                "manager_tools": [tool.__class__.__name__ for tool in getattr(self, "manager_tools", [])],
                "max_steps": self.config.get("max_steps", 10),
                "search_max_steps": self.config.get("search_max_steps", 15),
                "agent_hierarchy": "hierarchical",
                "agents": [{"name": w.name, "type": w.agent.__class__.__name__} for w in self.workers],
                "smolagents_available": True,
            }
        )
        return base_info


AgentSystemRegistry.register(
    "simple_smolagents",
    SimpleSmolagentsAgent,
    max_steps=15,
    search_max_steps=10,
    verbosity_level=1,
)
