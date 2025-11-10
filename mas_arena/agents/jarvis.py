from __future__ import annotations
import asyncio
import copy
from typing import Dict, List
import json
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import os

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain_core.callbacks.manager import Callbacks
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from openai.types.completion_usage import CompletionUsage


from mas_arena.agents.base import AgentSystem, AgentSystemRegistry


DEMONSTRATIONS: list[dict] = []


class MessageCollectorCallback(BaseCallbackHandler):
    """Callback to collect AIMessages and their usage metadata."""

    def __init__(self, name: str = "jarvis"):
        super().__init__()
        self.messages: List[Any] = []
        self.name = name

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called at the end of an LLM call."""
        if response.generations:
            generation = response.generations[0][0]
            message = generation.message
            message.name = self.name
            
            if response.llm_output and "token_usage" in response.llm_output:
                token_usage_data = response.llm_output["token_usage"]
                # Create a CompletionUsage object to match what base.py expects
                message.usage_metadata = CompletionUsage(
                    completion_tokens=token_usage_data.get("completion_tokens", 0),
                    prompt_tokens=token_usage_data.get("prompt_tokens", 0),
                    total_tokens=token_usage_data.get("total_tokens", 0),
                )

            self.messages.append(message)


class Task:
    """Task to be executed."""

    def __init__(self, task: str, id: int, dep: List[int], args: Dict, tool: BaseTool):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool
        self.status = "pending"
        self.message = ""
        self.result = ""

    def __str__(self) -> str:
        return f"{self.task}({self.args})"

    def save_product(self) -> None:
        """Save text-based products to result field."""
        # For text-based tasks, we directly store the result
        # No file saving needed for text outputs
        if hasattr(self, 'product'):
            self.result = str(self.product)

    def completed(self) -> bool:
        return self.status == "completed"

    def failed(self) -> bool:
        return self.status == "failed"

    def pending(self) -> bool:
        return self.status == "pending"

    def run(self) -> str:
        """Execute the task using the associated tool."""
        try:
            new_args = copy.deepcopy(self.args)
            # For text-based tasks, execute tool and get result
            result = self.tool(**new_args)
            
            # Store result directly for text-based outputs
            if isinstance(result, str):
                self.result = result
            else:
                # If tool returns complex object, store as product and convert to string
                self.product = result
                self.save_product()
                
        except Exception as e:
            self.status = "failed"
            self.message = str(e)
            return self.message

        self.status = "completed"
        return self.result

class Step:
    """A step in the plan."""

    def __init__(
        self, task: str, id: int, dep: List[int], args: Dict[str, str], tool: BaseTool
    ):
        self.task = task
        self.id = id
        self.dep = dep
        self.args = args
        self.tool = tool

class Plan:
    """A plan to execute."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    def __str__(self) -> str:
        return str([str(step) for step in self.steps])

    def __repr__(self) -> str:
        return str(self)


class BasePlanner(BaseModel):
    """Base class for a planner."""

    @abstractmethod
    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decide what to do."""

    @abstractmethod
    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decide what to do."""

class TaskPlanner(BasePlanner):
    """Planner for tasks."""

    llm_chain: LLMChain
    output_parser: PlanningOutputParser
    stop: Optional[List] = None
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm: BaseLanguageModel, verbose: bool = False):
        """
        Initialize task planner.
        
        Args:
            llm: base language model.
            verbose: whether to enable detailed logging.
        """
        llm_chain = TaskPlaningChain.from_llm(llm, verbose=verbose)
        output_parser = PlanningOutputParser()
        stop = None
        
        # use pydantic's correct initialization method
        super().__init__(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=stop
        )

    def plan(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> Plan:
        """Given input, decided what to do."""
        inputs["tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

    async def aplan(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> Plan:
        """Asynchronous Given input, decided what to do."""
        inputs["hf_tools"] = [
            f"{tool.name}: {tool.description}" for tool in inputs["hf_tools"]
        ]
        llm_response = await self.llm_chain.arun(
            **inputs, stop=self.stop, callbacks=callbacks
        )
        return self.output_parser.parse(llm_response, inputs["hf_tools"])

class ResponseGenerator:
    """Generates a response based on the input."""

    def __init__(self, llm_chain: LLMChain, stop: Optional[List] = None):
        self.llm_chain = llm_chain
        self.stop = stop

    def generate(self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Given input, decided what to do."""
        llm_response = self.llm_chain.run(**inputs, stop=self.stop, callbacks=callbacks)
        # print(f"[Jarvis-Debug] ResponseGenerator.generate LLM response: {llm_response}")
        return llm_response

    def run(self, problem: str, task_list: str, executed_task_list: str, format_prompt: str = None, **kwargs) -> str:
        """
        Run response generation, compatible with the calling method in evaluation_framework.
        
        Args:
            problem: original problem
            task_list: task list
            executed_task_list: execution result list
            format_prompt: format prompt, for guiding output format
        """
        # 构建基础输入
        task_execution = f"Problem: {problem}\nTasks: {task_list}\nResults: {executed_task_list}"
        
        # 如果提供了格式化提示，将其作为指令，否则使用通用指令
        format_instructions = "Please summarize the results and generate a response."
        if format_prompt:
            format_instructions = format_prompt
        
        inputs = {
            "task_execution": task_execution,
            "format_instructions": format_instructions
        }
        return self.generate(inputs, **kwargs)

class TaskExecutor:
    """Load tools and execute tasks."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.tasks = []
        self.id_task_map = {}
        self.status = "pending"
        for step in self.plan.steps:
            task = Task(step.task, step.id, step.dep, step.args, step.tool)
            self.tasks.append(task)
            self.id_task_map[step.id] = task

    def completed(self) -> bool:
        return all(task.completed() for task in self.tasks)

    def failed(self) -> bool:
        return any(task.failed() for task in self.tasks)

    def pending(self) -> bool:
        return any(task.pending() for task in self.tasks)

    def check_dependency(self, task: Task) -> bool:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            if dep_task.failed() or dep_task.pending():
                return False
        return True

    def update_args(self, task: Task) -> None:
        for dep_id in task.dep:
            if dep_id == -1:
                continue
            dep_task = self.id_task_map[dep_id]
            for k, v in task.args.items():
                if f"<resource-{dep_id}>" in v:
                    task.args[k] = task.args[k].replace(
                        f"<resource-{dep_id}>", dep_task.result
                    )

    def run(self) -> str:
        # for task in self.tasks:
        for task in self.tasks:
            # print(f"running {task}")  # noqa: T201
            if task.pending() and self.check_dependency(task):
                self.update_args(task)
                task.run()
        if self.completed():
            self.status = "completed"
        elif self.failed():
            self.status = "failed"
        else:
            self.status = "pending"
        return self.status

    def __str__(self) -> str:
        result = ""
        for task in self.tasks:
            result += f"{task}\n"
            result += f"status: {task.status}\n"
            if task.failed():
                result += f"message: {task.message}\n"
            if task.completed():
                result += f"result: {task.result}\n"
        return result

    def __repr__(self) -> str:
        return self.__str__()

    def describe(self) -> str:
        return self.__str__()
    
class TaskPlaningChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        demos: List[Dict] = DEMONSTRATIONS,
        verbose: bool = False,  # set to False to reduce output
    ) -> LLMChain:
        """Get the response parser."""
        system_template = """#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{{"task": task, "id": task_id, "dep": dependency_task_id, "args": {{"input name": text may contain <resource-dep_id>}}}}]. The special tag "dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The task MUST be selected from the following tools (along with tool description, input name and output type): {tools}. There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON []."""  # noqa: E501
        human_template = """Now I input: {input}."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        demo_messages: List[
            Union[HumanMessagePromptTemplate, AIMessagePromptTemplate]
        ] = []
        for demo in demos:
            if demo["role"] == "user":
                demo_messages.append(
                    HumanMessagePromptTemplate.from_template(demo["content"])
                )
            else:
                demo_messages.append(
                    AIMessagePromptTemplate.from_template(demo["content"])
                )
            # demo_messages.append(message)

        prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, *demo_messages, human_message_prompt]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)

class PlanningOutputParser(BaseModel):
    """Parses the output of the planning stage."""
    
    class Config:
        arbitrary_types_allowed = True

    def parse(self, text: str, hf_tools: List[BaseTool]) -> Plan:
        """Parse the output of the planning stage.

        Args:
            text: The output of the planning stage.
            hf_tools: The tools available.

        Returns:
            The plan.
        """
        steps = []
        try:
            # try to find JSON array
            json_match = re.findall(r"\[.*\]", text)
            if not json_match:
                # if no JSON array is found, return empty plan
                return Plan(steps=[])
            
            # try to parse JSON
            try:
                task_list = json.loads(json_match[0])
            except json.JSONDecodeError:
                # JSON parsing failed, return empty plan
                return Plan(steps=[])
            
            # process task list
            for v in task_list:
                if not isinstance(v, dict) or "task" not in v:
                    continue
                    
                choose_tool = None
                for tool in hf_tools:
                    if tool.name == v["task"]:
                        choose_tool = tool
                        break
                        
                if choose_tool:
                    steps.append(Step(v["task"], v["id"], v["dep"], v["args"], tool))
                    
        except Exception as e:
            # any other error, return empty plan
            pass
            
        return Plan(steps=steps)

class ResponseGenerationChain(LLMChain):
    """Chain to execute tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: bool = False) -> LLMChain:
        execution_template = (
            "The AI assistant has parsed the user input into several tasks"
            "and executed them. The results are as follows:\n"
            "{task_execution}"
            "\n{format_instructions}"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["task_execution", "format_instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


def load_response_generator(llm: BaseLanguageModel) -> ResponseGenerator:
    """Load the ResponseGenerator."""

    llm_chain = ResponseGenerationChain.from_llm(llm, verbose=False)
    return ResponseGenerator(
        llm_chain=llm_chain,
    ) 

def load_chat_planner(llm: BaseLanguageModel) -> TaskPlanner:
    """Load the chat planner."""

    return TaskPlanner(llm = llm)

class HuggingGPT:
    """Agent for interacting with HuggingGPT - Text Processing Version."""

    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], name: str = "jarvis"):
        self.llm = llm
        self.tools = tools
        self.name = name
        self.task_planner = load_chat_planner(llm)
        self.response_generator = load_response_generator(llm)
        self.task_executor = None

    def run(self, input: str) -> str:
        """Process text input through planning, execution, and response generation."""
        # Plan tasks based on input
        plan = self.task_planner.plan(inputs={"input": input, "hf_tools": self.tools})
        
        # Execute planned tasks
        self.task_executor = TaskExecutor(plan)
        execution_status = self.task_executor.run()
        
        # Generate response based on execution results
        response = self.response_generator.generate(
            {"task_execution": self.task_executor}
        )
        return response
    
    def get_plan(self, input: str):
        """Get the execution plan for debugging purposes."""
        return self.task_planner.plan(inputs={"input": input, "hf_tools": self.tools})
    
    def get_execution_details(self):
        """Get detailed execution information for analysis."""
        if hasattr(self, 'task_executor') and self.task_executor:
            return self.task_executor.describe()
        return "No execution performed yet."

    def run_with_trace(self, problem: str, format_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        Run HuggingGPT and return the result with the complete execution trace.
        
        Args:
            problem: input problem
            format_prompt: format prompt, for guiding output format
            **kwargs: other parameters
        """
        message_collector = MessageCollectorCallback(name=self.name)
        # Add collector to callbacks
        callbacks = kwargs.get("callbacks", [])
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        callbacks.append(message_collector)
        kwargs["callbacks"] = callbacks

        # temporarily disable langchain detailed output
        import logging
        import warnings
        
        # disable various log outputs
        loggers_to_silence = [
            "langchain",
            "langchain.chains", 
            "langchain.schema",
            "httpx",
        ]
        
        old_levels = {}
        for logger_name in loggers_to_silence:
            logger = logging.getLogger(logger_name)
            old_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)
        
        # suppress warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        try:
            # if format prompt is provided, add it to the problem
            formatted_problem = problem
            if format_prompt:
                formatted_problem = f"{problem}\n\n{format_prompt}"
            
            # use unified task_planner
            task_list = self.task_planner.plan(inputs={"input": formatted_problem, "hf_tools": self.tools}, **kwargs)
        finally:
            # restore original log levels
            for logger_name, old_level in old_levels.items():
                logging.getLogger(logger_name).setLevel(old_level)
            warnings.filterwarnings("default", category=DeprecationWarning)
        
        if not task_list.steps:
            # disable response_generator's output again
            old_levels = {}
            for logger_name in loggers_to_silence:
                logger = logging.getLogger(logger_name)
                old_levels[logger_name] = logger.level
                logger.setLevel(logging.ERROR)
            
            try:
                # pass format prompt to response_generator
                response_input = f"Problem: {problem}\nTasks: []\nResults: []"
                if format_prompt:
                    response_input += f"\n\nFormat Requirements: {format_prompt}"
                
                response = self.response_generator.run(
                    problem=problem,
                    task_list="[]",
                    executed_task_list="[]",
                    format_prompt=format_prompt,
                    **kwargs
                )
            finally:
                for logger_name, old_level in old_levels.items():
                    logging.getLogger(logger_name).setLevel(old_level)
            return {
                "tasks": [],
                "executed_tasks": [],
                "response": response,
                "messages": message_collector.messages or [("ai", response)] # simulate message history
            }

        # execute tasks
        self.task_executor = TaskExecutor(task_list)
        execution_status = self.task_executor.run()
        
        # collect execution results
        executed_task_list = []
        for task in self.task_executor.tasks:
            executed_task_list.append({
                "task": task.task,
                "id": task.id,
                "status": task.status,
                "result": task.result,
                "message": task.message
            })
        
        # convert Plan object to serializable format
        task_list_serializable = []
        for step in task_list.steps:
            task_list_serializable.append({
                "task": step.task,
                "id": step.id,
                "dep": step.dep,
                "args": step.args
            })
        
        # disable response_generator's output again
        old_levels = {}
        for logger_name in loggers_to_silence:
            logger = logging.getLogger(logger_name)
            old_levels[logger_name] = logger.level
            logger.setLevel(logging.ERROR)
        
        try:
            response = self.response_generator.run(
                problem=problem, 
                task_list=json.dumps(task_list_serializable), 
                executed_task_list=json.dumps(executed_task_list),
                format_prompt=format_prompt,
                **kwargs
            )
        finally:
            for logger_name, old_level in old_levels.items():
                logging.getLogger(logger_name).setLevel(old_level)
        
        # collect all messages for evaluation
        messages = message_collector.messages
        if not messages:
            messages = [("ai", response)]

        return {
            "tasks": task_list,
            "executed_tasks": executed_task_list,
            "response": response,
            "messages": messages
        }

class JarvisSingleAgent(AgentSystem):
    """
    AgentSystem wrapper for the HuggingGPT implementation.
    """

    def __init__(self, name: str = "jarvis", config: Dict[str, Any] | None = None):
        super().__init__(name, config)
        self.config = config or {}
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name, temperature=0.0)

        # As per user request, tools are initialized as an empty list
        self.tools: list[BaseTool] = []
        self.agent = HuggingGPT(self.llm, self.tools, name=self.name)
        # self.format_prompt is inherited from AgentSystem and set in super().__init__

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Runs the HuggingGPT agent.
        """
        problem_text = problem["problem"]
        # print(f"[Jarvis-Debug] JarvisSingleAgent.run_agent received problem: {problem_text}")

        result = await asyncio.to_thread(
            self.agent.run_with_trace,
            problem=problem_text,
            format_prompt=self.format_prompt,
            **kwargs
        )

        # print(f"[Jarvis-Debug] JarvisSingleAgent.run_agent final result: {result}")

        return {
            "messages": result.get("messages", []),
            "final_answer": result.get("response", "")
        }

    def _create_agents(self) -> dict[str, list]:
        """
        Create agents for tool integration. For HuggingGPT, we don't have separate agents,
        but we can expose the main LLM instance for tool binding.
        """
        # The wrapper expects a dictionary: {"workers": [worker1, worker2, ...]}
        # Each worker should have a .name and .llm attribute.
        # We can create a pseudo-worker representing the HuggingGPT planner.
        planner_pseudo_worker = type("Worker", (object,), {
            "name": "hugging_gpt_planner",
            "llm": self.llm
        })
        return {"workers": [planner_pseudo_worker]}


AgentSystemRegistry.register(
    "jarvis",
    JarvisSingleAgent
)
