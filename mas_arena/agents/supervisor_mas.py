"""
Supervisor-based Multi-Agent System

This module implements a supervisor-based multi-agent system where a supervisor
agent coordinates the work of specialized agents.
"""

import os
from typing import Literal, Dict, TypedDict, Any, Optional
import uuid

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

dotenv.load_dotenv()

load_dotenv()


class State(MessagesState):
    next: str


class Router(TypedDict):
    next: Literal["researcher", "coder", "FINISH"]


@traceable
def create_supervisor(model_name: str):
    model = ChatOpenAI(
        model=model_name,
    )
    members = ["researcher", "coder"]

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.\n\n"
        "For mathematical problems, you should first send the request to the 'coder' "
        "who can solve mathematical problems and provide formatted answers. "
        "Only use the 'researcher' if additional information needs to be looked up. "
        "The 'coder' is a mathematical expert who can solve problems directly."
    )

    async def supervisor_node(state: State) -> Command[Literal["researcher", "coder", "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        response = await model.with_structured_output(Router).ainvoke(messages)

        goto = response.get("next", "FINISH")

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


class AgentNode:
    def __init__(self, name: str, model_name: str, prompt: str):
        self.name = name
        self.model_name = model_name
        self.prompt = prompt
        self.llm = ChatOpenAI(model=os.getenv("MODEL_NAME", self.model_name))
        self.agent = None

    def _create_and_get_agent(self):
        if not hasattr(self, "tools") or not self.tools:
            effective_tool_objects = []
        else:
            effective_tool_objects = [t.get("tool_object") for t in self.tools if t.get("tool_object")]

        self.agent = create_react_agent(self.llm, tools=effective_tool_objects, prompt=self.prompt)
        return self.agent

    @traceable
    async def __call__(self, state: State) -> Command[Literal["supervisor"]]:
        current_agent = self._create_and_get_agent() if self.agent is None else self.agent
        
        agent_input = {"messages": state["messages"]}

        result = await current_agent.ainvoke(agent_input)

        ai_message = result["messages"][-1]
        ai_message.name = self.name
        return Command(
            update={"messages": [ai_message]},
            goto="supervisor",
        )


class SupervisorMAS(AgentSystem):
    """
    Supervisor-based Multi-Agent System

    This agent system uses a supervisor to coordinate specialized agents
    for solving problems.
    """

    def __init__(self, name: str = "supervisor_mas", config: Dict[str, Any] = None):
        """Initialize the Supervisor MAS"""
        super().__init__(name, config if config else {})
        
        self.graph = None
        self.workers: Optional[Dict[str, AgentNode]] = None
        

    def _create_agents(self, problem_input: Optional[Any] = None, feedback: Optional[Any] = None) -> Dict[str, AgentNode]:
        # This method will be called by ToolIntegrationWrapper if this system is wrapped.
        # It now returns a dictionary mapping names to AgentNode instances.
        # ToolIntegrationWrapper will find the AgentNode instances from the dict values,
        # modify them (set .tools, rebind .llm), and the patched _create_agents
        # will return this same dictionary structure with modified nodes.
        
        default_model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        researcher_model = self.config.get("researcher_model_name", self.config.get("model_name", default_model))
        coder_model = self.config.get("coder_model_name", self.config.get("model_name", default_model))

        researcher = AgentNode(name="researcher", model_name=researcher_model, prompt=f"""You are a helpful expert researcher,
Use your expertise to help with tasks and provide information. Requirement:
- {self.format_prompt}
        """
        )
        coder = AgentNode(name="coder", model_name=coder_model, prompt=f"""You are a helpful expert coder,
Use your expertise to help with tasks and provide information. Requirement:
- {self.format_prompt}
        """
        )
        
        return {
            "researcher": researcher,
            "coder": coder
        }

    def _init_graph_if_needed(self, problem_input: Optional[Any] = None, feedback: Optional[Any] = None):
        if self.graph is not None:
            return

        # _create_agents now returns a dict {"researcher": researcher_node, "coder": coder_node}
        # If wrapped by ToolIntegrationWrapper, the nodes will have been modified in-place.
        worker_nodes_map = self._create_agents(problem_input=problem_input, feedback=feedback)

        research_node_obj = worker_nodes_map.get("researcher")
        coder_node_obj = worker_nodes_map.get("coder")
        
        if not research_node_obj or not coder_node_obj:
            raise RuntimeError("Could not find researcher or coder agent nodes from _create_agents dictionary.")

        builder = StateGraph(State)
        checkpointer = InMemorySaver()

        supervisor_model = self.config.get("supervisor_model_name", self.config.get("model_name", os.getenv("MODEL_NAME", "gpt-4o-mini")))
        builder.add_node("supervisor", create_supervisor(model_name=supervisor_model))
        
        builder.add_node("researcher", research_node_obj)
        builder.add_node("coder", coder_node_obj)

        builder.add_edge(START, "supervisor")
        
        builder.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"researcher": "researcher", "coder": "coder", END: END},
        )
        
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("coder", "supervisor")

        self.graph = builder.compile(checkpointer=checkpointer)


    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a problem.
        """
        # Get problem content based on evaluator type
        problem_text = problem["problem"]
        self._init_graph_if_needed(problem_input=problem_text)
        
        initial_state = {
            "messages": [("user", problem_text)],
        }

        thread_id = str(uuid.uuid4())
        if self.graph is None:
             raise RuntimeError("Graph not compiled before run_agent call.")

        run_result = await self.graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        
        return {
            "messages": run_result.get("messages", []),
            "final_answer": run_result.get("messages", [])[-1].content if run_result.get("messages", []) else "",
        }


# Register the agent system
AgentSystemRegistry.register("supervisor_mas", SupervisorMAS)
