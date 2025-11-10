"""
Swarm Agent System

This module implements a swarm-based multi-agent system where multiple agents
work collaboratively to solve problems, with each agent working independently
and then aggregating their results.
"""

import time
import uuid
import os
from typing import Dict, Any, List
import asyncio

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SwarmAgent:
    """Individual agent in the swarm"""

    def __init__(self, agent_id: str, model_name: str = None, system_prompt: str = None):
        """
        Initialize a swarm agent.

        Args:
            agent_id: Unique identifier for this agent
            model_name: LLM model to use
            system_prompt: Custom system prompt for this agent
        """
        self.agent_id = agent_id
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            system_prompt
            or "You are an intelligent AI assistant specialized in solving problems carefully and step by step."
        )
        self.llm = ChatOpenAI(model=self.model_name)
        self.name = agent_id

    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem independently.

        Args:
            problem: The problem to solve

        Returns:
            Dictionary with the solution and AI message with usage metadata
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._create_prompt(problem)},
        ]

        start_time = time.time()
        response = await self.llm.ainvoke(messages)
        end_time = time.time()

        ai_message = response
        ai_message.id = f"{self.agent_id}_{uuid.uuid4()}"
        ai_message.name = self.agent_id
       
        return {
            "agent_id": self.agent_id,
            "solution": response.content,
            "message": ai_message,
            "latency_ms": (end_time - start_time) * 1000,
        }

    def _create_prompt(self, problem: str) -> str:
        """Create a tailored prompt for this agent"""
        return f"""
Please solve the following problem:

{problem}

Think carefully about the problem step by step. Show your work and reasoning.
For mathematical problems, make sure to provide your final answer in a clear format.

Agent ID: {self.agent_id}
"""


class Aggregator:
    """Aggregates results from swarm agents to produce a final solution"""

    def __init__(self, model_name: str = None, format_prompt: str = None):
        """
        Initialize the aggregator.

        Args:
            model_name: LLM model to use for aggregation
        """
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(model=self.model_name)
        self.name = "aggregator"
        self.format_prompt = format_prompt

    async def aggregate(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate solutions from multiple agents.

        Args:
            problem: Original problem
            solutions: List of agent solutions

        Returns:
            Aggregated solution and AI message with usage metadata
        """
        solutions_text = "\n\n".join([f"Agent {sol['agent_id']} solution:\n{sol['solution']}" for sol in solutions])

        prompt = f"""
I need you to analyze multiple solutions to the same problem and provide the most accurate answer.

The original problem:
{problem}

The solutions from different agents:
{solutions_text}

Please carefully analyze these solutions, identify the correct approach, and provide the final answer.
Make sure your final answer is clearly formatted and precise.

{self.format_prompt}
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert aggregator that analyzes multiple solutions and determines the most accurate one.",
            },
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        response = await self.llm.ainvoke(messages)
        end_time = time.time()

        ai_message = response
        ai_message.id = f"aggregator_{uuid.uuid4()}"
        ai_message.name = self.name


        return {
            "final_solution": response.content,
            "message": ai_message,
            "latency_ms": (end_time - start_time) * 1000,
        }


class SwarmSystem(AgentSystem):
    """
    Swarm Agent System

    This agent system uses multiple independent agents working in parallel,
    with results aggregated to produce a final solution.
    """

    def __init__(self, name: str = "swarm", config: Dict[str, Any] = None):
        """Initialize the Swarm Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.use_parallel = self.config.get("parallel", True)
        
     

    def _create_agents(self, problem_input: Dict[str, Any], feedback: Dict[str, Any] = None) -> Dict[str, List]:
        """Create the swarm agents"""
        # This method will be patched by ToolIntegrationWrapper if this system is wrapped.
        # The wrapper expects a dictionary: {"workers": [worker1, worker2, ...]}
        # Each worker should have a .name and .llm attribute.

        swarm_agents = [
            SwarmAgent(
                agent_id=f"agent_{i + 1}", 
                model_name=self.model_name, 
                system_prompt=self._get_system_prompt()
            )
            for i in range(self.num_agents)
        ]
        
        # Also create the aggregator here if it's to be managed for tools
        aggregator = Aggregator(model_name=self.model_name, format_prompt=self.format_prompt)
        
        return {
            "workers": swarm_agents + [aggregator]
        }

    def _get_system_prompt(self) -> str:
        """Get system prompt for an agent based on its index"""
        base_prompt = "You are an intelligent AI assistant specialized in solving problems carefully and step by step."
      
        return base_prompt

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method implements the actual agent logic without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]
        
        # Create swarm agents and aggregator
        # _create_agents now returns a dict, extract workers
        agent_components_dict = self._create_agents(problem)
        all_workers = agent_components_dict.get("workers", [])
        
        agents = [w for w in all_workers if isinstance(w, SwarmAgent)]
        # Find the aggregator instance; assumes only one.
        aggregators = [w for w in all_workers if isinstance(w, Aggregator)]
        if not aggregators:
            raise ValueError("Aggregator not found among workers created by _create_agents.")
        aggregator = aggregators[0]

        # Collect agent solutions and messages
        agent_solutions = []
        all_messages = []

        if self.use_parallel:
            # Solve problems in parallel
            tasks = [agent.solve(problem_text) for agent in agents]
            solutions = await asyncio.gather(*tasks)
            agent_solutions.extend(solutions)
        else:
            # Solve problems sequentially
            for agent in agents:
                solution = await agent.solve(problem_text)
                agent_solutions.append(solution)

        # Extract messages from solutions
        for sol in agent_solutions:
            all_messages.append(sol["message"])

        # Aggregate solutions
        agg_result = await aggregator.aggregate(problem_text, agent_solutions)
        all_messages.append(agg_result["message"])
        
        return {
            "messages": all_messages,
            "final_answer": agg_result["final_solution"],
        }


# Register the agent system
AgentSystemRegistry.register("swarm", SwarmSystem, num_agents=3)
