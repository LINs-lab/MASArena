"""
Swarm Agent System

This module implements a swarm-based multi-agent system where multiple agents
work collaboratively to solve problems, with each agent working independently
and then aggregating their results.
"""

import time
import uuid
import os
from typing import Dict, Any, List, Union, Optional
import asyncio

from dotenv import load_dotenv
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.agent_core import Tool

# Load environment variables
load_dotenv()


class SwarmAgent:
    """Swarm 中的单个 BenchAgent 包装"""

    def __init__(
        self,
        agent_id: str,
        bench_agent_params: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.system_prompt = (
            system_prompt
            or "You are an intelligent AI assistant specialized in solving problems carefully and step by step."
        )

        # 复制一份参数并覆盖 name
        params = dict(bench_agent_params)
        params["name"] = agent_id
        # model 名称沿用传入参数
        self.llm = BenchAgent(**params)
        self.name = agent_id

    async def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem independently.

        Args:
            problem: The problem to solve

        Returns:
            Dictionary with the solution and AI message with usage metadata
        """
        start_time = time.time()
        run_result = await self.llm.run_agent_step(
            augmented_question=self._create_prompt(problem),
            additional_args={},
        )
        end_time = time.time()

        return {
            "agent_id": self.agent_id,
            "solution": run_result.get("final_answer"),
            "messages": run_result.get("messages", []),
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
    """使用 BenchAgent 进行结果聚合"""

    def __init__(self, bench_agent_params: Dict[str, Any], format_prompt: str = None):
        params = dict(bench_agent_params)
        params["name"] = "aggregator"
        self.llm = BenchAgent(**params)
        self.name = "aggregator"
        self.format_prompt = format_prompt or ""

    async def aggregate(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        solutions_text = "\n\n".join(
            [f"Agent {sol['agent_id']} solution:\n{sol['solution']}" for sol in solutions]
        )

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

        start_time = time.time()
        run_result = await self.llm.run_agent_step(
            augmented_question=prompt,
            additional_args={},
        )
        end_time = time.time()

        return {
            "final_solution": run_result.get("final_answer"),
            "messages": run_result.get("messages", []),
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
        # 保持与 BenchAgent 一致的可选组件参数
        self.bench_agent_params = {
            "model": self.model_name,
            "manager_tools": self.config.get("manager_tools"),
            "search_tools": self.config.get("search_tools"),
            "memory": self.config.get("memory"),
            "api_key": self.config.get("api_key"),
            "api_base": self.config.get("api_base"),
            "max_steps": self.config.get("max_steps", 15),
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 1),
            # 如果未显式提供 additional_instructions，则自动落到 format_prompt（如 math 的 \boxed{} 约束）
            "additional_instructions": self.config.get("additional_instructions")
        }

    def _create_agents(self, problem_input: Dict[str, Any], feedback: Dict[str, Any] = None) -> Dict[str, List]:
        """Create the swarm agents"""
        # This method will be patched by ToolIntegrationWrapper if this system is wrapped.
        # The wrapper expects a dictionary: {"workers": [worker1, worker2, ...]}
        # Each worker should have a .name and .llm attribute.

        swarm_agents = [
            SwarmAgent(
                agent_id=f"agent_{i + 1}",
                bench_agent_params=self.bench_agent_params,
                system_prompt=self._get_system_prompt(),
            )
            for i in range(self.num_agents)
        ]

        # Also create the aggregator here if it's to be managed for tools
        aggregator = Aggregator(bench_agent_params=self.bench_agent_params, format_prompt=self.format_prompt)
        
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
            all_messages.extend(sol.get("messages", []))

        # Aggregate solutions
        agg_result = await aggregator.aggregate(problem_text, agent_solutions)
        all_messages.extend(agg_result.get("messages", []))
        
        return {
            "messages": all_messages,
            "final_answer": agg_result["final_solution"],
        }


# Register the agent system
AgentSystemRegistry.register("swarm", SwarmSystem, num_agents=3)
