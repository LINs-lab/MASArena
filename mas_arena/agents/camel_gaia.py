import re
import os
import logging
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent

load_dotenv()
logger = logging.getLogger(__name__)

class CamelAgent(AgentSystem):
    def __init__(self, name: str = "camel", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # 构建 Camel 风格的系统指令：模拟 User/Assistant/Critic 协作流程
        camel_instructions = (
            "You are participating in a CAMEL-style multi-agent simulation, but you are the only agent.\n"
            "Your task is to internally simulate a collaborative dialogue between three roles:\n"
            "- **User**: Defines the problem, asks clarifying questions, and signals completion with '<Camel_TASK_DONE>'.\n"
            "- **Assistant**: Provides detailed, accurate solutions based on the User's requests.\n"
            "- **Critic**: Evaluates the Assistant's output for correctness, completeness, and reasonableness.\n\n"
            "However, since you are a single agent, you must **internally perform all three roles sequentially**:\n"
            "1. **First**, act as the *User*: restate the problem clearly and pose it as a task.\n"
            "2. **Then**, act as the *Assistant*: solve the task thoroughly using available tools (code, search, etc.).\n"
            "3. **Finally**, act as the *Critic*: review your own solution, check for errors, and refine if needed.\n"
            "Only after this internal cycle should you produce a final answer.\n"
            "If the task involves code generation (e.g., HumanEval), format your response exactly as:\n"
            "## Implementation Details\n[...]\n## Features Implemented\n[...]\n## Optimizations\n[...]\n## Validated Code\n```python\n[...]\n```\n"
        )
        
        user_system_prompt = self.config.get("system_prompt", "")
        if user_system_prompt:
            camel_instructions += f"\n\nAdditional Instructions:\n{user_system_prompt}"
        
        if self.format_prompt:
            camel_instructions += f"\n\nOutput Requirements:\n{self.format_prompt}"

        # 构建 BenchAgent 配置
        self.bench_agent_config = {
            "model": self.model_name,
            "manager_tools": self.config.get("manager_tools"),
            "search_tools": self.config.get("search_tools"),
            "memory": self.config.get("memory"),
            "api_key": self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            "max_steps": self.config.get("max_steps", 20),  # 略高于 Jarvis，因需模拟多角色
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 1),
            "additional_instructions": camel_instructions,
            "name": f"{name}_core",
            **{k: v for k, v in self.config.items() if k not in [
                "model_name", "manager_tools", "search_tools", "memory",
                "api_key", "api_base", "max_steps", "search_max_steps",
                "verbosity_level", "additional_instructions", "system_prompt"
            ]}
        }
        
        self.bench_agent = BenchAgent(**self.bench_agent_config)

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            result = await self.bench_agent.run_agent(problem, **kwargs)
        except Exception as e:
            error_msg = f"Camel execution failed: {str(e)}"
            return {
                "messages": [{
                    "role": "assistant",
                    "content": error_msg,
                    "name": self.name,
                    "message_type": "error"
                }],
                "final_answer": error_msg,
                "extracted_answer": "",
                "error": str(e)
            }
        
        final_answer = result.get("final_answer", "No answer generated.")
        return {
            "messages": result.get("messages", []),
            "final_answer": final_answer,
            "extracted_answer": result.get("extracted_answer", final_answer),
            "manager_agent_steps": result.get("manager_agent_steps", []),
            "search_agent_steps": result.get("search_agent_steps", []),
            "search_keywords": result.get("search_keywords", ""),
        }
        
        
AgentSystemRegistry.register("camel", CamelAgent)