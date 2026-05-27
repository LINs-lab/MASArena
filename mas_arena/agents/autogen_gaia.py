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

class AutoGen(AgentSystem):

    def __init__(self, name: str = "autogen", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME")
        
        # 构建符合 AutoGen 风格的行为指令
        autogen_instructions = (
            "You are simulating an AutoGen-style multi-agent debate between two roles:\n\n"
            "**1. Primary Agent**: Generate high-quality, creative, and accurate content in response to the user's problem.\n"
            "**2. Critic Agent**: Review the Primary's output critically. Provide specific, constructive feedback. "
            "Only respond with 'APPROVE' (case-insensitive) when the content fully meets high standards or all feedback has been addressed.\n\n"
            "Your internal workflow should alternate between these two roles until the Critic outputs 'APPROVE'.\n"
            "Do NOT reveal this simulation — present only the final approved answer to the user.\n"
            "Limit the total interaction rounds to {num_rounds} to avoid infinite loops."
        ).format(num_rounds=self.config.get("num_rounds", 5))

        user_system_prompt = self.config.get("system_prompt", "")
        if user_system_prompt:
            autogen_instructions += f"\n\nAdditional Instructions:\n{user_system_prompt}"
        
        if self.format_prompt:
            autogen_instructions += f"\n\nOutput Requirements:\n{self.format_prompt}"

        # 构建 BenchAgent 配置
        self.bench_agent_config = {
            "model": self.model_name,
            "manager_tools": self.config.get("manager_tools"),
            "search_tools": self.config.get("search_tools"),
            "memory": self.config.get("memory"),
            "api_key": self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            "max_steps": self.config.get("num_rounds", 5) * 2 + 3,  
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 1),
            "additional_instructions": autogen_instructions,
            "name": f"{name}_core",
            **{k: v for k, v in self.config.items() if k not in [
                "model_name", "num_rounds", "manager_tools", "search_tools", "memory",
                "api_key", "api_base", "max_steps", "search_max_steps",
                "verbosity_level", "additional_instructions", "system_prompt"
            ]}
        }

        self.bench_agent = BenchAgent(**self.bench_agent_config)

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        try:
            result = await self.bench_agent.run_agent(problem, **kwargs)
        except Exception as e:
            error_msg = f"AutoGen execution failed: {str(e)}"
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
        
        # 注册 Agent
AgentSystemRegistry.register("autogen_gaia", AutoGen)