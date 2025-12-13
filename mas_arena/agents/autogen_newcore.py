import os
from typing import Dict, Any ,Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool

load_dotenv()

class AutoGenAgent:
    def __init__(
        self,
        agent_id: str,
        bench_agent_params: Dict[str, Any],
        messages: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.messages = messages
        # 复制一份参数并覆盖 name
        params = dict(bench_agent_params)
        params["name"] = agent_id
        # model 名称沿用传入参数
        self.llm = BenchAgent(**params)
        self.name = agent_id
    
    async def solve(self) -> Dict[str, Any]:
        run_result = await self.llm.run_agent_step(
            augmented_question=self.messages,
            additional_args={},
        )
        print("--------------------------------------------")
        print(run_result)
        print("--------------------------------------------")
        return {
            "agent_id": self.agent_id,
            "solution": run_result.get("final_answer"),
            "messages": run_result.get("messages", []),
            "usage": run_result['messages'][1]['usage_metadata']
        }

class AutoGen(AgentSystem):

    def __init__(self, name: str = "autogen", config: Dict[str, Any] = None):
        """Initialize the AutoGen System"""
        super().__init__(name, config)
        self.config = config or {}

        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")

        self.num_rounds = self.config.get("num_rounds", 5)
        
        self.agents_configs = [
            {
                "name": "primary",
                "system_prompt": """You are a helpful AI assistant, skilled at generating creative and accurate content."""
            },
            {
                "name": "critic",
                "system_prompt": "Provide constructive feedback on the content provided. Respond with 'APPROVE' when the content meets high standards or your feedback has been addressed."
            }
        ]
        
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
        # self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        
        problem_text = problem["problem"]
        messages = [
            {"role": "user", "content": f"Problem: {problem_text}"}
        ]
        conversation_history = messages.copy()

        all_messages = []
        final_answer = ""

        for _ in range(self.num_rounds):
            for n, agent in enumerate(self.agents_configs):
                agent_name = agent["name"]
                agent_prompt = agent["system_prompt"]

                agent_messages = [
                    {"role": "system", "content": agent_prompt},
                    *conversation_history
                ]

                response = await AutoGenAgent(
                    agent_id=f"agent_{1}",
                    bench_agent_params=self.bench_agent_params,
                    messages=self._ai_message_to_str(agent_messages),
                ).solve()

                response_content = response["solution"]

                ai_message = {
                    'content': response_content,
                    'name': agent_name,
                    'role': 'assistant',
                    'message_type': 'ai_response',
                    'usage_metadata': response["usage"]
                }

                conversation_history.append({"role": "assistant", "content": response_content, "name": agent_name})

                if (agent_name == "primary"):
                    final_answer = ai_message["content"]

                if agent_name == "critic" and "approve" in response_content.lower():
                    return {
                        "messages": all_messages,
                        "final_answer": final_answer
                    }
                all_messages.append(ai_message)

        return {
            "messages": all_messages,
            "final_answer": final_answer
        }
    def _ai_message_to_str(self, agent_messages):
        lines = []
        for msg in agent_messages:
            role = msg.get("role", "unknown").upper()
            content = str(msg.get("content", "")) 
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)
        


AgentSystemRegistry.register("autogen", AutoGen)
