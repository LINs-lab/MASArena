import os
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

load_dotenv()


class AutoGen(AgentSystem):

    def __init__(self, name: str = "autogen", config: Dict[str, Any] = None):
        """Initialize the AutoGen System"""
        super().__init__(name, config)
        self.config = config or {}

        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")

        self.num_rounds = self.config.get("num_rounds", 5)

        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

        self.agents = [
            {
                "name": "primary",
                "system_prompt": """You are a helpful AI assistant, skilled at generating creative and accurate content."""
            },
            {
                "name": "critic",
                "system_prompt": "Provide constructive feedback on the content provided. Respond with 'APPROVE' when the content meets high standards or your feedback has been addressed."
            }
        ]

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:

        problem_text = problem["problem"]
        messages = [
            {"role": "user", "content": f"Problem: {problem_text}"}
        ]
        conversation_history = messages.copy()

        all_messages = []
        final_answer = ""

        for _ in range(self.num_rounds):
            for n, agent in enumerate(self.agents):
                agent_name = agent["name"]
                agent_prompt = agent["system_prompt"]

                agent_messages = [
                    {"role": "system", "content": agent_prompt},
                    *conversation_history
                ]

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=agent_messages
                )

                response_content = response.choices[0].message.content

                ai_message = {
                    'content': response_content,
                    'name': agent_name,
                    'role': 'assistant',
                    'message_type': 'ai_response',
                    'usage_metadata': response.usage
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


AgentSystemRegistry.register("autogen", AutoGen)
