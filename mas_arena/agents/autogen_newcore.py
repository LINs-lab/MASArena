import os
from typing import Dict, Any ,Optional
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool
from openai import AsyncOpenAI
from mas_arena.agents.workflow_protocol import bench_settings, client_settings, model_name, request_settings, task_text

class AutoGen(AgentSystem):
    """
    使用 BenchAgent 作为核心 LLM 的 AutoGen 增强系统
    """

    def __init__(self, name: str = "autogen", config: Dict[str, Any] = None):
        """Initialize the AutoGen System with BenchAgent"""
        super().__init__(name, config)
        self.config = config or {}

        bench_agent_config = bench_settings(self.config)
        self.bench_agent = BenchAgent(**bench_agent_config)

        self.num_rounds = self.config.get("num_rounds", 5)
        self.openai_client = AsyncOpenAI(**client_settings(self.config))
        self.openai_model = model_name(self.config)
        self.agents = [
            {
                "name": "primary",
                "system_prompt": """You are a helpful AI assistant, skilled at generating creative and accurate content."""
            },
            {
                "name": "critic",
                "system_prompt": "Provide constructive feedback on the content provided. Respond with the word 'APPROVE' in capital letters when the content meets high standards or your feedback has been successfully addressed. Otherwise, provide detailed revision suggestions."
            }
        ]

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        运行基于 BenchAgent 的多轮对话
        """
        problem_text = task_text(problem)
        
        # 初始用户消息
        initial_user_prompt = f"Problem: {problem_text}"
        conversation_history = [
            {"role": "user", "content": initial_user_prompt}
        ]
        all_messages = []
        final_answer = ""
        additional_args = {"id": problem.get("id", "")}

        for round_idx in range(self.num_rounds):
            for agent in self.agents:
                agent_name = agent["name"]
                agent_prompt = agent["system_prompt"]
                try:
                    if agent_name == "primary":
                        full_input_prompt = [
                            f"System Prompt for {agent_name}: {agent_prompt}",
                            "\n--- Conversation History ---\n",
                        ]
                        for msg in conversation_history:
                            role = msg.get("name") or msg.get("role")
                            full_input_prompt.append(f"{role.upper()}: {msg['content']}")
                        
                        current_turn_prompt = f"\n\nNow, as the PRIMARY agent, generate or revise content. Respond directly.If the input is a multiple-choice question, output the letter of the correct answer ONLY. Do not provide any explanations, context, or additional characters."
                        augmented_question = "\n".join(full_input_prompt) + current_turn_prompt

                        result = await self.bench_agent.run_agent_step(
                            augmented_question=augmented_question, 
                            additional_args=additional_args
                        )
                        bench_messages = result.get("messages", [])
                        all_messages.extend(bench_messages)
                        if result.get("error"):
                            return {
                                "messages": all_messages,
                                "final_answer": result.get("final_answer", ""),
                                "error": result["error"],
                            }
                        
                        response_content = result["final_answer"]
                        final_answer = response_content
                        
                    else:
                        messages = [{"role": "system", "content": agent_prompt}]
                        for msg in conversation_history:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        messages.append({
                            "role": "user", 
                            "content": "Please review the latest response above. Respond with 'APPROVE' if it is correct, or provide feedback."
                        })

                        response = await self.openai_client.chat.completions.create(
                            model=self.openai_model,
                            messages=messages,
                            **request_settings(self.config, temperature=0.7)
                        )
                        
                        response_content = response.choices[0].message.content
                        critic_msg = {
                            'content': response_content,
                            'name': agent_name,
                            'role': 'assistant', 
                            'message_type': 'ai_response',
                            'usage_metadata': response.usage 
                        }
                        all_messages.append(critic_msg)

                    conversation_history.append({"role": "assistant", "content": response_content, "name": agent_name})

                except Exception as e:
                    error_message = f"Error during {agent_name} step: {str(e)}"
                    all_messages.append({
                        'content': error_message,
                        'name': agent_name,
                        'role': 'assistant',
                        'message_type': 'error_response',
                    })
                    return {
                        "messages": all_messages,
                        "final_answer": final_answer if final_answer else error_message,
                        "error": error_message,
                    }

        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

    async def aclose(self):
        await self.bench_agent.aclose()
        await self.openai_client.close()



AgentSystemRegistry.register("autogen", AutoGen)
