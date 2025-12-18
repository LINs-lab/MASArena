import os
from typing import Dict, Any ,Optional
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool

load_dotenv()

class AutoGen(AgentSystem):
    """
    使用 BenchAgent 作为核心 LLM 的 AutoGen 增强系统
    """

    def __init__(self, name: str = "autogen", config: Dict[str, Any] = None):
        """Initialize the AutoGen System with BenchAgent"""
        super().__init__(name, config)
        self.config = config or {}

        bench_agent_config = {
            "model": self.config.get("model_name", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "memory": self.config.get("memory"),
            "search_tools": self.config.get("search_tools"),
            "verbosity_level": self.config.get("verbosity_level", 2),
            "additional_instructions": self.config.get("additional_instructions"),
            "manager_tools": self.config.get("manager_tools"),
        }
        
        self.bench_agent = BenchAgent(**bench_agent_config)

        self.num_rounds = self.config.get("num_rounds", 5)

        self.agents = [
            {
                "name": "primary",
                "system_prompt": """You are a helpful AI assistant, skilled at generating creative and accurate content. Your current task is to generate the initial content or revise it based on the critic's feedback. Use the tool 'final_answer' only when the content is approved by the critic."""
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
        problem_text = problem["problem"]
        
        # 初始用户消息
        initial_user_prompt = f"Problem: {problem_text}"
        
        conversation_history = [
            {"role": "user", "content": initial_user_prompt}
        ]

        all_messages = []
        final_answer = ""
        
        additional_args = {"problem": problem_text, **kwargs} 

        for round_idx in range(self.num_rounds):
            for agent in self.agents:
                agent_name = agent["name"]
                agent_prompt = agent["system_prompt"]
                
                full_input_prompt = [
                    f"System Prompt for {agent_name}: {agent_prompt}",
                    "\n--- Conversation History ---\n",
                ]
                
                for msg in conversation_history:
                    role = msg.get("name") or msg.get("role")
                    full_input_prompt.append(f"{role.upper()}: {msg['content']}")
                    
                if agent_name == "primary":
                    current_turn_prompt = f"\n\nNow, as the {agent_name.upper()} agent, based on the above discussion, generate the initial content or revise it. DO NOT use the critic's system prompt. Respond directly with your content."
                elif agent_name == "critic":
                    current_turn_prompt = f"\n\nNow, as the {agent_name.upper()} agent, provide constructive feedback on the {conversation_history[-1].get('name', 'assistant')}'s latest output. Respond with 'APPROVE' if you find it satisfactory."
                
                augmented_question = "\n".join(full_input_prompt) + current_turn_prompt

                try:
                    result = await self.bench_agent.run_agent_step(
                        augmented_question=augmented_question, 
                        additional_args=additional_args
                    )
                    
                    response_content = result["final_answer"]
                    
                    ai_message = {
                        'content': response_content,
                        'name': agent_name,
                        'role': 'assistant', 
                        'message_type': 'ai_response',
                        'bench_agent_steps': result.get("manager_agent_steps", []) + result.get("search_agent_steps", []),
                    }

                    conversation_history.append({"role": "assistant", "content": response_content, "name": agent_name})

                    if agent_name == "primary":
                        final_answer = response_content
                        print("final_answer:",final_answer)

                    if agent_name == "critic" and "approve" in response_content.lower():
                        all_messages.append(ai_message)
                        print("final_answer:",final_answer)
                        return {
                            "messages": all_messages,
                            "final_answer": final_answer
                        }
                        
                    all_messages.append(ai_message)

                except Exception as e:
                    error_message = f"Error during BenchAgent step for {agent_name}: {str(e)}"
                    all_messages.append({
                        'content': error_message,
                        'name': agent_name,
                        'role': 'assistant',
                        'message_type': 'error_response',
                    })
                    return {
                        "messages": all_messages,
                        "final_answer": final_answer if final_answer else error_message
                    }

        print("final_answer:",final_answer)
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }



AgentSystemRegistry.register("autogen", AutoGen)
