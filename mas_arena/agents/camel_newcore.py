import os
import logging
from typing import Dict, Any, List
from openai import AsyncOpenAI
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool

# Load environment variables

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Camel(AgentSystem):
    """
    Simplified LangChain Multi-Agent System
    """

    def __init__(self, name: str = "camel", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = self.config.get("system_prompt", "") + self.format_prompt
        self.max_rounds = 3
        
        bench_agent_config = {
            "model": self.model_name,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "memory": self.config.get("memory"),
            "search_tools": self.config.get("search_tools"),
            "verbosity_level": self.config.get("verbosity_level", 2),
            "additional_instructions": self.config.get("additional_instructions"),
            "manager_tools": self.config.get("manager_tools"),
        }
        self.bench_agent = BenchAgent(**bench_agent_config)
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )


    async def _get_completion(self, system_prompt: str, messages: List[Dict[str, str]], agent_name: str) -> Dict[str, Any]:
    # 构造标准请求消息
        chat_messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            temperature=self.config.get("temperature", 0.7)
        )
        
        content = response.choices[0].message.content
        
        # 返回一个统一的结构：既包含用于对话的 standard_msg，也包含用于记录的 metadata
        return {
            "standard_msg": {"role": "assistant", "content": content}, # 给模型看的
            "full_record": {                                          # 给框架记录看的
                "content": content,
                "name": agent_name,
                "role": "assistant",
                "message_type": "ai_response",
                "usage_metadata": response.usage
            }
        }
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        problem_text = problem["problem"]
        
        all_messages = []        # 框架记录：包含 metadata
        conversation_history = [] # LLM 上下文：纯净的 role/content 列表
        final_answer = ""

        # ---------- 核心对话循环 ----------
        for round_idx in range(self.max_rounds):
            
            # --- 步骤 1: User Evaluator (扮演指令发起者) ---
            user_sys_prompt = (
                f"You are the Question USER. Your goal is to specify tasks and provide feedback.\n"
                f"""If you feel the assistant's response is satisfactory, please include 'TASK_FINISHED' in your reply.\n{self.system_prompt}"""
                
            )
            
            # 构造当前轮次的输入：第一轮发任务，后续轮次看历史反馈
            if round_idx == 0:
                current_input = f"Problem: {problem_text}\nPlease provide the initial instructions to the assistant."
            else:
                current_input = "Evaluate the assistant's previous response. Provide feedback or end the task."

            # 获取 User 角色的回复
            # 注意：传入的是之前的对话历史 + 当前的评价指令
            user_res = await self._get_completion(
                user_sys_prompt, 
                conversation_history + [{"role": "user", "content": current_input}], 
                "user_evaluator"
            )
            
            user_content = user_res["full_record"]["content"]
            all_messages.append(user_res["full_record"])
            
            # 重要：将 User 的回复作为 'user' 角色存入历史，这样 Assistant 才知道这是对它的要求
            conversation_history.append({"role": "user", "content": user_content})

            # 检查终止条件
            if "TASK_FINISHED" in user_content:
                break

            # --- 步骤 2: Assistant Agent (扮演执行者/求解者) ---
            assistant_sys_prompt = (
                f"You are the ASSISTANT SOLVER. Solve the task provided by the user.\n"
                f"Respond with clarity and accuracy.\n{self.system_prompt}"
                
            )
            
            # Assistant 看到的是 user 刚才提出的要求/反馈
            assistant_res = await self._get_completion(
                assistant_sys_prompt,
                conversation_history,
                "assistant_solver"
            )
            
            assistant_content = assistant_res["full_record"]["content"]
            all_messages.append(assistant_res["full_record"])
            final_answer = assistant_content
            
            # 将 Assistant 的回复作为 'assistant' 角色存入历史
            conversation_history.append({"role": "assistant", "content": assistant_content})

        # ---------- 最终总结阶段 ----------
        history_text = ""
        for msg in all_messages[-4:]:
            name = msg.get("name", "Unknown")
            history_text += f"[{name}]: {msg['content']}\n\n"
        extract_user_prompt = f"""You are a professional result analyzer. Synthesize all viewpoints. 

Original problem: {problem_text}

Below are the discussion histories of multiple AI agents:

{history_text}

Please analyze the above discussions and provide a final answer.
Requirements:{self.system_prompt}

""" 
        additional_args = {"problem": problem_text, **kwargs}
        summary_result = await self.bench_agent.run_agent_step(augmented_question=extract_user_prompt, additional_args=additional_args)
        final_answer = summary_result.get("final_answer", "")
        bench_messages = summary_result.get("messages", [])
        all_messages.extend(bench_messages)

        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

# 注册系统
AgentSystemRegistry.register("camel", Camel)
