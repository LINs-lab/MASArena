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

        # 实例化 BenchAgent
        # 使用 AutoGen 的 config 来配置 BenchAgent
        bench_agent_config = {
            "model": self.config.get("model_name", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "max_steps": self.config.get("max_steps", 15),
            "search_tools": ["search", "browser", "wikipedia"], # 示例工具
            # 可以在此处添加 BenchAgent 构造函数接收的其他参数
            **self.config.get("bench_agent_kwargs", {}) 
        }
        
        # 实例化 BenchAgent
        # ！！！注意：这里假设 BenchAgent 类已正确导入
        self.bench_agent = BenchAgent(**bench_agent_config)

        self.num_rounds = self.config.get("num_rounds", 5)

        # 代理定义保持不变，但它们的执行将通过 BenchAgent 完成
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
        
        # 将问题本身作为附加参数传递给 run_agent_step，以便其可以在记忆检索等中使用
        additional_args = {"problem": problem_text, **kwargs} 

        for round_idx in range(self.num_rounds):
            for agent in self.agents:
                agent_name = agent["name"]
                agent_prompt = agent["system_prompt"]
                
                # 构造用于 LLM 调用的完整 prompt
                # 格式: [System Prompt] + [Conversation History] + [User's/Agent's Turn Prompt]
                
                # 在这里，我们将整个对话历史和当前 Agent 的角色/指令结合起来，
                # 形成一个“增强型问题” (augmented_question) 传递给 BenchAgent.run_agent_step
                
                # 1. 构建完整的输入历史 (包含 System Prompt)
                full_input_prompt = [
                    f"System Prompt for {agent_name}: {agent_prompt}",
                    "\n--- Conversation History ---\n",
                ]
                
                # 添加历史消息
                for msg in conversation_history:
                    # 将历史消息格式化为纯文本，以便传递给 BenchAgent.run_agent_step
                    # BenchAgent 的 run_agent_step 接收一个 augmented_question 字符串
                    role = msg.get("name") or msg.get("role")
                    full_input_prompt.append(f"{role.upper()}: {msg['content']}")
                    
                # 2. 添加当前 Agent 的操作提示
                # primary 代理总是接收最新的输入并进行处理
                if agent_name == "primary":
                    # primary 代理会看到批评家的反馈，并被要求生成/修正内容
                    current_turn_prompt = f"\n\nNow, as the {agent_name.upper()} agent, based on the above discussion, generate the initial content or revise it. DO NOT use the critic's system prompt. Respond directly with your content."
                elif agent_name == "critic":
                    # critic 代理会看到 primary 代理的最新输出，并被要求进行评论
                    current_turn_prompt = f"\n\nNow, as the {agent_name.upper()} agent, provide constructive feedback on the {conversation_history[-1].get('name', 'assistant')}'s latest output. Respond with 'APPROVE' if you find it satisfactory."
                
                # 将所有部分合并成 BenchAgent 期望的 augmented_question 字符串
                augmented_question = "\n".join(full_input_prompt) + current_turn_prompt

                # ===============================================
                # 核心修改：使用 self.bench_agent.run_agent_step 替换 LLM 调用
                # ===============================================
                try:
                    # run_agent_step 是异步方法，需要 await
                    result = await self.bench_agent.run_agent_step(
                        augmented_question=augmented_question, 
                        additional_args=additional_args
                    )
                    
                    response_content = result["final_answer"]
                    
                    # BenchAgent.run_agent_step 返回的 final_answer 通常是最终的文本回复
                    ai_message = {
                        'content': response_content,
                        'name': agent_name,
                        'role': 'assistant', # 尽管是 BenchAgent 的输出，但在 AutoGen 对话中作为 'assistant' 角色
                        'message_type': 'ai_response',
                        # 'usage_metadata': result.get("usage_metadata"), # BenchAgent 返回结构可能不同，此处省略
                        # 记录 BenchAgent 内部的步骤日志
                        'bench_agent_steps': result.get("manager_agent_steps", []) + result.get("search_agent_steps", []),
                    }

                    # 将 BenchAgent 的最终回复添加到对话历史中
                    conversation_history.append({"role": "assistant", "content": response_content, "name": agent_name})

                    if agent_name == "primary":
                        # primary agent 的回复是当前的答案
                        final_answer = response_content
                        print("final_answer:",final_answer)

                    # 检查是否满足退出条件 (Critic 批准)
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
                    # 遇到错误则提前退出
                    return {
                        "messages": all_messages,
                        "final_answer": final_answer if final_answer else error_message
                    }

        # 达到最大轮次仍未批准
        print("final_answer:",final_answer)
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }



AgentSystemRegistry.register("autogen", AutoGen)
