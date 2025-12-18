"""
LLM Debate Multi-Agent System

This module implements a multi-agent debate system where multiple LLM agents
engage in multi-round discussions to collaboratively solve problems through debate.
"""

import os
from typing import Dict, Any, List, Optional
import contextlib
import asyncio
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool

# Load environment variables
load_dotenv(override=True)

class LLMDebate(AgentSystem):
    """
    LLM Debate Multi-Agent System (Enhanced with BenchAgent)
    
    This system implements a multi-round debate mechanism where multiple BenchAgents
    (enhanced with tools and memory) discuss and refine their answers.
    """

    def __init__(self, name: str = "llm_debate", config: Dict[str, Any] = None):
        """Initialize the LLM Debate System"""
        super().__init__(name, config)
        self.config = config or {}
        
        # Configuration parameters
        self.agents_num = self.config.get("agents_num", 2)  # Number of debate agents
        self.rounds_num = self.config.get("rounds_num", 3)  # Number of debate rounds
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # System prompt with format requirements
        # BenchAgent 会在其内部处理 format_prompt，因此这里只保留基础提示。
        self.base_system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
        self.system_prompt = self.base_system_prompt 
        
        # BenchAgent Initialization Parameters
        self.bench_agent_params = {
            "model": self.model_name,
            "manager_tools": self.config.get("manager_tools"),
            "search_tools": self.config.get("search_tools"),
            "memory": self.config.get("memory"),
            "api_key": self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 1),
            # additional_instructions 用于将 format_prompt 传递给 BenchAgent
            "additional_instructions": self.config.get("additional_instructions") or self.format_prompt
        }

        # 初始化 BenchAgent 实例列表
        self.debate_agents: List[BenchAgent] = []
        for i in range(self.agents_num):
            agent_name = f"bench_agent_{i+1}"
            agent = BenchAgent(
                name=agent_name,
                config={**self.config, 'evaluator': 'llm_debate'}, # 传递 evaluator 等 config
                **self.bench_agent_params
            )
            self.debate_agents.append(agent)

        # 聚合器也使用一个 BenchAgent
        self.aggregator_agent = BenchAgent(
            name="aggregator_bench_agent",
            config={**self.config, 'evaluator': 'llm_debate_aggregator'},
            **self.bench_agent_params
        )

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the LLM Debate system on a given problem.
        """
        # 1. 从输入中提取问题文本
        query = problem["problem"]
        
        # 2. 启动异步多智能体辩论流程
        #    返回每个智能体在整个辩论过程中的完整对话上下文（消息列表）
        agent_contexts = await self._process_single_question_async(
            query, 
            self.agents_num, 
            self.rounds_num,
            problem.get("expected_answer") # 传递预期答案以辅助 BenchAgent 记忆检索
        )
        
        # 3. 初始化用于记录结果的数据结构
        all_messages = []
        final_answers = []
        
        # 4. 遍历每一轮、每个智能体，提取其在该轮的回答
        for round_idx in range(self.rounds_num):
            for agent_idx, agent_context in enumerate(agent_contexts):
                # 提取每轮 BenchAgent 的 'final_answer' (代表该轮的最新结论)
                # 假设每轮的结果被存储在 agent_context 中（在 _process_single_question_async 中处理）
                response_index = 2 * round_idx + 1 # 理论上是 user_query (0), ai_response (1), user_debate (2), ai_response (3)...
                
                if response_index < len(agent_context):
                    response_msg = agent_context[response_index]
                    
                    # 构造标准化的 AI 消息对象
                    # 注意：这里需要从 BenchAgent 的运行结果中提取信息
                    ai_message = {
                        'content': response_msg.get('final_answer', response_msg.get('content', 'No content found.')),
                        'name': f'debate_agent_{agent_idx+1}',
                        'role': 'assistant',
                        'message_type': 'ai_response',
                        'round': round_idx + 1,
                        'agent_id': f'debate_agent_{agent_idx+1}',
                        # BenchAgent 会返回 usage_metadata
                        'usage_metadata': response_msg.get('usage_metadata') 
                    }
                    all_messages.append(ai_message)
        
        # 5. 提取每个智能体的最终答案
        for agent_context in agent_contexts:
            # 假设最后一个 'content' 是最终答案
            if len(agent_context) >= 2:
                # 提取最后一个 BenchAgent 运行结果中的 final_answer
                last_result = agent_context[-1]
                final_answers.append(last_result.get('final_answer', 'Aggregation Error'))
        
        # 6. 调用聚合器 BenchAgent，综合所有智能体的答案生成最终统一答案
        aggregated_answer_result = await self._aggregate_answers(query, final_answers)
        final_answer = aggregated_answer_result.get('final_answer')
        
        # 7. 构造聚合器的消息对象，并加入消息列表
        aggregation_message = {
            'content': final_answer,
            'name': 'debate_aggregator',
            'role': 'assistant',
            'message_type': 'aggregation',
            'usage_metadata': aggregated_answer_result.get('messages', [{}])[-1].get('usage_metadata') # 提取聚合 Agent 的 usage
        }
        all_messages.append(aggregation_message)
        
        # 8. Calculate score if evaluator is available (though BenchmarkRunner usually does this)
        # We can try to use the self.evaluate method logic or let BenchmarkRunner handle it.
        # But to be safe, we should return 'extracted_answer'.
        
        return {
            "messages": all_messages,
            "final_answer": final_answer,
            "extracted_answer": final_answer, # Required by BenchmarkRunner
            "agent_responses": final_answers,
            "rounds_completed": self.rounds_num,
            "agents_participated": self.agents_num
        }

    async def _process_single_question_async(self, question: str, agents_num: int, rounds: int, expected_answer: Optional[str] = None) -> List[List[Dict]]:
        """
        异步处理单个问题的辩论过程，使用 BenchAgent.run_agent_step。
        """
        
        # 1. 初始化每个智能体的上下文：第一轮都是用户提问
        agent_contexts = [[{
            "role": "user", 
            "content": f"Can you solve the following problem? {question} Please explain your reasoning step by step. Make sure to state your final answer clearly at the end of your response.",
            "agent_id": i + 1, 
            "message_type": "user_query", 
            "round": 0 
        }] for i in range(agents_num)]

        # 2. 开始多轮辩论
        for round_idx in range(rounds):
            tasks = []
            
            # 3. 为每个智能体构造输入上下文，并创建生成任务
            for i, current_agent_context in enumerate(agent_contexts):
                current_bench_agent = self.debate_agents[i]

                # 构造传递给 BenchAgent 的 augmented_question (完整的 prompt)
                augmented_question = current_agent_context[-1]['content']
                
                # 构造传递给 BenchAgent 的 additional_args
                additional_args = {
                    "expected_answer": expected_answer, # 用于记忆检索
                    "round": round_idx + 1,
                    # 可以添加其他配置或元数据
                }

                # 从第二轮开始 (round_idx != 0)，需要加入其他智能体上一轮的回答作为辩论提示
                if round_idx != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    # 构造一条“辩论提示”消息：汇总其他智能体的历史观点
                    # 这里的 idx 应该指向上一轮的 AI 回答 (2*round_idx - 1)
                    debate_message = self._construct_message(agent_contexts_other, question, 2 * round_idx - 1)
                    
                    # 将辩论提示（作为 user 消息）追加到当前 BenchAgent 的输入 augmented_question 中
                    # BenchAgent.run_agent_step 接收的是一个完整的 prompt/question
                    augmented_question = f"{current_agent_context[-1]['content']}\n\n--- Debate Input ---\n{debate_message['content']}\n--- End Debate Input ---"

                    # 同时，将 debate_message 追到 context 中，用于逻辑记录
                    current_agent_context.append({
                        **debate_message, 
                        "agent_id": i + 1,
                        "message_type": "debate_query",
                        "round": round_idx + 1
                    })

                # 4. 创建异步任务：让 BenchAgent 运行一个步骤
                # BenchAgent.run_agent_step 运行完整的工具调用和推理，返回最终答案
                task = current_bench_agent.run_agent_step(augmented_question, additional_args)
                tasks.append((i, task))
            
            print(f"第 {round_idx + 1} 轮：并行调用 {len(tasks)} 个 BenchAgent...")
            
            # 5. 并行执行所有 BenchAgent 的运行步骤
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # 6. 处理每个 BenchAgent 的返回结果，并追加到其上下文中
            for (i, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    print(f"BenchAgent {i+1} 出现异常: {result}")
                    content = f"抱歉，BenchAgent {i+1} 出现错误: {str(result)}"
                    final_answer = content
                    usage = None
                else:
                    # BenchAgent.run_agent_step 返回包含 'messages' 和 'final_answer' 的字典
                    final_answer = result.get('final_answer', 'Error: No final answer.')
                    # 将 BenchAgent 的完整运行结果（包括所有工具/步骤消息）包装成一个 AI 响应消息
                    # 我们只需要存储最后一轮的最终输出，以及 BenchAgent 内部的执行步骤
                    # 为了简化，我们只存储 BenchAgent 的最终输出
                    # 也可以将 BenchAgent 返回的 messages 列表整体或部分存储
                    
                    # 这里，我们提取 BenchAgent 的最终答案和它的整个对话历史作为本轮的输出
                    # 为了与 LLMDebate 的上下文结构兼容，我们构造一个 AI 响应消息
                    usage = result.get('messages', [{}])[-1].get('usage_metadata')
                    
                # 构造 assistant 消息（BenchAgent 本轮的最终回答）
                assistant_message = {
                    "role": "assistant",
                    "content": final_answer, # 这里的 content 主要是 BenchAgent 的最终答案
                    "final_answer": final_answer, # 存储 final_answer
                    "agent_id": i + 1,
                    "message_type": "ai_response",
                    "round": round_idx + 1,
                    "usage_metadata": usage,
                    "bench_agent_run_result": result # 存储完整结果，如果需要详细调试信息
                }
                # 将该回答追加到对应智能体的上下文末尾
                agent_contexts[i].append(assistant_message)

        # 7. 返回所有智能体的完整对话历史
        return agent_contexts


    # _construct_message 保持不变，因为它只负责构建辩论提示文本
    def _construct_message(self, agents: List[List[Dict]], question: str, idx: int) -> Dict[str, str]:
        """
        构造包含其他智能体意见的消息
        """
        if len(agents) == 0:
            return {
                "role": "user", 
                "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."
            }

        prefix_string = "These are the recent/updated opinions from other agents: "

        for agent in agents:
            # 提取的是 BenchAgent 运行结果中的 'final_answer'
            if idx < len(agent):
                # 提取 BenchAgent 运行结果中的 final_answer
                agent_response = agent[idx].get("final_answer", agent[idx].get("content", "No Answer Found")) 
                agent_id = agent[idx].get("agent_id", "unknown")
                response = f"\n\n Agent {agent_id} response: ```{agent_response}```"
                prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
        return {
            "role": "user", 
            "content": prefix_string
        }


    # 删除 _construct_assistant_message，不再需要处理原始 API 对象

    # 删除 _generate_answer_async，使用 BenchAgent.run_agent_step 替代

    # 删除 _call_llm，使用 BenchAgent.run_agent_step 替代

    async def _aggregate_answers(self, query: str, answers: List[str]) -> Dict[str, Any]:
        """
        Aggregate all agents' final answers into a single result, using the BenchAgent Aggregator.
        """
        # Build aggregation prompt
        aggregate_instruction = f"Task:\n{query}\n\n"
        
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        
        aggregate_instruction += (
            "Given all the above solutions, reason over them carefully and provide a final answer to the task. "
            f"Make sure to follow these requirements:\n{self.format_prompt}"
        )
        
        # 使用聚合器 BenchAgent 运行一步，生成聚合答案
        additional_args = {
            "is_aggregation": True,
            "debate_answers": answers
        }
        
        # 调用 BenchAgent 的 run_agent_step
        aggregation_result = await self.aggregator_agent.run_agent_step(
            augmented_question=aggregate_instruction, 
            additional_args=additional_args
        )
        
        # run_agent_step 返回包含 'final_answer' 和 'messages' 的字典
        return aggregation_result


# Register the agent system
AgentSystemRegistry.register("llm_debate", LLMDebate, agents_num=2, rounds_num=3)
