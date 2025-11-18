"""
LLM Debate Multi-Agent System

This module implements a multi-agent debate system where multiple LLM agents
engage in multi-round discussions to collaboratively solve problems through debate.
"""

import os
from typing import Dict, Any, List
import contextlib
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv(override=True)

class LLMDebate(AgentSystem):
    """
    LLM Debate Multi-Agent System
    
    This system implements a multi-round debate mechanism where multiple LLM agents
    discuss and refine their answers through iterative rounds of debate.
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
        self.base_system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
        self.system_prompt = self.base_system_prompt + "\n" + self.format_prompt
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            base_url=os.getenv("OPENAI_API_BASE"),
            timeout=40
        )

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the LLM Debate system on a given problem.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including all agent messages and final answer
        """
        query = problem["problem"]
        
        # Use the gen_math logic for processing
        agent_contexts = await self._process_single_question_async(query, self.agents_num, self.rounds_num)
        
        # Convert to evaluation framework format
        all_messages = []
        final_answers = []
        
        # Extract messages from agent contexts
        for round_idx in range(self.rounds_num):
            for agent_idx, agent_context in enumerate(agent_contexts):
                # Get the assistant response for this round
                response_index = 2 * round_idx + 1  # Based on gen_math logic
                if response_index < len(agent_context):
                    response_msg = agent_context[response_index]
                    
                    ai_message = {
                        'content': response_msg['content'],
                        'name': f'debate_agent_{agent_idx+1}',
                        'role': 'assistant',
                        'message_type': 'ai_response',
                        'round': round_idx + 1,
                        'agent_id': f'debate_agent_{agent_idx+1}',
                        'usage_metadata': None  # gen_math doesn't track usage
                    }
                    print(f"agent_name: {ai_message['name']}")
                    all_messages.append(ai_message)
        
        # Extract final answers from each agent
        for agent_context in agent_contexts:
            if len(agent_context) >= 2:
                final_answers.append(agent_context[-1]['content'])
        
        # Aggregate all answers into final result
        aggregated_answer = await self._aggregate_answers(query, final_answers)
        
        # Create aggregation message
        aggregation_message = {
            'content': aggregated_answer['content'],
            'name': 'debate_aggregator',
            'role': 'assistant',
            'message_type': 'aggregation',
            'usage_metadata': aggregated_answer['usage']
        }
        all_messages.append(aggregation_message)
        
        return {
            "messages": all_messages,
            "final_answer": aggregated_answer['content'],
            "agent_responses": final_answers,
            "rounds_completed": self.rounds_num,
            "agents_participated": self.agents_num
        }

    async def _process_single_question_async(self, question: str, agents: int, rounds: int) -> List[List[Dict]]:
        """
        异步处理单个问题的辩论过程
        """
        # 为每个智能体创建初始上下文
        agent_contexts = [[{
            "role": "user", 
            "content": """Can you solve the following problem? {} Please explain your reasoning step by step. Make sure to state your final answer clearly at the end of your response.""".format(question),
            "agent_id": i + 1,  # 添加agent_id
            "message_type": "user_query",  # 添加消息类型
            "round": 0  # 添加轮次信息
        }] for i in range(agents)]

        # 多轮辩论
        for round in range(rounds):
            # 收集所有智能体的任务
            tasks = []
            
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = self._construct_message(agent_contexts_other, question, 2*round - 1)
                    # 添加元数据到message
                    message.update({
                        "agent_id": i + 1,
                        "message_type": "debate_query",
                        "round": round + 1
                    })
                    agent_context.append(message)

                # 创建异步任务
                task = self._generate_answer_async(agent_context)
                tasks.append((i, task))
            
            # 并行执行所有智能体的API调用
            print(f"第 {round + 1} 轮：并行调用 {len(tasks)} 个智能体...")
            
            # 使用asyncio.gather真正并行执行所有任务
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # 处理结果
            for (i, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    print(f"智能体 {i} 出现异常: {result}")
                    content = f"抱歉，智能体 {i} 出现错误: {str(result)}"
                    usage = None
                else:
                    content = result.choices[0].message.content if hasattr(result, 'choices') else "无法获取回答内容"
                    usage = getattr(result, 'usage', None)
                
                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "agent_id": i + 1,
                    "message_type": "ai_response",
                    "round": round + 1,
                    "usage_metadata": usage
                }
                agent_contexts[i].append(assistant_message)

        return agent_contexts

    def _construct_message(self, agents: List[List[Dict]], question: str, idx: int) -> Dict[str, str]:
        """
        构造包含其他智能体意见的消息
        """
        # Use introspection in the case in which there are no other agents.
        if len(agents) == 0:
            return {
                "role": "user", 
                "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."
            }

        prefix_string = "These are the recent/updated opinions from other agents: "

        for agent in agents:
            if idx < len(agent):
                agent_response = agent[idx]["content"]
                agent_id = agent[idx].get("agent_id", "unknown")
                response = f"\n\n Agent {agent_id} response: ```{agent_response}```"
                prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
        return {
            "role": "user", 
            "content": prefix_string
        }

    def _construct_assistant_message(self, completion) -> Dict[str, str]:
        """
        构造assistant消息
        """
        # 检查 completion 是否为字符串（错误情况）
        if isinstance(completion, str):
            print(f"警告：收到字符串响应而非API对象: {completion[:100]}...")
            return {"role": "assistant", "content": "API返回了错误的响应格式"}
        
        # 检查是否有 choices 属性
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            return {"role": "assistant", "content": content}
        else:
            # 备用方案
            return {"role": "assistant", "content": "抱歉，无法获取回答内容。"}

    async def _generate_answer_async(self, answer_context: List[Dict]) -> Any:
        """
        异步版本的API调用函数
        """
        # 添加系统提示到消息开头，移除元数据字段
        api_messages = [{"role": "system", "content": self.system_prompt}]
        for msg in answer_context:
            api_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                completion = await self.client.chat.completions.create(
                          model=self.model_name,
                          messages=api_messages,
                          n=1
                          )
                return completion
            except Exception as e:
                print(f"API调用出错，正在重试... (尝试 {attempt + 1}/{max_retries})")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误详情: {str(e)}")
                if attempt < max_retries - 1:
                    print("等待5秒后重试...")  # 减少等待时间
                    await asyncio.sleep(5)
                else:
                    print("已达到最大重试次数，跳过此请求")
                    # 返回一个模拟的 completion 对象以避免程序崩溃
                    class MockCompletion:
                        def __init__(self):
                            self.choices = [type('obj', (object,), {
                                'message': type('obj', (object,), {
                                    'content': "抱歉，由于API错误无法生成回答。"
                                })()
                            })()]
                    return MockCompletion()
        
        return None

    async def _call_llm(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Call the LLM with given messages and return response with usage metadata.
        """
        # 使用新的异步生成答案方法
        completion = await self._generate_answer_async(messages)
        
        # 提取内容和使用信息
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            content = completion.choices[0].message.content
            content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
            with contextlib.suppress(UnicodeDecodeError):
                content = content.encode('utf-8').decode('utf-8-sig')  # Remove BOM
            
            usage = getattr(completion, 'usage', None)
            
            return {
                'content': content,
                'usage': usage
            }
        else:
            return {
                'content': "抱歉，无法获取回答内容。",
                'usage': None
            }

    async def _aggregate_answers(self, query: str, answers: List[str]) -> Dict[str, Any]:
        """
        Aggregate all agents' final answers into a single result.
        
        Args:
            query: Original query/problem
            answers: List of final answers from all agents
            
        Returns:
            Dictionary containing aggregated answer and usage metadata
        """
        # Build aggregation prompt
        aggregate_instruction = f"Task:\n{query}\n\n"
        
        for i, answer in enumerate(answers):
            aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        
        aggregate_instruction += (
            "Given all the above solutions, reason over them carefully and provide a final answer to the task. "
            f"Make sure to follow these requirements:\n{self.format_prompt}"
        )
        
        # Call LLM for aggregation
        messages = [{"role": "user", "content": aggregate_instruction}]
        return await self._call_llm(messages)


# Register the agent system
AgentSystemRegistry.register("llm_debate", LLMDebate, 
                           agents_num=2, rounds_num=3) 