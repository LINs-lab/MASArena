# """
# LLM Debate Multi-Agent System

# This module implements a multi-agent debate system where multiple LLM agents
# engage in multi-round discussions to collaboratively solve problems through debate.
# """

# import os
# from typing import Dict, Any, List
# import contextlib
# import asyncio
# from openai import AsyncOpenAI
# from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# # Load environment variables

# class LLMDebate(AgentSystem):
#     """
#     LLM Debate Multi-Agent System
    
#     This system implements a multi-round debate mechanism where multiple LLM agents
#     discuss and refine their answers through iterative rounds of debate.
#     """

#     def __init__(self, name: str = "llm_debate", config: Dict[str, Any] = None):
#         """Initialize the LLM Debate System"""
#         super().__init__(name, config)
#         self.config = config or {}
        
#         # Configuration parameters
#         self.agents_num = self.config.get("agents_num", 2)  # Number of debate agents
#         self.rounds_num = self.config.get("rounds_num", 3)  # Number of debate rounds
#         self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
#         # System prompt with format requirements
#         self.base_system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")
#         self.system_prompt = self.base_system_prompt + "\n" + self.format_prompt
        
#         # Initialize OpenAI client
#         self.client = AsyncOpenAI(
#             api_key=os.getenv("OPENAI_API_KEY"), 
#             base_url=os.getenv("OPENAI_API_BASE"),
#             timeout=40
#         )
        
#         self.bench_agent_params = {
#             "model": self.model_name,
#             "manager_tools": self.config.get("manager_tools"),
#             "search_tools": self.config.get("search_tools"),
#             "memory": self.config.get("memory"),
#             "api_key": self.config.get("api_key"),
#             "api_base": self.config.get("api_base"),
#             "max_steps": self.config.get("max_steps", 15),
#             "search_max_steps": self.config.get("search_max_steps", 10),
#             "verbosity_level": self.config.get("verbosity_level", 1),
#             # 如果未显式提供 additional_instructions，则自动落到 format_prompt（如 math 的 \boxed{} 约束）
#             "additional_instructions": self.config.get("additional_instructions")
#         }

#     async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
#         """
#         Run the LLM Debate system on a given problem.
        
#         Args:
#             problem: Dictionary containing the problem data
            
#         Returns:
#             Dictionary of run results including all agent messages and final answer
#         """
#         # 1. 从输入中提取问题文本
#         query = problem["problem"]
        
#         # 2. 启动异步多智能体辩论流程
#         #    返回每个智能体在整个辩论过程中的完整对话上下文（消息列表）
#         agent_contexts = await self._process_single_question_async(query, self.agents_num, self.rounds_num)
        
#         # 3. 初始化用于记录结果的数据结构
#         all_messages = []
#         final_answers = []
        
#         # 4. 遍历每一轮、每个智能体，提取其在该轮的回答
#         for round_idx in range(self.rounds_num):
#             for agent_idx, agent_context in enumerate(agent_contexts):
#                 # 根据 gen_math 的逻辑：每轮智能体的回答位于索引 2 * round_idx + 1
#                 # （因为上下文格式为 [user_msg, ai_msg, user_msg, ai_msg, ...]）
#                 response_index = 2 * round_idx + 1  # Based on gen_math logic
#                 # 确保索引不越界
#                 if response_index < len(agent_context):
#                     response_msg = agent_context[response_index]
#                     # 构造标准化的 AI 消息对象
#                     ai_message = {
#                         'content': response_msg['content'],# 回答内容
#                         'name': f'debate_agent_{agent_idx+1}',# 智能体名称（如 debate_agent_1）
#                         'role': 'assistant',# 角色为助手
#                         'message_type': 'ai_response',# 消息类型：智能体回答
#                         'round': round_idx + 1,# 当前轮次（从1开始）
#                         'agent_id': f'debate_agent_{agent_idx+1}',# 智能体唯一ID
#                         'usage_metadata': None  # gen_math 不记录 token 使用情况
#                     }
#                     # （可选）打印调试信息
#                     print(f"agent_name: {ai_message['name']}")
#                     # 将该消息加入全局消息列表
#                     all_messages.append(ai_message)
        
#         # 5. 提取每个智能体的最终答案（假设最后一次消息是其最终结论）
#         for agent_context in agent_contexts:
#             if len(agent_context) >= 2:
#                 final_answers.append(agent_context[-1]['content'])
        
#         # 6. 调用聚合器，综合所有智能体的答案生成最终统一答案
#         aggregated_answer = await self._aggregate_answers(query, final_answers)
        
#          # 7. 构造聚合器的消息对象，并加入消息列表
#         aggregation_message = {
#             'content': aggregated_answer['content'],
#             'name': 'debate_aggregator',
#             'role': 'assistant',
#             'message_type': 'aggregation',
#             'usage_metadata': aggregated_answer['usage']
#         }
#         all_messages.append(aggregation_message)
        
#         return {
#             "messages": all_messages,
#             "final_answer": aggregated_answer['content'],
#             "agent_responses": final_answers,
#             "rounds_completed": self.rounds_num,
#             "agents_participated": self.agents_num
#         }

#     async def _process_single_question_async(self, question: str, agents: int, rounds: int) -> List[List[Dict]]:
#         """
#         异步处理单个问题的辩论过程
#         """
#         # 1. 初始化每个智能体的上下文：第一轮都是用户提问
#     #    每个智能体看到相同的初始问题，但拥有独立的对话历史
#         agent_contexts = [[{
#             "role": "user", 
#             "content": """Can you solve the following problem? {} Please explain your reasoning step by step. Make sure to state your final answer clearly at the end of your response.""".format(question),
#             "agent_id": i + 1,  # # 标识这是第几个智能体的上下文（从1开始）
#             "message_type": "user_query",  # 消息类型：初始用户提问
#             "round": 0  # 初始轮次设为0（尚未开始辩论）
#         }] for i in range(agents)]# 为每个智能体创建一个独立的上下文列表

#         # 2. 开始多轮辩论（从第1轮到第rounds轮）
#         for round in range(rounds):
#             # 用于收集本轮所有智能体的异步生成任务
#             tasks = []
#             # 3. 为每个智能体构造输入上下文，并创建生成任务
#             for i, agent_context in enumerate(agent_contexts):
#                 # 从第二轮开始（round != 0），需要加入其他智能体上一轮的回答作为参考
#                 if round != 0:
#                     # 获取除当前智能体外的其他所有智能体上下文
#                     agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
#                     # 构造一条“辩论提示”消息：汇总其他智能体的历史观点
#                     # 注意：这里传入 2*round - 1 是为了提取到上一轮结束时的所有有效 AI 回答
#                     message = self._construct_message(agent_contexts_other, question, 2*round - 1)
#                     # 添加元数据到message
#                     message.update({
#                         "agent_id": i + 1,
#                         "message_type": "debate_query
#                         "round": round + 1
#                     })
#                     # 将这条“辩论提示”追加到当前智能体的上下文中（作为 user 消息）
#                     agent_context.append(message)

#                 # 4. 创建异步任务：让当前智能体基于其完整上下文生成回答
#             #    _generate_answer_async 会调用 LLM API 并返回响应对象
#                 task = self._generate_answer_async(agent_context)
#                 tasks.append((i, task))
            
#              # 5. 打印日志：提示即将并行调用多少个智能体
#             print(f"第 {round + 1} 轮：并行调用 {len(tasks)} 个智能体...")
            
#             # 6. 使用 asyncio.gather 并行执行所有智能体的 LLM 调用（真正并发！）
#         #    return_exceptions=True 确保某个智能体出错不会中断整个流程
#             task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
#             # 7. 处理每个智能体的返回结果，并追加到其上下文中
#             for (i, _), result in zip(tasks, task_results):
#             # 处理异常情况（如网络错误、API 限流等）
#                 if isinstance(result, Exception):
#                     print(f"智能体 {i} 出现异常: {result}")
#                     content = f"抱歉，智能体 {i} 出现错误: {str(result)}"
#                     usage = None
#                 else:
#                 # 正常情况下，从 LLM 响应中提取回答内容和 token 使用情况
#                     content = result.choices[0].message.content if hasattr(result, 'choices') else "无法获取回答内容"
#                     usage = getattr(result, 'usage', None)
#                 # 构造 assistant 消息（智能体本轮的回答）
#                 assistant_message = {
#                     "role": "assistant",
#                     "content": content,
#                     "agent_id": i + 1,
#                     "message_type": "ai_response",
#                     "round": round + 1,# 当前轮次
#                     "usage_metadata": usage# token 使用统计（可用于成本分析）
#                 }
#                 # 将该回答追加到对应智能体的上下文末尾
#                 agent_contexts[i].append(assistant_message)
#         # 8. 返回所有智能体的完整对话历史
#         return agent_contexts

#     def _construct_message(self, agents: List[List[Dict]], question: str, idx: int) -> Dict[str, str]:
#         """
#         构造包含其他智能体意见的消息
#         """
#         # Use introspection in the case in which there are no other agents.
#         if len(agents) == 0:
#             return {
#                 "role": "user", 
#                 "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."
#             }

#         prefix_string = "These are the recent/updated opinions from other agents: "

#         for agent in agents:
#             if idx < len(agent):
#                 agent_response = agent[idx]["content"]
#                 agent_id = agent[idx].get("agent_id", "unknown")
#                 response = f"\n\n Agent {agent_id} response: ```{agent_response}```"
#                 prefix_string = prefix_string + response

#         prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response."
#         return {
#             "role": "user", 
#             "content": prefix_string
#         }

#     def _construct_assistant_message(self, completion) -> Dict[str, str]:
#         """
#         构造assistant消息
#         """
#         # 检查 completion 是否为字符串（错误情况）
#         if isinstance(completion, str):
#             print(f"警告：收到字符串响应而非API对象: {completion[:100]}...")
#             return {"role": "assistant", "content": "API返回了错误的响应格式"}
        
#         # 检查是否有 choices 属性
#         if hasattr(completion, 'choices') and len(completion.choices) > 0:
#             content = completion.choices[0].message.content
#             return {"role": "assistant", "content": content}
#         else:
#             # 备用方案
#             return {"role": "assistant", "content": "抱歉，无法获取回答内容。"}

#     async def _generate_answer_async(self, answer_context: List[Dict]) -> Any:
#         """
#         异步版本的API调用函数
#         """
#         # 添加系统提示到消息开头，移除元数据字段
#         api_messages = [{"role": "system", "content": self.system_prompt}]
#         for msg in answer_context:
#             api_messages.append({
#                 "role": msg["role"],
#                 "content": msg["content"]
#             })
        
#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 completion = await self.client.chat.completions.create(
#                           model=self.model_name,
#                           messages=api_messages,
#                           n=1
#                           )
#                 return completion
#             except Exception as e:
#                 print(f"API调用出错，正在重试... (尝试 {attempt + 1}/{max_retries})")
#                 print(f"错误类型: {type(e).__name__}")
#                 print(f"错误详情: {str(e)}")
#                 if attempt < max_retries - 1:
#                     print("等待5秒后重试...")  # 减少等待时间
#                     await asyncio.sleep(5)
#                 else:
#                     print("已达到最大重试次数，跳过此请求")
#                     # 返回一个模拟的 completion 对象以避免程序崩溃
#                     class MockCompletion:
#                         def __init__(self):
#                             self.choices = [type('obj', (object,), {
#                                 'message': type('obj', (object,), {
#                                     'content': "抱歉，由于API错误无法生成回答。"
#                                 })()
#                             })()]
#                     return MockCompletion()
        
#         return None

#     async def _call_llm(self, messages: List[Dict]) -> Dict[str, Any]:
#         """
#         Call the LLM with given messages and return response with usage metadata.
#         """
#         # 使用新的异步生成答案方法
#         completion = await self._generate_answer_async(messages)
        
#         # 提取内容和使用信息
#         if hasattr(completion, 'choices') and len(completion.choices) > 0:
#             content = completion.choices[0].message.content
#             content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
#             with contextlib.suppress(UnicodeDecodeError):
#                 content = content.encode('utf-8').decode('utf-8-sig')  # Remove BOM
            
#             usage = getattr(completion, 'usage', None)
            
#             return {
#                 'content': content,
#                 'usage': usage
#             }
#         else:
#             return {
#                 'content': "抱歉，无法获取回答内容。",
#                 'usage': None
#             }

#     async def _aggregate_answers(self, query: str, answers: List[str]) -> Dict[str, Any]:
#         """
#         Aggregate all agents' final answers into a single result.
        
#         Args:
#             query: Original query/problem
#             answers: List of final answers from all agents
            
#         Returns:
#             Dictionary containing aggregated answer and usage metadata
#         """
#         # Build aggregation prompt
#         aggregate_instruction = f"Task:\n{query}\n\n"
        
#         for i, answer in enumerate(answers):
#             aggregate_instruction += f"Solution {i+1}:\n{answer}\n\n"
        
#         aggregate_instruction += (
#             "Given all the above solutions, reason over them carefully and provide a final answer to the task. "
#             f"Make sure to follow these requirements:\n{self.format_prompt}"
#         )
        
#         # Call LLM for aggregation
#         messages = [{"role": "user", "content": aggregate_instruction}]
#         return await self._call_llm(messages)


# # Register the agent system
# AgentSystemRegistry.register("llm_debate", LLMDebate, 
#                            agents_num=2, rounds_num=3) 