# import os
# import logging
# from typing import Dict, Any, List
# from dataclasses import dataclass
# from typing import TypedDict
# from openai import AsyncOpenAI
# from langchain_openai import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage, AIMessage
# from langchain.callbacks.openai_info import OpenAICallbackHandler
# from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# # Load environment variables

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# class AgentResponse(TypedDict):
#     """Structured output for agent responses"""
#     analysis: str  # Problem analysis
#     solution: str  # Solution
#     confidence: int  # Confidence level in the solution, range 1-5

# @dataclass
# class Agent:
#     """Represents an LLM agent"""
#     agent_id: str
#     name: str
#     model_name: str
#     system_prompt: str
#     chat_history: List[Dict[str, str]] = None
#     max_history_length: int = 100  # Maximum length of chat history
    
#     def __post_init__(self):
#         self.chat_history = []
#         self.llm = ChatOpenAI(
#             model=self.model_name,
#             api_key=os.getenv("OPENAI_API_KEY"),
#             base_url=os.getenv("OPENAI_API_BASE"),
#             request_timeout=60,
#             max_retries=2
#         )

#     async def generate_response(self, context: str) -> Dict[str, Any]:
#         """Generate agent response"""
#         messages = [
#             SystemMessage(content=self.system_prompt),
#             *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
#               else AIMessage(content=msg["ai"]) 
#               for msg in self.chat_history[-self.max_history_length:]],
#             HumanMessage(content=context)
#         ]
        
#         try:
#             callback_handler = OpenAICallbackHandler()
#             config = {"callbacks": [callback_handler]}
#             llm_with_schema = self.llm.with_structured_output(schema=AgentResponse, include_raw=True)
#             response = await llm_with_schema.ainvoke(messages, config=config)
            
#             structured_data = response["parsed"]
#             raw_response = response["raw"]
            
#             if isinstance(raw_response, AIMessage):
#                 raw_response.usage_metadata = {
#                     "input_tokens": callback_handler.prompt_tokens,
#                     "output_tokens": callback_handler.completion_tokens,
#                     "total_tokens": callback_handler.total_tokens,
#                     "input_token_details": {},
#                     "output_token_details": {"reasoning": callback_handler.completion_tokens}
#                 }
            
#             if hasattr(structured_data, "dict"):
#                 structured_data = structured_data.dict()
#             elif hasattr(structured_data, "model_dump"):
#                 structured_data = structured_data.model_dump()
            
#             raw_response.name = self.name
            
#             self.chat_history.append({
#                 "role": "human",
#                 "human": context
#             })
#             self.chat_history.append({
#                 "role": "ai",
#                 "ai": raw_response.content
#             })
            
#             return {
#                 "message": raw_response,
#                 "structured_solution": structured_data,
#                 "solution": raw_response.content
#             }
            
#         except Exception as e:
#             logger.error(f"Structured output failed: {str(e)}, falling back to standard output")
            
#             callback_handler = OpenAICallbackHandler()
#             config = {"callbacks": [callback_handler]}
#             response = await self.llm.ainvoke(messages, config=config)
#             response.name = self.name
            
#             if isinstance(response, AIMessage):
#                 response.usage_metadata = {
#                     "input_tokens": callback_handler.prompt_tokens,
#                     "output_tokens": callback_handler.completion_tokens,
#                     "total_tokens": callback_handler.total_tokens,
#                     "input_token_details": {},
#                     "output_token_details": {"reasoning": callback_handler.completion_tokens}
#                 }
            
#             self.chat_history.append({
#                 "role": "human",
#                 "human": context
#             })
#             self.chat_history.append({
#                 "role": "ai",
#                 "ai": response.content
#             })
            
#             return {
#                 "message": response,
#                 "solution": response.content
#             }

# class ResultExtractor:
#     """Extract final results from conversation history"""
#     def __init__(self, model_name: str = None, system_prompt: str = ""):
#         self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
#         self.system_prompt = system_prompt
#         self.llm = ChatOpenAI(
#             model=self.model_name,
#             api_key=os.getenv("OPENAI_API_KEY"),
#             base_url=os.getenv("OPENAI_API_BASE"),
#             request_timeout=60,
#             max_retries=2
#         )
#         self.name = "result_extractor"
        
#     async def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> Dict[str, Any]:
#         """Extract final answer from all agents' conversation histories"""
#         prompt = f"""Original problem: {problem}

# Below are the discussion histories of multiple AI agents:

# {self._format_histories(all_histories)}

# Please analyze the above discussions and provide a final answer. Requirements:
# - Synthesize all agents' viewpoints.
# - Choose the most reasonable solution/option.
# - For HumanEval problems, ensure the response is formatted as:
#   ## Implementation Details
#   [Implementation explanation]
#   ## Features Implemented
#   [List of implemented features]
#   ## Optimizations
#   [List of optimizations or "None"]
#   ## Validated Code
#   ```python
#   [Final validated Python code]
#   ```
# """
#         messages = [
#             SystemMessage(content=self.system_prompt),
#             HumanMessage(content=prompt)
#         ]
        
#         try:
#             callback_handler = OpenAICallbackHandler()
#             config = {"callbacks": [callback_handler]}
#             response = await self.llm.ainvoke(messages, config=config)
#             response.name = "evaluator"
            
#             if isinstance(response, AIMessage):
#                 response.usage_metadata = {
#                     "input_tokens": callback_handler.prompt_tokens,
#                     "output_tokens": callback_handler.completion_tokens,
#                     "total_tokens": callback_handler.total_tokens,
#                     "input_token_details": {},
#                     "output_token_details": {"reasoning": callback_handler.completion_tokens}
#                 }
                
#             return {
#                 "message": response
#             }
#         except Exception as e:
#             logger.error(f"LLM call failed: {str(e)}")
#             return {
#                 "message": None
#             }

#     def _format_histories(self, all_histories: List[List[Dict[str, str]]]) -> str:
#         """Format all conversation histories"""
#         formatted = []
#         agent_names = [f"Agent_{i+1}" for i in range(len(all_histories))]
#         for i, history in enumerate(all_histories):
#             formatted.append(f"\n{agent_names[i]}'s discussion:")
#             for msg in history:
#                 if msg.get("role") == "human":
#                     formatted.append(f"Question: {msg['human']}")
#                 else:
#                     formatted.append(f"Answer: {msg['ai']}")
#         return "\n".join(formatted)

# class Camel(AgentSystem):
#     """
#     LangChain Multi-Agent System

#     This agent system uses multiple agents from the LangChain framework to collaboratively solve problems, including execution and evaluation.
#     """

#     def __init__(self, name: str = "camel", config: Dict[str, Any] = None):
#         """Initialize the LangChain Multi-Agent System"""
#         super().__init__(name, config)
#         self.config = config or {}
        
#         # Extract configuration parameters with default values
#         self.assistant_role_name = self.config.get("assistant_role_name", "Assistant")
#         self.user_role_name = self.config.get("user_role_name", "User")
#         self.critic_role_name = self.config.get("critic_role_name", "Critic")
#         self.output_language = self.config.get("output_language", "English")
#         self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o")
#         self.system_prompt = self.config.get("system_prompt", "") + self.format_prompt

#         # Initialize evaluator and metrics collector through base class methods
#         self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

#         # Initialize agents
#         self._initialize_agents()

#         # Expose LLM instance for tool integration
#         self.llm = ChatOpenAI(
#             model=self.model_name,
#             api_key=os.getenv("OPENAI_API_KEY"),
#             base_url=os.getenv("OPENAI_API_BASE")
#         )
#         self.workers = [
#             {"name": self.assistant_role_name, "llm": self.llm},
#             {"name": self.user_role_name, "llm": self.llm}
#         ]

#     def _initialize_agents(self):
#         """Initialize all agents used in the system with task-specific prompts"""
#         self.assistant_agent = Agent(
#             agent_id="assistant",
#             name=self.assistant_role_name,
#             model_name=self.model_name,
#             system_prompt=f"""You are an {self.assistant_role_name}, a professional problem solver collaborating with the {self.user_role_name} to complete tasks. Your goal is to provide accurate, clear, and detailed responses based on the {self.user_role_name}'s questions or feedback. Adjust your answers according to the {self.user_role_name}'s input to ensure clarity and satisfaction. Use {self.output_language} for all responses.
# {self.system_prompt}"""
#         )
#         self.user_agent = Agent(
#             agent_id="user",
#             name=self.user_role_name,
#             model_name=self.model_name,
#             system_prompt=f"""You are the {self.user_role_name}, responsible for proposing tasks, asking questions, and providing feedback to the {self.assistant_role_name}. Collaborate with the {self.assistant_role_name} to complete tasks. After each response from the {self.assistant_role_name}, evaluate if the answer is satisfactory. If satisfied, include '<Camel_TASK_DONE>' in your response to indicate completion. If not satisfied, provide specific feedback or ask follow-up questions to refine the answer. Use {self.output_language} for all interactions.
# {self.system_prompt}"""
#         )
#         self.critic_agent = Agent(
#             agent_id="critic",
#             name=self.critic_role_name,
#             model_name=self.model_name,
#             system_prompt=f"""You are the {self.critic_role_name}, evaluating task results and selecting the most reasonable answer. Use {self.output_language}.
# {self.system_prompt}"""
#         )
#         self.result_extractor = ResultExtractor(
#             model_name=self.model_name,
#             system_prompt=f"""You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents. Synthesize all agents' viewpoints and choose the most reasonable solution/option, using {self.output_language}.
# {self.system_prompt}"""
#         )
#         logger.info("All agents initialized successfully")

#     async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
#         """
#         Run the multi-agent system on the given problem.
        
#         Args:
#             problem: Dictionary containing problem data and command
            
#         Returns:
#             Dictionary containing the run results with messages and usage metadata
#         """
#         problem_text = problem["problem"]
#         problem_id = problem.get("id", "unknown")

#         # Reinitialize agents with task-specific prompts
#         self._initialize_agents()

#         # Clear chat history for all agents to prevent token accumulation
#         self.assistant_agent.chat_history = []
#         self.user_agent.chat_history = []
#         self.critic_agent.chat_history = []

#         # Execute tasks using appropriate agent
#         execution_result = await self._execute_tasks([problem_text])

#         # Evaluate results using critic agent
#         await self._evaluate_result(execution_result)

#         # Extract final answer
#         all_histories = [
#             self.assistant_agent.chat_history,
#             self.user_agent.chat_history,
#             self.critic_agent.chat_history
#         ]
#         final_result = await self.result_extractor.extract(all_histories, problem_text)
#         final_answer = final_result["message"].content.encode('utf-8').decode('utf-8-sig') if final_result["message"] else "No valid response generated"
           
#         ai_message = {
#             'content': final_answer,
#             'name': self.assistant_role_name,
#             'role': 'assistant',
#             'message_type': 'ai_response',
#             'usage_metadata': final_result["message"].usage_metadata if final_result["message"] else {}
#         }

#         # Record agent responses
#         self._record_agent_responses(problem_id, [ai_message])

#         return {
#             "messages": [ai_message],
#             "final_answer": final_answer
#         }

#     async def _execute_tasks(self, tasks: List[str]) -> str:
#         """Execute tasks using multiple rounds of interaction between user and assistant"""
#         execution_result = ""
#         max_rounds = 10  # Default maximum number of interaction rounds

#         for task in tasks:
#             # Initialize the conversation with the task
#             init_msg = f"""Prompt: {self.system_prompt}. Task: {task}"""
#             current_context = init_msg
#             round_count = 0

#             while round_count < max_rounds:
#                 # User generates a question or feedback based on the current context
#                 user_response = await self.user_agent.generate_response(current_context)
#                 user_answer = user_response["solution"]

#                 # Check if user is satisfied (contains <Camel_TASK_DONE>)
#                 if "<Camel_TASK_DONE>" in user_answer:
#                     # Extract the final answer before <Camel_TASK_DONE>
#                     execution_result += user_answer.split("<Camel_TASK_DONE>")[0].strip() + "\n"
#                     logger.debug(f"Task completed in {round_count + 1} rounds: {execution_result}")
#                     break

#                 # Assistant responds to the user's question or feedback
#                 assistant_response = await self.assistant_agent.generate_response(user_answer)
#                 assistant_answer = assistant_response["solution"]

#                 # Update the context for the next round (user will respond to assistant's answer)
#                 current_context = assistant_answer
#                 round_count += 1

#                 # If maximum rounds reached, use the last assistant's answer
#                 if round_count == max_rounds:
#                     execution_result += assistant_answer + "\n"
#                     logger.debug(f"Max rounds ({max_rounds}) reached, using last answer: {assistant_answer}")

#         return execution_result.strip()

#     async def _evaluate_result(self, execution_result: str) -> str:
#         """Use CriticAgent to evaluate execution results and generate multiple-choice options"""
#         try:
#             result = await self.critic_agent.generate_response(execution_result)
#             return result["solution"]
#         except Exception as e:
#             logger.error(f"Result evaluation failed: {str(e)}")
#             return execution_result  # Fallback to execution result

# # Register the agent system
# AgentSystemRegistry.register("camel", Camel)