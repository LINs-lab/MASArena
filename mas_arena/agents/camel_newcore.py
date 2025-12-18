import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import TypedDict
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.openai_info import OpenAICallbackHandler
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.agents.agent_core import Tool

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AgentResponse(TypedDict):
    """Structured output for agent responses"""
    analysis: str  # Problem analysis
    solution: str  # Solution
    confidence: int  # Confidence level in the solution, range 1-5

@dataclass
class EnhancedAgent:
    """Represents an enhanced LLM agent using BenchAgent for execution"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str

    bench_agent_instance: BenchAgent 
    chat_history: List[Dict[str, str]] = None
    max_history_length: int = 100  # Maximum length of chat history
    
    def __post_init__(self):
        self.chat_history = []
       
    async def generate_response(self, context: str, additional_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate agent response using BenchAgent's run_agent_step.
        
        Note: The original 'Agent' used structured output (AgentResponse schema) 
        and LangChain's ChatOpenAI. BenchAgent's `run_agent_step` handles its 
        own multi-step execution and final answer extraction, thus bypassing 
        the need for the local structured output logic. We simplify to directly 
        call `run_agent_step` with the system prompt and context.
        """
        additional_args = additional_args or {}


        full_prompt_messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history[-self.max_history_length:]],
            HumanMessage(content=context)
        ]

        
        history_str = "\n".join([
            f"{'Human' if msg.get('role') == 'human' else 'Assistant'}: {msg.get('human') or msg.get('ai')}" 
            for msg in self.chat_history[-self.max_history_length:]
        ])

        augmented_question = (
            f"System Instruction: {self.system_prompt}\n\n"
            f"Conversation History:\n{history_str}\n\n"
            f"Current Input: {context}"
        )

        try:

            bench_result = await self.bench_agent_instance.run_agent_step(
                augmented_question=augmented_question,
                additional_args=additional_args
            )
            
            final_answer = bench_result.get("final_answer", "BenchAgent did not return a final answer.")
            

            ai_message_from_bench = next(
                (msg for msg in bench_result.get("messages", []) if msg.get("role") == "assistant"),
                {'content': final_answer, 'usage_metadata': {}}
            )

            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": final_answer
            })
            

            class MockAIMessage:
                def __init__(self, content, name, usage_metadata):
                    self.content = content
                    self.name = name
                    self.usage_metadata = usage_metadata

            mock_ai_message = MockAIMessage(
                content=final_answer, 
                name=self.name, 
                usage_metadata=ai_message_from_bench.get("usage_metadata", {})
            )
            

            structured_data_placeholder = {
                "analysis": "Analysis from BenchAgent's process.",
                "solution": final_answer,
                "confidence": 5 
            }

            return {
                "message": mock_ai_message,
                "structured_solution": structured_data_placeholder,
                "solution": final_answer
            }
            
        except Exception as e:
            logger.error(f"BenchAgent run_agent_step failed for {self.name}: {str(e)}")
            return {
                "message": None,
                "solution": f"Error running BenchAgent: {str(e)}"
            }

class ResultExtractor:
    """Extract final results from conversation history"""
    def __init__(self, model_name: str = None, system_prompt: str = ""):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o")
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            request_timeout=60,
            max_retries=2
        )
        self.name = "result_extractor"
        
    async def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> Dict[str, Any]:
        """Extract final answer from all agents' conversation histories"""
        prompt = f"""Original problem: {problem}

Below are the discussion histories of multiple AI agents:

{self._format_histories(all_histories)}

Please analyze the above discussions and provide a final answer. Requirements:
- Synthesize all agents' viewpoints.
- Choose the most reasonable solution/option.
- For HumanEval problems, ensure the response is formatted as:
  ## Implementation Details
  [Implementation explanation]
  ## Features Implemented
  [List of implemented features]
  ## Optimizations
  [List of optimizations or "None"]
  ## Validated Code
  ```python
  [Final validated Python code]
  ```
"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            callback_handler = OpenAICallbackHandler()
            config = {"callbacks": [callback_handler]}
            response = await self.llm.ainvoke(messages, config=config)
            response.name = "evaluator"
            
            if isinstance(response, AIMessage):
                response.usage_metadata = {
                    "input_tokens": callback_handler.prompt_tokens,
                    "output_tokens": callback_handler.completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                    "input_token_details": {},
                    "output_token_details": {"reasoning": callback_handler.completion_tokens}
                }
                
            return {
                "message": response
            }
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return {
                "message": None
            }

    def _format_histories(self, all_histories: List[List[Dict[str, str]]]) -> str:
        """Format all conversation histories"""
        formatted = []
        agent_names = [f"Agent_{i+1}" for i in range(len(all_histories))]
        for i, history in enumerate(all_histories):
            formatted.append(f"\n{agent_names[i]}'s discussion:")
            for msg in history:
                if msg.get("role") == "human":
                    formatted.append(f"Question: {msg['human']}")
                else:
                    formatted.append(f"Answer: {msg['ai']}")
        return "\n".join(formatted)

class Camel(AgentSystem):
    """
    LangChain Multi-Agent System

    This agent system uses multiple agents from the LangChain framework to collaboratively solve problems, including execution and evaluation.
    """

    def __init__(self, name: str = "camel", config: Dict[str, Any] = None):
        """Initialize the Enhanced Multi-Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        
        # Extract configuration parameters with default values
        self.assistant_role_name = self.config.get("assistant_role_name", "Assistant")
        self.user_role_name = self.config.get("user_role_name", "User")
        self.critic_role_name = self.config.get("critic_role_name", "Critic")
        self.output_language = self.config.get("output_language", "English")
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o")
        self.system_prompt = self.config.get("system_prompt", "") + self.format_prompt


        bench_agent_config = {
            "model": self.model_name,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 2),
            "additional_instructions": self.config.get("additional_instructions"),
            "name": "camel_bench_agent",
            "manager_tools": self.config.get("manager_tools"),
            "search_tools": self.config.get("search_tools"),
        }
        

        self.bench_agent_instance = BenchAgent(**bench_agent_config) 

        self._initialize_agents()

        self.workers = []

    def _initialize_agents(self):
        """Initialize all agents used in the system with task-specific prompts"""
        self.assistant_agent = EnhancedAgent(
            agent_id="assistant",
            name=self.assistant_role_name,
            model_name=self.model_name,
            bench_agent_instance=self.bench_agent_instance,
            system_prompt=f"""You are an {self.assistant_role_name}, a professional problem solver collaborating with the {self.user_role_name} to complete tasks. Your goal is to provide accurate, clear, and detailed responses based on the {self.user_role_name}'s questions or feedback. Adjust your answers according to the {self.user_role_name}'s input to ensure clarity and satisfaction. Use {self.output_language} for all responses.
{self.system_prompt}"""
        )
        self.user_agent = EnhancedAgent(
            agent_id="user",
            name=self.user_role_name,
            model_name=self.model_name,
            bench_agent_instance=self.bench_agent_instance,
            system_prompt=f"""You are the {self.user_role_name}, responsible for proposing tasks, asking questions, and providing feedback to the {self.assistant_role_name}. Collaborate with the {self.assistant_role_name} to complete tasks. After each response from the {self.assistant_role_name}, evaluate if the answer is satisfactory. If satisfied, include '<Camel_TASK_DONE>' in your response to indicate completion. If not satisfied, provide specific feedback or ask follow-up questions to refine the answer. Use {self.output_language} for all interactions.
{self.system_prompt}"""
        )
        self.critic_agent = EnhancedAgent(
            agent_id="critic",
            name=self.critic_role_name,
            model_name=self.model_name,
            bench_agent_instance=self.bench_agent_instance,
            system_prompt=f"""You are the {self.critic_role_name}, evaluating task results and selecting the most reasonable answer. Use {self.output_language}.
{self.system_prompt}"""
        )
        self.result_extractor = ResultExtractor(
            model_name=self.model_name,
            system_prompt=f"""You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents. Synthesize all agents' viewpoints and choose the most reasonable solution/option, using {self.output_language}.
{self.system_prompt}"""
        )
        logger.info("All agents initialized successfully")

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the multi-agent system on the given problem.
        
        Args:
            problem: Dictionary containing problem data and command
            
        Returns:
            Dictionary containing the run results with messages and usage metadata
        """
        problem_text = problem["problem"]
        problem_id = problem.get("id", "unknown")

        # Reinitialize agents with task-specific prompts
        self._initialize_agents()

        # Clear chat history for all agents to prevent token accumulation
        self.assistant_agent.chat_history = []
        self.user_agent.chat_history = []
        self.critic_agent.chat_history = []

        # Execute tasks using appropriate agent
        execution_result = await self._execute_tasks([problem_text])

        # Evaluate results using critic agent
        await self._evaluate_result(execution_result)

        # Extract final answer
        all_histories = [
            self.assistant_agent.chat_history,
            self.user_agent.chat_history,
            self.critic_agent.chat_history
        ]
        final_result = await self.result_extractor.extract(all_histories, problem_text)
        final_answer = final_result["message"].content.encode('utf-8').decode('utf-8-sig') if final_result["message"] else "No valid response generated"
           
        ai_message = {
            'content': final_answer,
            'name': self.assistant_role_name,
            'role': 'assistant',
            'message_type': 'ai_response',
            'usage_metadata': final_result["message"].usage_metadata if final_result["message"] else {}
        }

        # Record agent responses
        self._record_agent_responses(problem_id, [ai_message])

        return {
            "messages": [ai_message],
            "final_answer": final_answer
        }

    async def _execute_tasks(self, tasks: List[str]) -> str:
        """Execute tasks using multiple rounds of interaction between user and assistant"""
        execution_result = ""
        max_rounds = 10  # Default maximum number of interaction rounds

        for task in tasks:
            # Initialize the conversation with the task
            init_msg = f"""Prompt: {self.system_prompt}. Task: {task}"""
            current_context = init_msg
            round_count = 0

            while round_count < max_rounds:
                # User generates a question or feedback based on the current context
                user_response = await self.user_agent.generate_response(current_context)
                user_answer = user_response["solution"]

                # Check if user is satisfied (contains <Camel_TASK_DONE>)
                if "<Camel_TASK_DONE>" in user_answer:
                    # Extract the final answer before <Camel_TASK_DONE>
                    execution_result += user_answer.split("<Camel_TASK_DONE>")[0].strip() + "\n"
                    logger.debug(f"Task completed in {round_count + 1} rounds: {execution_result}")
                    break

                # Assistant responds to the user's question or feedback
                assistant_response = await self.assistant_agent.generate_response(user_answer)
                assistant_answer = assistant_response["solution"]

                # Update the context for the next round (user will respond to assistant's answer)
                current_context = assistant_answer
                round_count += 1

                # If maximum rounds reached, use the last assistant's answer
                if round_count == max_rounds:
                    execution_result += assistant_answer + "\n"
                    logger.debug(f"Max rounds ({max_rounds}) reached, using last answer: {assistant_answer}")

        return execution_result.strip()

    async def _evaluate_result(self, execution_result: str) -> str:
        """Use CriticAgent to evaluate execution results and generate multiple-choice options"""
        try:
            # 这里的 generate_response 调用了 BenchAgent.run_agent_step
            result = await self.critic_agent.generate_response(execution_result)
            return result["solution"]
        except Exception as e:
            logger.error(f"Result evaluation failed: {str(e)}")
            return execution_result

# Register the agent system
AgentSystemRegistry.register("camel", Camel)