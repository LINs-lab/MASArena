import os
from typing import Dict, Any, List
import contextlib
# from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()

class AutoGen(AgentSystem):
    """
    AutoGen System

    This agent system implements a collaborative AutoGen framework similar to AutoGen,
    where AutoGen agents with distinct roles interact to solve problems through conversation.
    """
    def __init__(self, name: str = "autogen", config: Dict[str, Any] = None):
        """Initialize the AutoGen System"""
        super().__init__(name, config)
        self.config = config or {}
        
        # Default model and agent configurations
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")
        self.system_prompt = self.config.get("system_prompt", "") + self.format_prompt
        
        # Initialize OpenAI client
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))
        # Define AutoGen agents with distinct roles
        self.agents = self.config.get("agents", [
            {
                "name": "planner",
                "role": "Creates a step-by-step plan to solve the problem.",
                "system_prompt": "You are a Planner Agent. Your role is to analyze the problem and create a clear, step-by-step plan to solve it. Provide concise instructions for other agents to follow."
            },
            {
                "name": "executor",
                "role": "Executes the plan and provides the final solution.",
                "system_prompt": "You are an Executor Agent. Your role is to follow the plan provided by the Planner Agent and produce a detailed solution to the problem."
            },
            {
                "name": "reviewer",
                "role": "Reviews the solution and suggests improvements.",
                "system_prompt": "You are a Reviewer Agent. Your role is to evaluate the solution provided by the Executor Agent, identify any issues, and suggest improvements or confirm the solution is correct."
            }
        ])
        
        # Maximum number of conversation rounds to prevent infinite loops
        self.max_rounds = self.config.get("max_rounds", 5)

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the AutoGen system on a given problem.

        Agents collaborate in a conversational loop, with each agent contributing based on its role.

        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem_text}"}
        ]
        conversation_history = messages.copy()
        all_messages = []

        # Conversation loop
        for round_num in range(self.max_rounds):
            for agent in self.agents:
                agent_name = agent["name"]
                agent_prompt = agent["system_prompt"]
                
                # Prepare messages for the current agent, including conversation history
                agent_messages = [
                    {"role": "system", "content": agent_prompt},
                    *conversation_history
                ]
                
                # Get response from the current agent
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=agent_messages
                )
                
                # Extract and clean response content
                response_content = response.choices[0].message.content
                response_content = response_content.replace('\r\n', '\n').replace('\r', '\n').strip()
                with contextlib.suppress(UnicodeDecodeError):
                    response_content = response_content.encode('utf-8').decode('utf-8-sig')

                # Create message object with usage metadata
                ai_message = {
                    'content': response_content,
                    'name': agent_name,
                    'role': 'assistant',
                    'message_type': 'ai_response',
                    'usage_metadata': response.usage
                }
                
                # Add to conversation history and all messages
                conversation_history.append({"role": "assistant", "content": response_content, "name": agent_name})
                all_messages.append(ai_message)
                
                # Check if the reviewer has approved the solution (simplified termination condition)
                if agent_name == "reviewer" and "Solution approved" in response_content.lower():
                    return {
                        "messages": all_messages,
                        "final_answer": all_messages[-2]["content"]  # Executor's solution
                    }
        
        # If max rounds reached, return the last executor's solution
        final_answer = next(
            (msg["content"] for msg in reversed(all_messages) if msg["name"] == "executor"),
            "No solution found within maximum rounds."
        )
        
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

# Register the agent system
AgentSystemRegistry.register("autogen", AutoGen)