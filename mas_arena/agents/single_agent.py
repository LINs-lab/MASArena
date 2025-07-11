"""
Single Agent System

This module implements a simple single-agent system that uses a single LLM
to solve problems directly.
"""

import os
from typing import Dict, Any
import contextlib

from openai import AsyncOpenAI
from dotenv import load_dotenv

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SingleAgent(AgentSystem):
    """
    Single Agent System

    This agent system uses a single LLM to solve problems directly.
    """

    def __init__(self, name: str = "single_agent", config: Dict[str, Any] = None):
        """Initialize the Single Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")  # Use qwen-plus
        self.system_prompt = self.config.get("system_prompt", "") + self.format_prompt

        # Initialize evaluator and metrics collector through base class methods
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method implements the actual agent logic without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]
    
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem_text}"},
        ]

        # Get solution from OpenAI API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        
        # Extract content from response
        response_content = response.choices[0].message.content
        response_content = response_content.replace('\r\n', '\n').replace('\r', '\n').strip()
        with contextlib.suppress(UnicodeDecodeError):
            response_content = response_content.encode('utf-8').decode('utf-8-sig')  # Remove BOM
        

        # Create message object with usage metadata
        # The base class expects AIMessage-like objects or dicts.
        # Let's create a dict to be handled by the generic dict case.
        ai_message = {
            'content': response_content,
            'name': 'single_agent',
            'role': 'assistant',
            'message_type': 'ai_response',
            'usage_metadata': response.usage
        }

        # Return the response and message with usage metadata for the evaluate method
        return {
            "messages": [ai_message],
            "final_answer": response_content
        }


# Register the agent system
AgentSystemRegistry.register("single_agent", SingleAgent)

# Register the agent as a component for workflow usage
from mas_arena.core.registry import COMPONENT_REGISTRY
COMPONENT_REGISTRY.register("SingleAgent", SingleAgent)