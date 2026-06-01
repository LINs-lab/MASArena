"""
Jarvis Agent (New Core - Balanced Implementation)

This module implements the new core version of the Jarvis agent, powered by BenchAgent.
It strikes a balance between the original HuggingGPT "Plan-Execute-Summarize" philosophy
and the modern BenchAgent ReAct core.

Logic:
1. It uses BenchAgent's ReAct core (CodeAgent) as the execution engine to ensure
   parity with other agents (same tool capabilities, memory, robustness).
2. It enforces the "Jarvis Style" behavior via specific System Prompts and Instructions.
   Instead of a static JSON plan, we instruct the ReAct agent to "Plan first, then execute".
   
This ensures the *behavior* feels like Jarvis (planning-heavy), but the *capability* matches
other BenchAgents (robust execution, error handling).
"""

import os
from typing import Dict, Any, List, Optional

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent

# Load environment variables

class JarvisAgent(AgentSystem):
    """
    Jarvis Agent (New Core - Balanced)
    
    Uses BenchAgent's robust ReAct core but prompted to behave in a 
    Plan-then-Execute manner similar to the original Jarvis.
    """

    def __init__(self, name: str = "jarvis", config: Dict[str, Any] = None):
        """
        Initialize the Jarvis Agent.
        """
        super().__init__(name, config)
        self.config = config or {}
        
        # 1. Configuration Setup
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # 2. Jarvis-Style Prompt Engineering
        # We inject a System Prompt that encourages the "Planning" behavior
        # distinct from a standard ReAct agent that might jump straight to action.
        jarvis_instructions = (
            "You are Jarvis, a highly capable AI assistant that solves problems through rigorous planning and execution.\n"
            "Your workflow MUST follow these principles:\n"
            "1. **Analyze & Plan**: Before writing any code or calling tools, briefly analyze the user's request and outline the necessary steps.\n"
            "2. **Execute**: Implement your plan using the available tools (Python code, Search, etc.).\n"
            "3. **Verify & Summarize**: Check your results and provide a clear, concise final answer.\n\n"
            "You have access to a powerful Python interpreter and search tools. Use them proactively."
        )
        
        # Merge with user-provided system prompt if any
        user_system_prompt = self.config.get("system_prompt", "")
        if user_system_prompt:
            jarvis_instructions = f"{jarvis_instructions}\n\nAdditional Instructions:\n{user_system_prompt}"
            
        # Ensure format prompt is included (for benchmarks like Math)
        if self.format_prompt:
            jarvis_instructions += f"\n\nOutput Requirements:\n{self.format_prompt}"

        # 3. BenchAgent Initialization
        # We use the standard BenchAgent, ensuring the "Core" is identical to Debate/AutoGen
        self.bench_agent_config = {
            "model": self.model_name,
            # Tools: default to config or allow BenchAgent's defaults
            "manager_tools": self.config.get("manager_tools"), 
            "search_tools": self.config.get("search_tools"),
            "memory": self.config.get("memory"),
            
            # API Settings
            "api_key": self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            
            # Execution Limits
            "max_steps": self.config.get("max_steps", 15),
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 1),
            
            # Instructions: This is where we inject the Jarvis "soul" into the BenchAgent body
            "additional_instructions": jarvis_instructions,
            
            # Identity
            "name": f"{name}_core",
            
            # Pass through any other valid kwargs
            **{k: v for k, v in self.config.items() if k not in [
                "model_name", "manager_tools", "search_tools", "memory", 
                "api_key", "api_base", "max_steps", "search_max_steps", 
                "verbosity_level", "additional_instructions", "system_prompt"
            ]}
        }
        
        self.bench_agent = BenchAgent(**self.bench_agent_config)

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the Jarvis agent to solve a problem.
        Delegates to BenchAgent.run_agent to preserve full capabilities:
        - GAIA `files` attachment handling (auto descriptions + tool usage)
        - semantic check / retry loop (when enabled in BenchAgent)
        - consistent logging/steps collection
        """
        # NOTE: `BenchAgent.run_agent_step()` is intentionally a simplified single-shot
        # mode and does NOT run the attachment preprocessing in `BenchAgent.run_agent()`.
        # GAIA tasks frequently include local files (png/xlsx/docx/mp3, etc.), so we must
        # call the full `run_agent()` path here.
        try:
            result = await self.bench_agent.run_agent(problem, **kwargs)
        except Exception as e:
            # Fallback error handling
            error_msg = f"Jarvis execution failed: {str(e)}"
            return {
                "messages": [{
                    "role": "assistant",
                    "content": error_msg,
                    "name": self.name,
                    "message_type": "error"
                }],
                "final_answer": error_msg,
                "extracted_answer": "",
                "error": str(e)
            }
        
        # Standardize output for BenchmarkRunner: pass-through + safe defaults
        final_answer = result.get("final_answer", "No answer generated.")
        return {
            "messages": result.get("messages", []),
            "final_answer": final_answer,
            "extracted_answer": result.get("extracted_answer", final_answer),
            "manager_agent_steps": result.get("manager_agent_steps", []),
            "search_agent_steps": result.get("search_agent_steps", []),
            "search_keywords": result.get("search_keywords", ""),
        }

# Register as "jarvis" to replace the old implementation
AgentSystemRegistry.register("jarvis", JarvisAgent)
