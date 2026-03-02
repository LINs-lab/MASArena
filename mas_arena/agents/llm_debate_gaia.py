"""
LLM Debate Agent (New Core - Balanced Implementation)

This module reimplements the LLM Debate agent using the unified BenchAgent core.
Instead of hardcoding multi-agent debate orchestration logic, we inject 
debate-specific behavior via system prompts while leveraging BenchAgent's 
robust execution framework (tool handling, file processing, retry logic).

Key Philosophy:
- Preserve the *essence* of debate (multi-perspective reasoning) via prompt engineering
- Delegate *execution* to BenchAgent's mature ReAct engine
- Eliminate fragile custom orchestration code (round management, context stitching)
- Maintain compatibility with benchmark requirements (format_prompt, file handling)
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent

load_dotenv(override=True)

class LLMDebateAgent(AgentSystem):
    """
    LLM Debate Agent (New Core)
    
    Uses BenchAgent as the execution engine with debate-specific prompt injection.
    The "multi-agent debate" behavior is simulated through structured instructions
    rather than hardcoded orchestration logic.
    """

    def __init__(self, name: str = "llm_debate", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        
        # Core debate parameters (preserved for metadata reporting)
        self.agents_num = self.config.get("agents_num", 2)
        self.rounds_num = self.config.get("rounds_num", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # Construct debate-specific system instructions
        debate_instructions = (
            f"You are a debate coordinator simulating a structured debate among {self.agents_num} distinct AI agents "
            f"over {self.rounds_num} rounds to solve the problem.\n\n"
            f"**Debate Protocol**:\n"
            f"1. **Perspective Generation**: Initially generate {self.agents_num} distinct viewpoints with clear reasoning.\n"
            f"2. **Iterative Refinement**: For {self.rounds_num} rounds, have each perspective critically engage with others:\n"
            f"   - Identify strengths/weaknesses in other viewpoints\n"
            f"   - Revise own position with new evidence or logic\n"
            f"   - Maintain distinct agent identities (Agent 1, Agent 2, etc.)\n"
            f"3. **Synthesis**: After final round, objectively aggregate insights into a single verified answer.\n\n"
            f"**Critical Requirements**:\n"
            f"- Maintain clear separation between agent perspectives in your reasoning trace\n"
            f"- Explicitly label each agent's contributions (e.g., '[Agent 1 Round 2]')\n"
            f"- Final answer must be unambiguous and directly address the problem\n"
            f"- NEVER fabricate consensus; acknowledge unresolved tensions if they exist"
        )
        
        # Inject benchmark-specific formatting requirements
        if self.format_prompt:
            debate_instructions += f"\n\n**Output Format Requirement**:\n{self.format_prompt}"
        
        # Merge with user-provided instructions if exists
        user_instructions = self.config.get("additional_instructions", "")
        if user_instructions:
            debate_instructions = f"{debate_instructions}\n\n**Additional Guidance**:\n{user_instructions}"
        
        # Configure BenchAgent with debate personality
        self.bench_agent_config = {
            "model": self.model_name,
            # Explicitly disable tools since debate is reasoning-focused
            # (Preserves BenchAgent's file handling capabilities for GAIA-style tasks)
            "manager_tools": self.config.get("manager_tools", []),
            "search_tools": self.config.get("search_tools", []),
            "memory": self.config.get("memory"),
            
            # API Configuration
            "api_key": self.config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            "api_base": self.config.get("api_base") or os.getenv("OPENAI_API_BASE"),
            
            # Execution Parameters
            "max_steps": self.config.get("max_steps", 25),  # Higher steps for multi-round simulation
            "search_max_steps": self.config.get("search_max_steps", 10),
            "verbosity_level": self.config.get("verbosity_level", 1),
            
            # Core behavior injection: Debate protocol as system personality
            "additional_instructions": debate_instructions,
            
            # Identity
            "name": f"{name}_core",
            
            # Pass through other valid config keys
            **{k: v for k, v in self.config.items() if k not in [
                "agents_num", "rounds_num", "model_name", "manager_tools", "search_tools",
                "memory", "api_key", "api_base", "max_steps", "search_max_steps",
                "verbosity_level", "additional_instructions", "system_prompt"
            ]}
        }
        
        self.bench_agent = BenchAgent(**self.bench_agent_config)

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute debate simulation via BenchAgent's unified interface.
        Leverages BenchAgent.run_agent() to handle:
        - File attachment preprocessing (critical for GAIA benchmarks)
        - Semantic verification loops
        - Structured step logging
        - Error resilience
        """
        try:
            # Full run_agent path preserves file handling and retry logic
            result = await self.bench_agent.run_agent(problem, **kwargs)
        except Exception as e:
            error_msg = f"Debate execution failed: {str(e)}"
            return {
                "messages": [{
                    "role": "assistant",
                    "content": error_msg,
                    "name": self.name,
                    "message_type": "error"
                }],
                "final_answer": error_msg,
                "extracted_answer": "",
                "error": str(e),
                "agents_participated": self.agents_num,
                "rounds_completed": self.rounds_num
            }
        
        # Standardize output with preserved metadata for benchmark compatibility
        final_answer = result.get("final_answer", "No answer generated.")
        return {
            "messages": result.get("messages", []),
            "final_answer": final_answer,
            "extracted_answer": result.get("extracted_answer", final_answer),
            # Preserve original metadata fields for benchmark reporting
            "agents_participated": self.agents_num,
            "rounds_completed": self.rounds_num,
            # Pass through BenchAgent's rich execution artifacts
            "manager_agent_steps": result.get("manager_agent_steps", []),
            "search_agent_steps": result.get("search_agent_steps", []),
            "search_keywords": result.get("search_keywords", ""),
            "error": result.get("error")
        }

# Register with original identifier to enable seamless migration
AgentSystemRegistry.register(
    "llm_debate", 
    LLMDebateAgent,
    agents_num=2, 
    rounds_num=3
)