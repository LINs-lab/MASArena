"""
ChatEval Agent (New Core - BenchAgent based)

This module implements the new core version of the ChatEval multi-agent debate system.
It follows the pattern of Jarvis New Core, using BenchAgent as the underlying engine for each debater.

Logic:
1.  It orchestrates a debate between 3 specialized agents: Math Expert, Logic Expert, and Critical Thinking Expert.
2.  Each expert is a BenchAgent instance.
    - Math Expert: Uses Python tools to verify calculations.
    - Logic & Critical Thinking Experts: Pure LLM (no tools) for faster, conceptual analysis.
3.  The system runs for a specified number of rounds.
4.  A final synthesizer (ResultExtractor) aggregates the debate into a final answer.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from mas_arena.agents.bench_agent import BenchAgent
from mas_arena.utils.llm_utils import call_model

# Load environment variables
load_dotenv(override=True)

class ChatEvalNewCore(AgentSystem):
    """
    ChatEval (New Core)
    
    A multi-agent debate system where specialized agents (Math, Logic, Critical Thinking)
    discuss a problem before a final answer is extracted.
    
    Each debater is powered by BenchAgent.
    Optimization: Only Math Expert gets Python tools; others are pure LLMs to save time/tokens.
    """

    def __init__(self, name: str = "chateval_newcore", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        
        # Configuration
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.num_rounds = self.config.get("num_rounds", 2)
        
        # Initialize Debaters (BenchAgents)
        self.agents = self._create_debate_agents()
        
        # Store workers for tool integration wrappers or inspection
        self.workers = [{"name": agent.name, "agent": agent} for agent in self.agents]

    def _create_debate_agents(self) -> List[BenchAgent]:
        """Create the 3 specialized BenchAgents"""
        agents = []
        definitions = [
            {
                "name": "Math Expert",
                "role_prompt": (
                    "You are a Mathematics Expert. Analyze the problem mathematically.\n"
                    "Use Python to VERIFY calculations if needed, but be CONCISE.\n"
                    "Do not repeat the problem statement."
                ),
                "use_tools": True
            },
            {
                "name": "Logic Expert",
                "role_prompt": (
                    "You are a Logic Expert. Analyze the logical structure.\n"
                    "Check for loopholes or implicit conditions.\n"
                    "Be EXTREMELY CONCISE. Do not use tools."
                ),
                "use_tools": False
            },
            {
                "name": "Critical Thinking Expert",
                "role_prompt": (
                    "You are a Critical Thinking Expert. Analyze from multiple angles.\n"
                    "Check for traps and misconceptions.\n"
                    "Be EXTREMELY CONCISE. Do not use tools."
                ),
                "use_tools": False
            }
        ]
        
        for i, definition in enumerate(definitions):
            # Create a config for this specific agent
            agent_config = self.config.copy()
            
            # Inject the persona into the instructions
            base_instructions = agent_config.get("additional_instructions", "")
            specific_instructions = f"{definition['role_prompt']}\n\n{base_instructions}"
            
            # Determine tools
            if definition["use_tools"]:
                # Math expert gets configured tools (usually python)
                manager_tools = self.config.get("manager_tools", ["python_interpreter"])
            else:
                # Others get no extra tools (BenchAgent adds FinalAnswer automatically)
                manager_tools = []

            # Initialize BenchAgent
            bench_agent = BenchAgent(
                name=f"agent_{i}_{definition['name'].replace(' ', '_')}",
                model=self.model_name,
                manager_tools=manager_tools,
                search_tools=[], # No search for debaters to keep it focused/fast
                memory=None, # No long-term memory needed for debate turns
                api_key=self.config.get("api_key"),
                api_base=self.config.get("api_base"),
                max_steps=self.config.get("max_steps", 5), # Keep steps low
                verbosity_level=self.config.get("verbosity_level", 1),
                additional_instructions=specific_instructions,
                # Pass other config items
                **{k: v for k, v in self.config.items() if k not in [
                    "model_name", "manager_tools", "search_tools", "memory", 
                    "api_key", "api_base", "max_steps", "verbosity_level", 
                    "additional_instructions", "name", "num_rounds"
                ]}
            )
            agents.append(bench_agent)
            
        return agents

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the iterative debate process.
        """
        problem_text = problem["problem"]
        all_messages = []
        debate_history = []  # Stores string representation of the debate for context
        
        # 1. Debate Loop
        for round_idx in range(self.num_rounds):
            for agent_idx, agent in enumerate(self.agents):
                agent_name = ["Math Expert", "Logic Expert", "Critical Thinking Expert"][agent_idx]
                
                # Build context for the current agent
                context = self._build_context(problem_text, debate_history, round_idx, agent_name)
                
                # Run the agent (step mode)
                try:
                    result = await agent.run_agent_step(
                        augmented_question=context,
                        additional_args={
                            "expected_answer": problem.get("solution")
                        }
                    )
                    
                    response_text = result.get("final_answer", "No answer provided.")
                    
                    # Store result
                    debate_history.append(f"{agent_name} (Round {round_idx + 1}): {response_text}")
                    
                    # Collect messages for logging/visualization
                    agent_messages = result.get("messages", [])
                    for msg in agent_messages:
                        if isinstance(msg, dict) and "name" not in msg:
                             msg["name"] = agent.name
                    all_messages.extend(agent_messages)
                    
                except Exception as e:
                    error_msg = f"Error in debate round {round_idx} with {agent_name}: {str(e)}"
                    debate_history.append(f"{agent_name} (Round {round_idx + 1}): [Error] {error_msg}")
                    all_messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "name": agent.name,
                        "message_type": "error"
                    })

        # 2. Result Extraction
        final_answer = await self._extract_final_result(debate_history, problem_text)
        
        return {
            "messages": all_messages,
            "final_answer": final_answer,
            "extracted_answer": final_answer,
            "debate_history": debate_history
        }

    def _build_context(self, problem: str, debate_history: List[str], round_idx: int, agent_name: str) -> str:
        """Build the prompt context for the agent"""
        
        history_text = "\n\n".join(debate_history) if debate_history else "No previous discussion."
        format_prompt = self.format_prompt or ""
        
        # Concise prompt structure
        return (
            f"Original Problem: {problem}\n\n"
            f"Discussion History:\n{history_text}\n\n"
            f"{format_prompt}\n"
            f"It is now Round {round_idx + 1}. You are {agent_name}.\n"
            "Provide your insights. Be CONCISE. "
            "If you are the Math Expert, verify with tools if needed. "
            "Others: pure reasoning only."
        )

    async def _extract_final_result(self, debate_history: List[str], problem: str) -> str:
        """Extract the final answer using a synthesizer LLM call"""
        history_text = "\n".join(debate_history)
        
        prompt = (
            f"Original problem: {problem}\n\n"
            f"Debate History:\n{history_text}\n\n"
            "Synthesize the debate and provide the final answer.\n"
            "Requirements:\n"
            "- Choose the most reasonable solution.\n"
            "- Provide ONLY the final answer text.\n"
            f"{self.format_prompt or ''}"
        )
        
        try:
            response = await asyncio.to_thread(
                call_model,
                prompt,
                self.model_name
            )
            return response
        except Exception as e:
            return f"Error extracting final answer: {str(e)}"

# Register
AgentSystemRegistry.register("chateval_newcore", ChatEvalNewCore)
