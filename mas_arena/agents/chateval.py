import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry


# Constants
AGENT_NAMES = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def get_model_name(config_model_name: Optional[str] = None) -> str:
    """Get model name from config or environment variable"""
    return config_model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

def create_llm(model_name: str) -> ChatOpenAI:
    """Create ChatOpenAI instance with standardized configuration"""
    return ChatOpenAI(
        model=model_name,
        timeout=60,  # Set request timeout to 60 seconds
        max_retries=2,  # Set maximum retry attempts to 2
    )


@dataclass
class Agent:
    """Represents an LLM agent"""

    agent_id: int
    name: str
    model_name: str
    system_prompt: str
    chat_history: Optional[List[Dict[str, str]]] 

    def __post_init__(self):
        if self.chat_history is None:
            self.chat_history = []
        self.llm = create_llm(self.model_name)

    def _update_chat_history(self, context: str, response_content: Any) -> None:
        """Update chat history with human input and AI response"""
        # Handle case where response_content might be a list or other type
        content = response_content if isinstance(response_content, str) else str(response_content)
        self.chat_history.append({"role": "human", "human": context})
        self.chat_history.append({"role": "ai", "ai": content})

    def _build_messages(self, context: str) -> List[Union[SystemMessage, HumanMessage, AIMessage]]:
        """Build message list for LLM input"""
        return [
            SystemMessage(content=self.system_prompt),
            *[
                (
                    HumanMessage(content=msg["human"])
                    if msg.get("role") == "human"
                    else AIMessage(content=msg["ai"])
                )
                for msg in (self.chat_history or [])
            ],
            HumanMessage(content=context),
        ]

    async def generate_response(self, context: str) -> Dict[str, Union[Any, str]]:
        """Generate agent response"""
        messages = self._build_messages(context)

        # Use standard output directly
        response = await self.llm.ainvoke(messages)
        response.name = self.name

        self._update_chat_history(context, response.content)

        return {"message": response, "solution": response.content}


class ResultExtractor:
    """Extract final results from conversation history"""

    def __init__(self, model_name: Optional[str] = None, format_prompt: str = ""):
        self.model_name = get_model_name(model_name)
        self.format_prompt = format_prompt
        self.llm = create_llm(self.model_name)
        self.name = "result_extractor"

    async def extract(
        self, conversation_history: List[Dict[str, Any]], problem: str
    ) -> Dict[str, Any]:
        """
        Extract final answer from chronologically ordered conversation history
        """
        # Select different prompts based on problem type
        prompt = f"""Original problem: {problem}

Below are the discussion histories of multiple AI agents in chronological order:

{self._format_histories(conversation_history)}

Please analyze the above discussions and provide a final answer. Requirements:
- Synthesize all agents' viewpoints.
- Choose the most reasonable solution/option.
{self.format_prompt}
"""

        messages = [
            SystemMessage(
                content="You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents."
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            response.name = "evaluator"

            return {"message": response}
        except Exception as e:
            print(f"LLM call failed: {str(e)}")
            return {"message": None}

    def _format_histories(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Format conversation history in chronological order"""
        formatted = []
        current_round = None
        
        for entry in conversation_history:
            round_num = entry["round"]
            agent_name = entry["agent_name"]
            response = entry["response"]
            
            # Add round header if this is a new round
            if current_round != round_num:
                formatted.append(f"\n--- Round {round_num} ---")
                current_round = round_num
            
            formatted.append(f"{agent_name}: {response}")
        
        return "\n".join(formatted)


class ChatEval(AgentSystem):
    """Multi-agent evaluation system based on iterative debate"""

    def __init__(self, name: str = "chateval", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config or {})
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.num_rounds = self.config.get("num_rounds", 2)
        self.model_name = get_model_name(self.config.get("model_name"))

        agent_components = self._create_agents()
        self.agents = [w for w in agent_components["workers"] if isinstance(w, Agent)]
        extractors = [
            w for w in agent_components["workers"] if isinstance(w, ResultExtractor)
        ]
        if not extractors:
            raise ValueError(
                "ResultExtractor not found in components created by _create_agents."
            )
        self.extractor = extractors[0]

    def _create_agents(self) -> Dict[str, List[Any]]:
        """Create multiple agent instances and result extractor"""
        debate_agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=i+1,
                chat_history=[],
                name=AGENT_NAMES[i],
                model_name=self.model_name,
                system_prompt=self._get_agent_prompt(i),
            )
            debate_agents.append(agent)

        # Create and assign the extractor here  
        extractor = ResultExtractor(self.model_name, getattr(self, 'format_prompt', ''))

        return {"workers": debate_agents + [extractor]}

    def _get_agent_prompt(self, agent_index: int) -> str:
        """Generate specific system prompt for each agent"""
        if agent_index == 0:
            return """You are a Mathematics Expert, focused on solving mathematical problems. You need to:
1. Carefully analyze the key points of mathematical problems
2. Provide clear mathematical reasoning processes
3. Question or supplement other experts' viewpoints when necessary
4. Ensure answers are accurate and logically sound
5. Use mathematical symbols and formulas to express your thoughts

You are the Mathematics Expert, focused on providing mathematical perspective analysis."""
        elif agent_index == 1:
            return """You are a Logic Expert, focused on logical analysis of problems. You need to:
1. Carefully analyze the logical structure of problems
2. Provide clear reasoning chains
3. Question or supplement other experts' viewpoints when necessary
4. Ensure reasoning processes are rigorous and without loopholes
5. Pay attention to implicit conditions and boundary cases

You are the Logic Expert, focused on providing logical perspective analysis."""
        else:  # agent_index == 2
            return """You are a Critical Thinking Expert, focused on multi-angle analysis of problems. You need to:
1. Carefully analyze multiple aspects of problems
2. Provide comprehensive thinking perspectives
3. Question or supplement other experts' viewpoints when necessary
4. Ensure consideration of various possibilities
5. Pay attention to potential traps and misconceptions

You are the Critical Thinking Expert, focused on providing multi-angle perspective analysis."""

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run iterative debate process"""
        problem_text = problem["problem"]

        # store all LLM response objects
        all_messages = []
        # Store all responses for context building - indexed by [round][agent_index]
        round_responses = []

        # iterative discussion process
        for t in range(self.num_rounds):
            # Clear all agents' chat history at the start of each round to prevent accumulation
            for agent in self.agents:
                agent.chat_history = []
            
            # Store responses for this round
            current_round_responses = []
            
            for n, agent in enumerate(self.agents):
                # Rebuild context from scratch using all previous responses
                context = self._build_context(problem_text, n, t, round_responses)
                response_data = await agent.generate_response(context)

                # save response object
                if "message" in response_data:
                    all_messages.append(response_data["message"])

                # Store this agent's response for current round
                solution_text = response_data.get("solution", "")
                current_round_responses.append({
                    "agent_index": n,
                    "agent_name": AGENT_NAMES[n],
                    "response": solution_text
                })
            
            # Add current round responses to the overall history
            round_responses.append(current_round_responses)

        # Generate conversation history from round responses
        conversation_history = self._generate_conversation_history(round_responses)
        
        # extract final answer using ResultExtractor
        extractor_result = await self.extractor.extract(conversation_history, problem_text)

        # add evaluator message
        if "message" in extractor_result and extractor_result["message"]:
            all_messages.append(extractor_result["message"])
        
        return {
            "messages": all_messages,
            "conversation_history": conversation_history,
            "final_answer": extractor_result["message"].content if extractor_result["message"] else None,
        }

    def _generate_conversation_history(self, round_responses: List[List[Dict]]) -> List[Dict[str, Any]]:
        """Generate conversation history from round responses"""
        conversation_history = []
        for round_idx, round_data in enumerate(round_responses):
            for response_data in round_data:
                conversation_entry = {
                    "round": round_idx + 1,
                    "agent_index": response_data["agent_index"],
                    "agent_name": response_data["agent_name"],
                    "response": response_data["response"],
                }
                conversation_history.append(conversation_entry)
        return conversation_history

    def _build_context(self, problem: str, agent_index: int, round_num: int, 
                                   round_responses: List[List[Dict]]) -> str:
        """Build context for current agent with complete history reconstruction"""
        agent_name = AGENT_NAMES[agent_index]

        problem_statement = f"Original problem: {problem}"
        # Get format_prompt from self if it exists, otherwise use empty string
        format_prompt = getattr(self, 'format_prompt', '')
        if format_prompt:
            problem_statement += format_prompt

        # For the very first agent in the very first round
        if round_num == 0 and agent_index == 0:
            return f"Please solve this problem or select the best option based on your expertise:\n{problem_statement}"

        # Build context with all previous discussions
        context_parts = [f"Round {round_num + 1}, {agent_name}"]
        context_parts.append(problem_statement)
        
        # Add previous rounds' discussions
        if round_responses:
            context_parts.append("\nPrevious rounds' discussions:")
            for prev_round_idx, prev_round in enumerate(round_responses):
                context_parts.append(f"\n--- Round {prev_round_idx + 1} ---")
                for response in prev_round:
                    context_parts.append(f"{response['agent_name']}: {response['response']}")

        context_parts.append("""
Please provide your insights based on previous discussions. You can:
1. Agree with and supplement previous viewpoints
2. Propose different solutions or select a different option if applicable
3. Point out potential issues with previous solutions/selected options
4. Provide new ideas or methods
5. Do not overly expand to other problems
If the problem is multiple choice, please indicate your chosen option clearly in your response.""")

        return "\n".join(context_parts)

# register agent system
AgentSystemRegistry.register("chateval", ChatEval, num_agents=3, num_rounds=2)

if __name__ == "__main__":
    import asyncio
    
    async def test():
        problem = {
            "problem": "A positive integer, its square root is 452, find this positive integer."
        }
        agent = ChatEval(name="chateval", config={"num_agents": 3, "num_rounds": 1})
        result = await agent.run_agent(problem)
        print(result)
    
    asyncio.run(test())
