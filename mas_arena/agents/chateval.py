import os
from typing import Dict, List, Any, TypedDict
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# define structured output class, use TypedDict instead of Pydantic
class AgentResponse(TypedDict):
    """Structured output for agent responses"""
    analysis: str  # Problem analysis
    solution: str  # Solution
    confidence: int  # Confidence level in the solution, range 1-5

@dataclass
class Agent:
    """Represents an LLM agent"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        self.chat_history = []
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,  # Set request timeout to 60 seconds
            max_retries=2        # Set maximum retry attempts to 2
        )

    async def generate_response(self, context: str) -> Any:
        """Generate agent response"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history],
            HumanMessage(content=context)
        ]
        
        # Use structured output
        try:
            llm_with_schema = self.llm.with_structured_output(schema=AgentResponse, include_raw=True)
            response = await llm_with_schema.ainvoke(messages)
            
            # Get structured data and raw response
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            
            # Ensure structured_data is a dictionary, not an object
            if hasattr(structured_data, "dict"):
                structured_data = structured_data.dict()
            elif hasattr(structured_data, "model_dump"):
                structured_data = structured_data.model_dump()
            
            # Set AI message name
            raw_response.name = self.name
            
            # Update chat history
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": raw_response.content
            })
            
            # Return raw response object
            return {
                "message": raw_response,
                "structured_solution": structured_data,
                "solution": raw_response.content
            }
            
        except Exception as e:
            print(f"Structured output failed: {str(e)}, falling back to standard output")
            
            # Fallback to standard output
            response = await self.llm.ainvoke(messages)
            response.name = self.name
            
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": response.content
            })
            
            return {
                "message": response,
                "solution": response.content
            }

class ResultExtractor:
    """Extract final results from conversation history"""
    def __init__(self, model_name: str = None, format_prompt: str = ""):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.format_prompt = format_prompt
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,  # Set request timeout to 60 seconds
            max_retries=2        # Set maximum retry attempts to 2
        )
        self.name = "result_extractor"
        
    async def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> Dict[str, Any]:
        """
        Extract final answer from all agents' conversation histories
        """
        # Select different prompts based on problem type
        prompt = f"""Original problem: {problem}

Below are the discussion histories of multiple AI agents:

{self._format_histories(all_histories)}

Please analyze the above discussions and provide a final answer. Requirements:
- Synthesize all agents' viewpoints.
- Choose the most reasonable solution/option.
{self.format_prompt}
"""
  
        messages = [
            SystemMessage(content="You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            response.name = "evaluator"
            
            return {
                "message": response
            }
        except Exception as e:
            print(f"LLM call failed: {str(e)}")
            return {
                "message": None
            }

    def _format_histories(self, all_histories: List[List[Dict[str, str]]]) -> str:
        """Format all conversation histories"""
        formatted = []
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for i, history in enumerate(all_histories):
            formatted.append(f"\n{agent_names[i]}'s discussion:")
            for msg in history:
                if msg.get("role") == "human":
                    formatted.append(f"Question: {msg['human']}")
                else:
                    formatted.append(f"Answer: {msg['ai']}")
        return "\n".join(formatted)
        

class ChatEval(AgentSystem):
    """Multi-agent evaluation system based on iterative debate"""
    
    def __init__(self, name: str = "chateval", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.num_rounds = self.config.get("num_rounds", 2)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # Initialize agents and extractor via _create_agents
        # self.agents and self.extractor will be set by _create_agents
        agent_components = self._create_agents()
        self.agents = [w for w in agent_components["workers"] if isinstance(w, Agent)]
        extractors = [w for w in agent_components["workers"] if isinstance(w, ResultExtractor)]
        if not extractors:
            raise ValueError("ResultExtractor not found in components created by _create_agents.")
        self.extractor = extractors[0]

    def _create_agents(self) -> List[Agent]:
        """Create multiple agent instances and result extractor"""
        # This method will be patched by ToolIntegrationWrapper if this system is wrapped.
        # The wrapper expects a dictionary: {"workers": [worker1, worker2, ...]}
        # Each worker should have a .name and .llm attribute.
        
        debate_agents = []
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=f"agent_{i+1}",
                name=agent_names[i],
                model_name=self.model_name,
                system_prompt=self._get_agent_prompt(i)
            )
            debate_agents.append(agent)
        
        # Create and assign the extractor here
        extractor = ResultExtractor(self.model_name, self.format_prompt)
        # self.extractor = extractor # Assign to self if needed elsewhere before run_agent completes,
                                 # but __init__ already handles setting self.extractor.

        return {
            "workers": debate_agents + [extractor]
        }

    def _get_agent_prompt(self, agent_index: int) -> str:
        """Generate specific system prompt for each agent"""
        # Set different prompts for three different roles
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
        agent_histories = []
        
        # iterative discussion process
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for t in range(self.num_rounds):
            for n, agent in enumerate(self.agents):
                # generate response for current agent
                context = self._build_context(problem_text, n, t)
                response_data = await agent.generate_response(context)
                
                # save response object
                if "message" in response_data:
                    all_messages.append(response_data["message"])
                
                # add response to context of subsequent agents
                solution_text = response_data.get("solution", "")
                for m in range(n + 1, len(self.agents)):
                    self.agents[m].chat_history.append({
                        "role": "human",
                        "human": f"{agent_names[n]}'s response: {solution_text}"
                    })
        
        # extract all agent chat histories
        agent_histories = [agent.chat_history for agent in self.agents]
        
        # extract final answer
        extractor_result = await self.extractor.extract(agent_histories, problem_text)
        
        # add evaluator message
        if "message" in extractor_result and extractor_result["message"]:
            all_messages.append(extractor_result["message"])
        return {
            "messages": all_messages,  # contains all LLM response objects
            "final_answer": extractor_result["message"].content
        }

    def _build_context(self, problem: str, agent_index: int, round_num: int) -> str:
        """Build context for current agent"""
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        agent_name = agent_names[agent_index]
        
        problem_statement = f"Original problem: {problem}"
        problem_statement += self.format_prompt

        if round_num == 0 and agent_index == 0:
            return f"Please solve this problem or select the best option based on your expertise:\n{problem_statement}"
        
        return f"""Round {round_num + 1}, {agent_name}
        
{problem_statement}

Please provide your insights based on previous discussions. You can:
1. Agree with and supplement previous viewpoints
2. Propose different solutions or select a different option if applicable
3. Point out potential issues with previous solutions/selected options
4. Provide new ideas or methods
5. Do not overly expand to other problems
If the problem is multiple choice, please indicate your chosen option clearly in your response."""

# register agent system
AgentSystemRegistry.register(
    "chateval",
    ChatEval,
    num_agents=3,
    num_rounds=2
)

if __name__ == "__main__":
    # test
    problem = {
        "problem": "A positive integer, its square root is 452, find this positive integer."
    }
    agent = ChatEval(name="chateval", config={"num_agents": 3, "num_rounds": 2})
    result = agent.run_agent(problem)
    print(result)
