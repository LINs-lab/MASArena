import time
import json
import asyncio
import random
import uuid
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Add color output support
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(step_name: str, color: str = Colors.BLUE):
    """Print step information"""
    print(f"\n{color}{Colors.BOLD}===== {step_name} ====={Colors.ENDC}")

def print_agent_info(agent: 'Agent', score: float = None):
    """Print agent information"""
    score_str = f" (Score: {score:.4f})" if score is not None else ""
    print(f"{Colors.CYAN}Agent: {agent.name}{score_str}{Colors.ENDC}")
    print(f"  System Prompt: {agent.system_prompt[:100]}...")

@dataclass
class Agent:
    """Represents an LLM agent"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    score: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    llm: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize LLM after dataclass init."""
        self.llm = ChatOpenAI(model=self.model_name)
    
    async def solve(self, problem: str) -> Dict[str, Any]:
        """Solve the given problem and return results"""
        # Create callback handler to collect token usage
        callback_handler = OpenAICallbackHandler()
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=problem)
        ]
        
        start_time = time.time()
        response = await self.llm.ainvoke(messages, config={'callbacks': [callback_handler]})
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Get token usage from callback handler
        input_tokens = callback_handler.prompt_tokens
        output_tokens = callback_handler.completion_tokens
        total_tokens = callback_handler.total_tokens
        
        # Add usage_metadata to AIMessage
        if isinstance(response, AIMessage):
            response.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_token_details": {
                    "system_prompt": len(self.system_prompt.split()),
                    "user_prompt": len(problem.split())
                },
                "output_token_details": {
                    "reasoning": output_tokens,  # Simplified handling, treat all output tokens as reasoning
                }
            }
        
        result = {
            "agent_id": self.agent_id,
            "name": self.name,
            "execution_time_ms": execution_time_ms,
            "extracted_answer": response.content,
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
        }
        
        self.result = result
        return result

class EvoAgent(AgentSystem):
    """
    Multi-agent system based on evolutionary algorithm
    
    Algorithm flow:
    1. Initialize 3 base agents
    2. First iteration: Crossover operation, update parent agent settings based on parent agent results and initial agents, generate new offspring agents
    3. Second iteration: Mutation operation, generate more offspring agents based on parent agents and initial agents
    4. Select 5 best agents
    5. Pass the problem to the final five agents, each generating an answer
    6. Aggregate output through a new LLM
    """
    
    def __init__(self, name: str = "evoagent", config: Dict[str, Any] = None):
        """
        Initialize the evolutionary agent system
        
        Args:
            name: System name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Default configuration
        self.model_name = self.config.get("model_name", "gpt-4o-mini")
        self.initial_agents_count = self.config.get("initial_agents_count", 3)
        self.final_agents_count = self.config.get("final_agents_count", 5)
        self.crossover_rate = self.config.get("crossover_rate", 0.7)
        self.mutation_rate = self.config.get("mutation_rate", 0.3)
        
   
        
    def _initialize_base_agents(self) -> List[Agent]:
        """Initialize base agents"""
        base_agents = []
        
        # Base system prompt templates
        base_prompts = [
            "You are a mathematics expert, skilled in solving mathematical problems. Please think step by step and solve the problem.",
            "You are a logical reasoning expert, skilled in analyzing problems and finding solutions. Please provide detailed reasoning process.",
            "You are a problem-solving expert, skilled in breaking down complex problems into simple steps. Please clearly show your thinking process."
        ]
        
        # Create initial agents
        for i in range(self.initial_agents_count):
            agent_id = str(uuid.uuid4())
            name = f"EVO-{i+1}"  # Use EVO-1, EVO-2, EVO-3 etc. format
            system_prompt = base_prompts[i % len(base_prompts)]
            
            agent = Agent(
                agent_id=agent_id,
                name=name,
                model_name=self.model_name,
                system_prompt=system_prompt
            )
            
            base_agents.append(agent)
            
        return base_agents
    
    async def _crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        Crossover operation: combine features of two parent agents to create offspring
        
        Args:
            parent1: First parent agent
            parent2: Second parent agent
            
        Returns:
            Offspring agent
        """
        try:
            # Add timeout handling
            async with asyncio.timeout(30):  # Set 30 second timeout
                # Use LLM for crossover, add callback to collect token usage
                callback_handler = OpenAICallbackHandler()
                
                llm = ChatOpenAI(
                    model=self.model_name
                )
                
                prompt = f"""
                You are performing a crossover operation on two AI agent configurations to create a new, improved agent.

                Parent 1 configuration:
                - Name: {parent1.name}
                - System prompt: {parent1.system_prompt}
                - Result: {parent1.result.get('extracted_answer', 'No result')}

                Parent 2 configuration:
                - Name: {parent2.name}
                - System prompt: {parent2.system_prompt}
                - Result: {parent2.result.get('extracted_answer', 'No result')}

                Please create a new agent configuration that combines the best features of both parents.
                The new configuration should inherit the advantages of both parents while avoiding their weaknesses.

                Please return the new configuration in JSON format with the following fields:
                - name: Name of the new agent (must start with "EVO-C-", e.g., "EVO-C-1")
                - system_prompt: System prompt for the new agent

                Please ensure the returned format is valid JSON without any additional text or explanations.
                """
                
                response = await llm.ainvoke([{"role": "user", "content": prompt}], config={'callbacks': [callback_handler]})
                
                # Add token usage metadata
                if isinstance(response, AIMessage):
                    response.usage_metadata = {
                        "input_tokens": callback_handler.prompt_tokens,
                        "output_tokens": callback_handler.completion_tokens,
                        "total_tokens": callback_handler.total_tokens,
                        "input_token_details": {"prompt": len(prompt.split())},
                        "output_token_details": {"reasoning": callback_handler.completion_tokens}
                    }
                
                try:
                    # Try to extract JSON content
                    content = response.content.strip()
                    
                    # Try to find JSON start and end positions
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        print(f"{Colors.CYAN}Extracted JSON: {json_str[:100]}...{Colors.ENDC}")
                        
                        # Try to parse JSON response
                        config = json.loads(json_str)
                        
                        # Ensure name starts with EVO-C-
                        name = config.get("name", "")
                        if not name.startswith("EVO-C-"):
                            name = f"EVO-C-{random.randint(1, 999)}"
                        
                        # Create offspring agent
                        child = Agent(
                            agent_id=str(uuid.uuid4()),
                            name=name,
                            model_name=self.model_name,
                            system_prompt=config.get("system_prompt", parent1.system_prompt)
                        )
                        
                        return child
                    else:
                        print(f"{Colors.YELLOW}Warning: Cannot find JSON in response: {content[:100]}...{Colors.ENDC}")
                        raise ValueError("Cannot find JSON in response")
                except Exception as e:
                    print(f"{Colors.YELLOW}Warning: Failed to parse crossover result: {str(e)}, using simple random selection{Colors.ENDC}")
                    # If parsing fails, use simple random selection
                    child = Agent(
                        agent_id=str(uuid.uuid4()),
                        name=f"EVO-C-{random.randint(1, 999)}",
                        model_name=self.model_name,
                        system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
                    )
                    
                    return child
        except asyncio.TimeoutError:
            print(f"{Colors.RED}Warning: Crossover operation timeout, using simple random selection{Colors.ENDC}")
            # Use simple random selection on timeout
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-C-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
            )
            
            return child
        except Exception as e:
            print(f"{Colors.RED}Warning: Crossover operation error: {str(e)}, using simple random selection{Colors.ENDC}")
            # Use simple random selection on error
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-C-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
            )
            
            return child
    
    async def _mutation(self, parent: Agent) -> Agent:
        """
        Mutation operation: create a mutated offspring based on the parent agent
        
        Args:
            parent: Parent agent
            
        Returns:
            Mutated offspring agent
        """
        try:
            # Add timeout handling
            async with asyncio.timeout(30):  # Set 30 second timeout
                # Use LLM for mutation, add callback to collect token usage
                callback_handler = OpenAICallbackHandler()
                
                llm = ChatOpenAI(
                    model=self.model_name
                )
                
                prompt = f"""
                You are performing a mutation operation on an AI agent configuration to create a mutated version.

                Parent configuration:
                - Name: {parent.name}
                - System prompt: {parent.system_prompt}
                - Result: {parent.result.get('extracted_answer', 'No result')}

                Please create a mutated agent configuration that is different from the parent but still effective.
                The mutation should introduce some randomness while maintaining the agent's ability to solve problems.

                Please return the mutated configuration in JSON format with the following fields:
                - name: Name of the mutated agent (must start with "EVO-M-", e.g., "EVO-M-1")
                - system_prompt: System prompt for the mutated agent

                Please ensure the returned format is valid JSON without any additional text or explanations.
                """
                
                response = await llm.ainvoke([{"role": "user", "content": prompt}], config={'callbacks': [callback_handler]})
                
                # Add token usage metadata
                if isinstance(response, AIMessage):
                    response.usage_metadata = {
                        "input_tokens": callback_handler.prompt_tokens,
                        "output_tokens": callback_handler.completion_tokens,
                        "total_tokens": callback_handler.total_tokens,
                        "input_token_details": {"prompt": len(prompt.split())},
                        "output_token_details": {"reasoning": callback_handler.completion_tokens}
                    }
                
                try:
                    # Try to extract JSON content
                    content = response.content.strip()
                    
                    # Try to find JSON start and end positions
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        print(f"{Colors.CYAN}Extracted JSON: {json_str[:100]}...{Colors.ENDC}")
                        
                        # Try to parse JSON response
                        config = json.loads(json_str)
                        
                        # Ensure name starts with EVO-M-
                        name = config.get("name", "")
                        if not name.startswith("EVO-M-"):
                            name = f"EVO-M-{random.randint(1, 999)}"
                        
                        # Create mutated offspring agent
                        child = Agent(
                            agent_id=str(uuid.uuid4()),
                            name=name,
                            model_name=self.model_name,
                            system_prompt=config.get("system_prompt", parent.system_prompt)
                        )
                        
                        return child
                    else:
                        print(f"{Colors.YELLOW}Warning: Cannot find JSON in response: {content[:100]}...{Colors.ENDC}")
                        raise ValueError("Cannot find JSON in response")
                except Exception as e:
                    print(f"{Colors.YELLOW}Warning: Failed to parse mutation result: {str(e)}, using simple random modification{Colors.ENDC}")
                    # If parsing fails, use simple random modification
                    child = Agent(
                        agent_id=str(uuid.uuid4()),
                        name=f"EVO-M-{random.randint(1, 999)}",
                        model_name=self.model_name,
                        system_prompt=parent.system_prompt + f" Mutation version {random.randint(1, 100)}"
                    )
                    
                    return child
        except asyncio.TimeoutError:
            print(f"{Colors.RED}Warning: Mutation operation timeout, using simple random modification{Colors.ENDC}")
            # Use simple random modification on timeout
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-M-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent.system_prompt + f" Mutation version {random.randint(1, 100)}"
            )
            
            return child
        except Exception as e:
            print(f"{Colors.RED}Warning: Mutation operation error: {str(e)}, using simple random modification{Colors.ENDC}")
            # Use simple random modification on error
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-M-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent.system_prompt + f" Mutation version {random.randint(1, 100)}"
            )
            
            return child
    
    def _calculate_score(self, result: Dict[str, Any], problem: Dict[str, Any]) -> float:
        """
        Calculate result score
        
        Args:
            result: Agent's result
            problem: Problem
            
        Returns:
            Score (between 0-1)
        """
        try:
            # Extract answer
            extracted_answer = result.get("extracted_answer", "")
            
            # Use evaluator to calculate score
            score, _ = self.evaluator.calculate_score(problem.get("solution", ""), extracted_answer)
            
            return score
        except Exception:
            return 0.0
    
    async def _summarize_results(self, problem: str, results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        Use LLM to summarize results from multiple agents
        
        Args:
            problem: Problem
            results: List of results from multiple agents
            
        Returns:
            Summarized result and token usage information
        """
        try:
            # Add timeout handling
            async with asyncio.timeout(60):  # Set 60 second timeout
                # Add callback to collect token usage
                callback_handler = OpenAICallbackHandler()
                
                llm = ChatOpenAI(
                    model=self.model_name
                )
                
                # Build summary prompt
                results_text = ""
                for i, result in enumerate(results):
                    results_text += f"Agent {i+1} ({result.get('name', f'Agent-{i+1}')}) answer:\n"
                    results_text += f"{result.get('extracted_answer', 'No answer')}\n\n"
                
                prompt = f"""
                Please summarize the following multiple agents' answers to the same problem, generating a comprehensive and thorough answer.

                Problem:
                {problem}

                {results_text}

                Please provide a comprehensive answer that combines the advantages of all agents and resolves any conflicts or contradictions.
                Remember the following rules:
                {self.format_prompt}
                """
                
                response = await llm.ainvoke([{"role": "user", "content": prompt}], config={'callbacks': [callback_handler]})
                
                # Create token usage metadata
                usage_metadata = {
                    "input_tokens": callback_handler.prompt_tokens,
                    "output_tokens": callback_handler.completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                    "input_token_details": {"prompt": len(prompt.split())},
                    "output_token_details": {"reasoning": callback_handler.completion_tokens}
                }
                
                # Add token usage metadata to AIMessage
                if isinstance(response, AIMessage):
                    response.usage_metadata = usage_metadata
                
                return response.content, usage_metadata
        except asyncio.TimeoutError:
            print(f"{Colors.RED}Warning: Summary result timeout, using best agent's answer{Colors.ENDC}")
            # Use best agent's answer on timeout
            best_result = max(results, key=lambda x: x.get("score", 0))
            return best_result.get("extracted_answer", "Unable to summarize results, using best agent's answer"), {}
        except Exception as e:
            print(f"{Colors.RED}Warning: Summary result error: {str(e)}, using best agent's answer{Colors.ENDC}")
            # Use best agent's answer on error
            best_result = max(results, key=lambda x: x.get("score", 0))
            return best_result.get("extracted_answer", f"Unable to summarize results: {str(e)}"), {}
    
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run evolutionary agent system to solve given problem (async version)
        
        Args:
            problem: Problem dictionary
            **kwargs: Additional parameters
            
        Returns:
            Result dictionary
        """
        # Record start time
        start_time = time.time()
        
        # Extract problem text and problem ID
        problem_text = problem.get("problem", "")
        
        # Display problem
        print_step("Problem", Colors.GREEN)
        print(f"{Colors.YELLOW}{problem_text}{Colors.ENDC}")
        
        # Initialize base agents
        print_step("Initialize Base Agents")
        base_agents = self._initialize_base_agents()
        for agent in base_agents:
            print_agent_info(agent)
        
        # Run base agents asynchronously to get initial results
        print_step("Run Base Agents")
        tasks = []
        for agent in base_agents:
            tasks.append(self._run_agent_task(agent, problem_text, problem))
        
        # Use simple progress display
        print(f"{Colors.CYAN}Base agents progress: 0/{len(tasks)}{Colors.ENDC}")
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                await task
            except Exception as e:
                print(f"{Colors.RED}Warning: Task execution error: {str(e)}{Colors.ENDC}")
            print(f"{Colors.CYAN}Base agents progress: {i + 1}/{len(tasks)}{Colors.ENDC}")
        
        # Sort base agents by score
        base_agents.sort(key=lambda x: x.score, reverse=True)
        
        # Display base agents results
        print_step("Base Agents Results", Colors.GREEN)
        for agent in base_agents:
            print_agent_info(agent, agent.score)
            print(f"  Answer: {agent.result.get('extracted_answer', '')[:100]}...")
        
        # First iteration: Crossover operation
        print_step("First Iteration: Crossover Operation")
        crossover_agents = []
        
        # Keep best base agent
        crossover_agents.append(base_agents[0])
        print(f"{Colors.GREEN}Keep best base agent: {base_agents[0].name}{Colors.ENDC}")
        
        # Create new crossover agents - asynchronous parallel execution
        print_step("Create Crossover Agents")
        crossover_tasks = []
        for _ in range(self.initial_agents_count * 2 - 1):  # -1 because we already added one best base agent
            # Randomly select two parents
            parent1 = random.choice(base_agents)
            parent2 = random.choice(base_agents)
            
            # Execute crossover asynchronously
            crossover_tasks.append(self._crossover(parent1, parent2))
        
        # Wait for all crossover tasks to complete
        crossover_results = await asyncio.gather(*crossover_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(crossover_results):
            if isinstance(result, Exception):
                print(f"{Colors.RED}Warning: Crossover task {i+1} execution error: {str(result)}{Colors.ENDC}")
                continue
                
            crossover_agents.append(result)
            print_agent_info(result)
            print("  Parents: Random selection")
        
        # Run crossover agents asynchronously
        print_step("Run Crossover Agents")
        tasks = []
        for agent in crossover_agents[1:]:  # Skip already evaluated best base agent
            tasks.append(self._run_agent_task(agent, problem_text, problem))
        
        # Use simple progress display
        print(f"{Colors.CYAN}Crossover agents progress: 0/{len(tasks)}{Colors.ENDC}")
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                await task
            except Exception as e:
                print(f"{Colors.RED}Warning: Task execution error: {str(e)}{Colors.ENDC}")
            print(f"{Colors.CYAN}Crossover agents progress: {i + 1}/{len(tasks)}{Colors.ENDC}")
        
        # Sort crossover agents by score
        crossover_agents.sort(key=lambda x: x.score, reverse=True)
        
        # Display crossover agents results
        print_step("Crossover Agents Results", Colors.GREEN)
        for agent in crossover_agents:
            print_agent_info(agent, agent.score)
            print(f"  Answer: {agent.result.get('extracted_answer', '')[:100]}...")
        
        # Second iteration: Mutation operation
        print_step("Second Iteration: Mutation Operation")
        mutation_agents = []
        
        # Keep best crossover agent
        mutation_agents.append(crossover_agents[0])
        print(f"{Colors.GREEN}Keep best crossover agent: {crossover_agents[0].name}{Colors.ENDC}")
        
        # Create new mutation agents - asynchronous parallel execution
        print_step("Create Mutation Agents")
        mutation_tasks = []
        for _ in range(self.initial_agents_count * 3 - 1):  # -1 because we already added one best crossover agent
            # Randomly select one parent
            parent = random.choice(crossover_agents)
            
            # Execute mutation asynchronously
            mutation_tasks.append(self._mutation(parent))
        
        # Wait for all mutation tasks to complete
        mutation_results = await asyncio.gather(*mutation_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(mutation_results):
            if isinstance(result, Exception):
                print(f"{Colors.RED}Warning: Mutation task {i+1} execution error: {str(result)}{Colors.ENDC}")
                continue
                
            mutation_agents.append(result)
            print_agent_info(result)
            print("  Parent: Random selection")
        
        # Run mutation agents asynchronously
        print_step("Run Mutation Agents")
        tasks = []
        for agent in mutation_agents[1:]:  # Skip already evaluated best crossover agent
            tasks.append(self._run_agent_task(agent, problem_text, problem))
        
        # Use simple progress display
        print(f"{Colors.CYAN}Mutation agents progress: 0/{len(tasks)}{Colors.ENDC}")
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                await task
            except Exception as e:
                print(f"{Colors.RED}Warning: Task execution error: {str(e)}{Colors.ENDC}")
            print(f"{Colors.CYAN}Mutation agents progress: {i + 1}/{len(tasks)}{Colors.ENDC}")
        
        # Sort all agents by score
        mutation_agents.sort(key=lambda x: x.score, reverse=True)
        
        # Display mutation agents results
        print_step("Mutation Agents Results", Colors.GREEN)
        for agent in mutation_agents:
            print_agent_info(agent, agent.score)
            print(f"  Answer: {agent.result.get('extracted_answer', '')[:100]}...")
        
        # Select final 5 best agents
        print_step("Select Final Agents", Colors.GREEN)
        final_agents = mutation_agents[:self.final_agents_count]
        for agent in final_agents:
            print_agent_info(agent, agent.score)
        
        # Summarize final agents' results
        print_step("Summarize Final Results")
        final_results = [agent.result for agent in final_agents]
        summary, summary_usage = await self._summarize_results(problem_text, final_results)
        
        # Record end time
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Display summary results
        print_step("Final Summary Results", Colors.GREEN)
        print(f"{Colors.YELLOW}{summary}{Colors.ENDC}")
        print(f"{Colors.CYAN}Execution time: {execution_time_ms:.2f}ms{Colors.ENDC}")
        
        # Build message list, adapted to AgentSystem.evaluate format
        messages = []
        
        # Add user's original question
        user_message = HumanMessage(content=problem_text)
        messages.append(user_message)
        
        # Add all agents' answers as AIMessage, including usage_metadata
        for agent in final_agents:
            ai_message = AIMessage(
                content=agent.result.get("extracted_answer", ""),
                name=agent.name
            )
            
            # Add token usage metadata
            usage_metadata = agent.result.get("usage_metadata", {})
            if usage_metadata:
                ai_message.usage_metadata = {
                    "input_tokens": usage_metadata.get("input_tokens", 0),
                    "output_tokens": usage_metadata.get("output_tokens", 0),
                    "total_tokens": usage_metadata.get("total_tokens", 0),
                    "input_token_details": {
                        "system_prompt": len(agent.system_prompt.split()),
                        "user_prompt": len(problem_text.split())
                    },
                    "output_token_details": {
                        "reasoning": usage_metadata.get("output_tokens", 0)
                    }
                }
            
            messages.append(ai_message)
        
        # Add final summary result as AIMessage, including usage_metadata
        summary_message = AIMessage(
            content=summary,
            name="EVO-SUMMARY"
        )
        
        # Add summary token usage metadata
        if summary_usage:
            summary_message.usage_metadata = summary_usage
            
        messages.append(summary_message)
        
        # Return results, including messages, execution time and evolution metrics
        return {
            "messages": messages,
            "final_answer": summary,
            "execution_time_ms": execution_time_ms,
            "evolution_metrics": {
                "initial_agents": len(base_agents),
                "crossover_agents": len(crossover_agents),
                "mutation_agents": len(mutation_agents),
                "final_agents": len(final_agents),
                "best_score": final_agents[0].score if final_agents else 0.0
            }
        }
        
    async def _run_agent_task(self, agent: Agent, problem_text: str, problem: Dict[str, Any]) -> None:
        """
        Run single agent asynchronously and calculate score
        
        Args:
            agent: Agent to run
            problem_text: Problem text
            problem: Problem dictionary
        """
        try:
            # Add timeout handling
            async with asyncio.timeout(60):  # Set 60 second timeout
                result = await agent.solve(problem_text)
                score = self._calculate_score(result, problem)
                agent.score = score
                agent.result = result
        except asyncio.TimeoutError:
            print(f"{Colors.RED}Warning: Agent {agent.name} execution timeout{Colors.ENDC}")
            agent.score = 0.0
            agent.result = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "execution_time_ms": 60000,  # Timeout duration
                "extracted_answer": "Execution timeout, unable to get answer",
                "usage_metadata": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_token_details": {},
                    "output_token_details": {}
                }
            }
        except Exception as e:
            print(f"{Colors.RED}Warning: Agent {agent.name} execution error: {str(e)}{Colors.ENDC}")
            agent.score = 0.0
            agent.result = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "execution_time_ms": 0,
                "extracted_answer": f"Execution error: {str(e)}",
                "usage_metadata": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_token_details": {},
                    "output_token_details": {}
                }
            }

# Register agent system
AgentSystemRegistry.register("evoagent", EvoAgent)

if __name__ == "__main__":
    # Test EvoAgent
    problem = {
        "problem": "A positive integer whose square root is 452, find this positive integer."
    }
    
    # Use synchronous version of run_agent method
    evo_agent = EvoAgent()
    result = asyncio.run(evo_agent.run_agent(problem))
    print(result)
