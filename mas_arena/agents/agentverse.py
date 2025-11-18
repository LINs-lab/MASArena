import time
import json
import os
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, TypedDict, Any, List

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

# Define TypedDict classes for structured output
class ExpertTeam(TypedDict):
    """Expert team configuration"""
    agents: List[Dict[str, Any]]

class ExpertSolution(TypedDict):
    """Expert solution"""
    analysis: str
    solution: str
    confidence: int  # 1-5 score indicating expert's confidence in their answer

class EvaluationResult(TypedDict):
    """Evaluation result"""
    status: str  # "complete" or "need_new_experts"
    final_solution: str  # final solution
    feedback: str  # feedback (if new experts are needed)
    reasoning: str  # evaluation reasoning
    improvement_score: float  # improvement degree compared to last iteration (0-1)
    solution_quality: float  # solution quality score (0-1)

# Load environment variables
load_dotenv()

@dataclass
class ExpertProfile:
    id: str
    name: str
    description: str

class Agent(BaseModel):
    name: str
    describe: str
    agent_id: int

class Agents(BaseModel):
    agents: List[Agent]

class Discussion(TypedDict):
    agent_id: int
    context: str

class SumDiscussion(TypedDict):
    sum_context: List[Discussion]

class RecruiterAgent:
    """Recruitment agent: generates descriptions for work agents"""
    def __init__(self, agent_id: str, model_name: str = None, num_agents: int = 3):
        self.agent_id = agent_id
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.num_agents = num_agents
        self.system_prompt = (
            "You are the leader of a group of experts who needs to recruit the right team configuration to solve complex problems.\n\n"
            "Your responsibilities:\n"
            "1. Analyze the problem and identify necessary expertise areas\n"
            "2. Generate diverse expert descriptions with complementary specializations\n"
            "3. Ensure each expert has clearly defined roles and responsibilities\n"
            "4. Adapt team composition based on feedback when provided\n\n"
            "Each expert should have specialized knowledge directly relevant to the problem, "
            "and the team should collectively be capable of solving the complete problem."
        )
        # Initialize LLM with structured output
        self.llm = ChatOpenAI(
            model=self.model_name
        )
        
    def _create_prompt(self, problem: str, feedback: str = None) -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"""
            Previous evaluation feedback:
            {feedback}
            
            Please consider this feedback when forming your new team of experts.
            """
            
        return f"""
            Generate the configuration of {self.num_agents} expert agents based on the following problem:

            Problem:
            {problem}
            
            {feedback_section}

            What experts will you recruit to better solve this problem?

            For each expert, provide:
            1. Agent ID (starting from 1)
            2. Expert name reflecting their expertise area  
            3. Detailed description of their role and responsibilities


            Agent ID: {self.agent_id}
        """

    async def describe(self, problem: str, feedback: str = None):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_prompt(problem, feedback))
        ]

        start_time = time.time()
        end_time = start_time  # Initialize end_time to prevent undefined errors
        
        try:
            # Use structured output to call LLM
            llm_with_schema = self.llm.with_structured_output(schema=ExpertTeam, include_raw=True)
            response = await llm_with_schema.ainvoke(messages)
            end_time = time.time()  # Update end_time
            
            # Extract content from structured response
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # Validate structured data
            if not isinstance(structured_data, dict) or "agents" not in structured_data or not structured_data["agents"]:
                print(f"Warning: Invalid or empty response from recruiter. Raw content: {raw_response.content[:200]}...")
                # Try to extract expert information from raw response
                try:
                    # Try to parse JSON
                    import re
                    # Look for possible JSON objects
                    json_match = re.search(r'(\{.*\})', raw_response.content.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(1)
                        parsed_data = json.loads(potential_json)
                        if "agents" in parsed_data and parsed_data["agents"]:
                            structured_data = parsed_data
                        else:
                            # Create default expert team
                            structured_data = {"agents": self._create_default_experts()}
                    else:
                        structured_data = {"agents": self._create_default_experts()}
                except Exception as parse_err:
                    print(f"Error parsing recruiter response: {str(parse_err)}")
                    structured_data = {"agents": self._create_default_experts()}
            
            
            # Set name
            raw_response.name = f"recruiter_{self.agent_id}"
            
            return {
                "agent_id": self.agent_id,
                "solution": structured_data,
                "message": raw_response,  # Save original message to preserve usage_metadata
                "latency_ms": (end_time - start_time) * 1000,
            }
            
        except Exception as e:
            # If structured output fails, fall back to standard mode
            print(f"Structured output failed for recruiter: {str(e)}. Falling back to standard output.")
            
            # Re-invoke model without structured output
            response = await self.llm.ainvoke(messages)
            end_time = time.time()
            
            # Set name
            response.name = f"recruiter_{self.agent_id}"
            
            # Try to extract JSON from response content
            try:
                # Try to parse directly as JSON
                content_text = response.content
                try:
                    content_json = json.loads(content_text)
                    if "agents" in content_json and content_json["agents"]:
                        structured_data = content_json
                    else:
                        structured_data = {"agents": self._create_default_experts()}
                except json.JSONDecodeError:
                    # Try to find JSON part in text
                    import re
                    json_match = re.search(r'(\{.*\})', content_text.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(1)
                        try:
                            parsed_json = json.loads(potential_json)
                            if "agents" in parsed_json and parsed_json["agents"]:
                                structured_data = parsed_json
                            else:
                                structured_data = {"agents": self._create_default_experts()}
                        except Exception as e:
                            print(f"[WARNING] Error parsing recruiter content: {str(e)}")
                            structured_data = {"agents": self._create_default_experts()}
                    else:
                        # If no valid JSON found, create default experts
                        structured_data = {"agents": self._create_default_experts()}
            except Exception as parse_error:
                print(f"[WARNING] Error parsing recruiter content: {str(parse_error)}")
                structured_data = {"agents": self._create_default_experts()}
            
            return {
                "agent_id": self.agent_id,
                "solution": structured_data,
                "message": response,
                "latency_ms": (end_time - start_time) * 1000,
            }
    
    def _create_default_experts(self) -> List[Dict[str, Any]]:
        """Create default expert team when structured output fails"""
        default_experts = []
        expert_types = [
            {"name": "Mathematics Expert", "describe": "Expert specialized in mathematical problems, calculations and proofs."},
            {"name": "Problem Analysis Expert", "describe": "Expert responsible for analyzing problem structure and breaking down complex problems."},
            {"name": "Solution Expert", "describe": "Expert who integrates analysis results and provides complete solutions."}
        ]
        
        # Create experts based on configured number
        for i in range(1, min(self.num_agents + 1, len(expert_types) + 1)):
            expert = expert_types[i-1].copy()
            expert["agent_id"] = i
            default_experts.append(expert)
        
        return default_experts

class WorkAgent:
    """Work agent that solves specific aspects of a problem"""
    def __init__(self, agent_id: str, system_prompt: str = None, format_prompt: str = ""):
        self.agent_id = agent_id
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            f"{system_prompt}\n\n"
            f"You are a specialized expert participating in a collaborative problem-solving team.\n"
            f"Your task is to analyze the problem from your expertise area and provide a detailed solution.\n\n"
            f"Output requirements:\n"
            f"- Analyze the problem from your expert perspective\n"
            f"- Provide a detailed solution with clear reasoning\n"
            f"- Rate your confidence in the solution (1-5 scale, 5 = highest confidence)\n"
            f"- Explain your approach and methodology\n"
            f"{format_prompt}"
        )
        self.llm = ChatOpenAI(
            model=self.model_name,
            max_tokens=1000
        )

    async def solve(self, problem: str, feedback: str = None):
        """Solve a problem with optional feedback"""
        feedback_section = ""
        if feedback:
            feedback_section = f"""
            Feedback from previous evaluation:
            {feedback}
            
            Please consider this feedback when analyzing the problem.
            """
            
        problem_content = f"""
        Problem to solve:
        {problem}
        
        {feedback_section}
        
        As the expert described in your role, please analyze this problem from your specialized perspective and provide your solution. 
        Include your reasoning process and rate your confidence in the solution.
        """
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=problem_content)
        ]

        try:
            llm_with_schema = self.llm.with_structured_output(schema=ExpertSolution, include_raw=True)
            response = await llm_with_schema.ainvoke(messages)
            
            # Extract structured data and raw response
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # Ensure structured_data contains all required fields
            if "solution" not in structured_data:
                structured_data["solution"] = raw_response.content
            if "analysis" not in structured_data:
                structured_data["analysis"] = "No analysis provided"
            if "confidence" not in structured_data:
                structured_data["confidence"] = 3  # Default medium confidence
            
            
            # Set name
            raw_response.name = f"expert_{self.agent_id}"
            
            return {
                "agent_id": self.agent_id,
                "solution": structured_data,
                "message": raw_response,
            }
        except Exception as e:
            # If structured output fails, fall back to standard mode
            print(f"Structured output failed for agent {self.agent_id}: {str(e)}. Falling back to standard output.")
            
            # Re-invoke model without structured output
            response = await self.llm.ainvoke(messages)
            end_time = time.time()
            
            # Set name
            response.name = f"expert_{self.agent_id}"
            
            return {
                "agent_id": self.agent_id,
                "solution": response.content,  # Return raw content on fallback
                "message": response,
            }

class Evaluator:
    """Evaluates agent solutions and decides whether to recruit new experts or provide final solution"""
    def __init__(self, model_name: str = None, max_iterations: int = 3, min_quality_threshold: float = 0.7, min_improvement_threshold: float = 0.1):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.max_iterations = max_iterations
        self.min_quality_threshold = min_quality_threshold  # Minimum solution quality threshold
        self.min_improvement_threshold = min_improvement_threshold  # Minimum improvement threshold
        self.previous_solution_quality = 0  # Quality of previous round's solution
        self.llm = ChatOpenAI(
            model=self.model_name
        )
        
    async def evaluate(self, problem: str, solutions: List[Dict[str, Any]], iteration: int, previous_solutions: List[Dict[str, Any]] = None, format_prompt: str = "") -> Dict[str, Any]:
        """
        Evaluate solutions from multiple agents and decide whether to:
        1. Provide final solution if satisfactory
        2. Provide feedback for another round of recruitment
        
        Args:
            problem: Original problem description
            solutions: List of solutions from agents
            iteration: Current iteration count
            previous_solutions: Solutions from previous iteration for comparison
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract structured solutions (if available)
        solutions_details = []
        for sol in solutions:
            try:
                if "structured_solution" in sol:
                    # Use structured solution
                    structured = sol["structured_solution"]
                    solutions_details.append(
                        f"Expert {sol['agent_id']}:\n"
                        f"Analysis: {structured.get('analysis', 'No analysis provided')}\n"
                        f"Solution: {structured.get('solution', 'No solution provided')}\n"
                        f"Confidence: {structured.get('confidence', 3)}/5\n"
                    )
                else:
                    # Use regular solution
                    solutions_details.append(f"Expert {sol['agent_id']} solution:\n{sol.get('solution', 'No solution provided')}\n")
            except Exception as e:
                print(f"Error processing solution from agent {sol.get('agent_id', 'unknown')}: {str(e)}")
                solutions_details.append(f"Expert {sol.get('agent_id', 'unknown')} solution:\nError: Could not process solution\n")
                
        solutions_text = "\n\n".join(solutions_details)
        
        # If there are previous round solutions, add them to prompt for comparison
        previous_solutions_text = ""
        if previous_solutions and len(previous_solutions) > 0:
            prev_details = []
            for sol in previous_solutions:
                try:
                    if "structured_solution" in sol:
                        structured = sol["structured_solution"]
                        prev_details.append(
                            f"Expert {sol['agent_id']}:\n"
                            f"Analysis: {structured.get('analysis', 'No analysis provided')}\n"
                            f"Solution: {structured.get('solution', 'No solution provided')}\n"
                            f"Confidence: {structured.get('confidence', 3)}/5\n"
                        )
                    else:
                        prev_details.append(f"Expert {sol['agent_id']} solution:\n{sol.get('solution', 'No solution provided')}\n")
                except Exception:
                    continue
            
            if prev_details:
                previous_solutions_text = "\n\nPrevious iteration solutions:\n" + "\n\n".join(prev_details)
        
        prompt = f"""
        # Evaluation Task: Multi-Expert Solution Assessment
        
        You are an expert evaluator responsible for analyzing multiple expert solutions and making critical decisions about the problem-solving process.
        
        ## Problem Context
        **Original Problem:**
        {problem}
        
        **Current Iteration:** {iteration} of {self.max_iterations}
        
        ## Expert Solutions Analysis
        **Current Expert Solutions:**
        {solutions_text}
        {previous_solutions_text}
        
        ## Your Evaluation Mission
        As the lead evaluator, you must:
        
        ### 1. **Solution Quality Assessment**
        - Analyze each expert's contribution and confidence level
        - Identify strengths and weaknesses in the proposed solutions
        - Evaluate how well the solutions collectively address the problem
        - Rate the overall solution quality (0-1 scale, where 1 = perfect solution)
        
        ### 2. **Progress Evaluation** 
        - Compare current solutions with previous iterations (if applicable)
        - Assess the degree of improvement from previous rounds
        - Rate the improvement score (0-1 scale, where 1 = major improvement)
        
        ### 3. **Decision Making**
        Choose one of the following decisions:
        
        **Option A: Complete the Process**
        - If solutions collectively solve the problem satisfactorily
        - If quality meets acceptable standards
        - If maximum iterations reached
        
        **Option B: Continue with New Experts**
        - If solutions have significant gaps or errors
        - If specific expertise is missing
        - If improvement potential remains high
        
        ## Response Requirements
        
        ### If Completing (status: "complete"):
        - Provide a comprehensive final solution combining the best insights
        - Include step-by-step reasoning and methodology
        - Ensure the solution directly answers the original problem
        {format_prompt}
        
        ### If Continuing (status: "need_new_experts"):
        - Explain specific shortcomings in current solutions
        - Identify what types of expertise are needed
        - Provide actionable feedback for expert recruitment
        
        ## Quality Metrics
        You must provide two numerical scores:
        1. **solution_quality** (0-1): Overall quality of current solutions
        2. **improvement_score** (0-1): Improvement compared to previous iteration
        
        ## Evaluation Standards
        - Be objective and thorough in your analysis
        - Consider both correctness and completeness of solutions
        - Balance perfectionism with practical problem-solving needs
        - Provide constructive feedback that guides improvement
        
        Your evaluation will determine whether the problem-solving process continues or concludes.
        """
        
        messages = [
            SystemMessage(content="You are an expert evaluator that analyzes multiple solutions and determines if they adequately solve the problem. You can either provide a final comprehensive solution or request improvements with specific feedback."),
            HumanMessage(content=prompt)
        ]
        
        start_time = time.time()
        end_time = start_time  # Initialize end_time to prevent undefined errors
        
        try:
            # Use structured output to call LLM
            llm_with_schema = self.llm.with_structured_output(schema=EvaluationResult, include_raw=True)
            response = await llm_with_schema.ainvoke(messages)
            end_time = time.time()  # Update end_time
            
            # Extract content from structured response
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # Ensure all required fields exist in structured_data
            if "status" not in structured_data:
                structured_data["status"] = "need_new_experts" if iteration < self.max_iterations else "complete"
            if "final_solution" not in structured_data:
                structured_data["final_solution"] = raw_response.content
            if "feedback" not in structured_data:
                structured_data["feedback"] = ""
            if "reasoning" not in structured_data:
                structured_data["reasoning"] = "No reasoning provided"
            if "solution_quality" not in structured_data:
                structured_data["solution_quality"] = 0.5  # Default medium quality
            if "improvement_score" not in structured_data:
                structured_data["improvement_score"] = 0.1  # Default minimal improvement
                
            # Smart termination decision logic
            current_quality = structured_data["solution_quality"]
            improvement = structured_data["improvement_score"]
            
            # Condition 1: If quality exceeds threshold, complete early
            if current_quality >= self.min_quality_threshold:
                structured_data["status"] = "complete"
                print(f"Solution quality {current_quality} exceeds threshold {self.min_quality_threshold}, completing early.")
            
            # Condition 2: If improvement below threshold and not first iteration, may be stuck
            if iteration > 1 and improvement < self.min_improvement_threshold:
                structured_data["status"] = "complete"
                print(f"Improvement {improvement} below threshold {self.min_improvement_threshold}, stopping iterations.")
            
            # Condition 3: If reached max iterations, must complete
            if iteration >= self.max_iterations:
                structured_data["status"] = "complete"
                print(f"Reached maximum iterations ({self.max_iterations}), completing.")
            
            # Store current quality score for next iteration comparison
            self.previous_solution_quality = current_quality
            
            # Set name
            raw_response.name = "evaluator"
            
            return {
                "final_solution": structured_data["final_solution"],
                "message": raw_response,  # Save original message to preserve usage_metadata
                "latency_ms": (end_time - start_time) * 1000,
                "evaluation": structured_data,
            }
            
        except Exception as e:
            # If structured output fails, fall back to standard mode and try JSON parsing
            print(f"Structured output failed for evaluator: {str(e)}. Falling back to standard output and JSON parsing.")
            
            # Re-invoke model without structured output
            response = await self.llm.ainvoke(messages)
            end_time = time.time()
            
            # Set name
            response.name = "evaluator"
            
            # Try to parse JSON from response
            try:
                # Clean response, remove markdown code blocks
                content = response.content
                import re
                content = re.sub(r'```(?:json)?', '', content)
                content = content.strip()
                content = re.sub(r'```$', '', content).strip()
                
                # Try direct JSON parsing
                try:
                    evaluation = json.loads(content)
                except json.JSONDecodeError:
                    # If parsing fails, try regex to extract JSON object
                    json_match = re.search(r'({.*})', content.replace('\n', ' '), re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                        # Handle escape sequences
                        json_str = json_str.replace('\\', '\\\\')
                        evaluation = json.loads(json_str)
                    else:
                        # If still can't extract, create default evaluation
                        evaluation = {
                            "status": "need_new_experts" if iteration < self.max_iterations else "complete",
                            "final_solution": response.content,
                            "feedback": f"Could not extract structured evaluation. Please provide a team that can solve: {problem[:100]}...",
                            "reasoning": "Error parsing evaluation",
                            "solution_quality": 0.5,
                            "improvement_score": 0.1
                        }
            except Exception as parse_error:
                print(f"Error parsing evaluator response: {str(parse_error)}")
                # Create default evaluation
                evaluation = {
                    "status": "need_new_experts" if iteration < self.max_iterations else "complete",
                    "final_solution": response.content,
                    "feedback": f"Error processing evaluation. Please provide a team that can solve: {problem[:100]}...",
                    "reasoning": "Error in evaluation process",
                    "solution_quality": 0.5,
                    "improvement_score": 0.1
                }
            
            # Ensure all required fields exist
            if "status" not in evaluation:
                evaluation["status"] = "need_new_experts" if iteration < self.max_iterations else "complete"
            if "final_solution" not in evaluation:
                evaluation["final_solution"] = response.content
            if "feedback" not in evaluation:
                evaluation["feedback"] = ""
            if "reasoning" not in evaluation:
                evaluation["reasoning"] = "No reasoning provided"
            if "solution_quality" not in evaluation:
                evaluation["solution_quality"] = 0.5
            if "improvement_score" not in evaluation:
                evaluation["improvement_score"] = 0.1
            
            # Smart termination decision logic
            current_quality = evaluation["solution_quality"]
            improvement = evaluation["improvement_score"]
            
            # Condition 1: If quality exceeds threshold, complete early
            if current_quality >= self.min_quality_threshold:
                evaluation["status"] = "complete"
                print(f"Solution quality {current_quality} exceeds threshold {self.min_quality_threshold}, completing early.")
            
            # Condition 2: If improvement below threshold and not first iteration, may be stuck
            if iteration > 1 and improvement < self.min_improvement_threshold:
                evaluation["status"] = "complete"
                print(f"Improvement {improvement} below threshold {self.min_improvement_threshold}, stopping iterations.")
            
            # Condition 3: If reached max iterations, must complete
            if iteration >= self.max_iterations:
                evaluation["status"] = "complete"
                print(f"Reached maximum iterations ({self.max_iterations}), completing.")
            
            # Store current quality score for next iteration comparison
            self.previous_solution_quality = current_quality
                
            return {
                "final_solution": evaluation.get("final_solution", ""),
                "message": response,
                "latency_ms": (end_time - start_time) * 1000,
                "evaluation": evaluation,
            }

class AgentVerse(AgentSystem):
    """
    AgentVerse Multi-Agent System
    
    This agent system uses a recruiter to create specialized agents for different aspects 
    of a problem, with results aggregated to produce a final solution.
    """
    
    def __init__(self, name: str = "agentverse", config: Dict[str, Any] = None):
        """Initialize the AgentVerse System"""
        super().__init__(name, config)
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.use_parallel = self.config.get("parallel", True)
        self.max_iterations = self.config.get("max_iterations", 3)
        
        # New quality control and early stopping configuration
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.7)
        self.min_improvement_threshold = self.config.get("min_improvement_threshold", 0.1)
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 2)
        self.max_runtime = self.config.get("max_runtime", 300)  # Default maximum runtime 5 minutes
        
     
    async def _create_agents(self, problem: str, feedback: str = None) -> Dict[str, Any]:
        """
        Create specialized work agents based on the problem and optional feedback
        
        Args:
            problem: Original problem description
            feedback: Optional feedback from previous evaluation
            
        Returns:
            Dictionary with workers and message
        """
        # Use recruiter to determine agent profiles
        recruiter = RecruiterAgent(
            agent_id="recruiter_001", 
            model_name=self.model_name,
            num_agents=self.num_agents
        )
        response_dict = await recruiter.describe(problem, feedback)
        
        # Get expert configuration from structured output
        expert_config = response_dict.get("solution", {})
        agents_list = expert_config.get("agents", [])
        
        # Create expert team
        expert_team = []
        for idx, agent in enumerate(agents_list, 1):
            # Ensure dictionary has necessary fields
            if isinstance(agent, dict):
                agent_id = agent.get("agent_id", str(idx))
                if not isinstance(agent_id, str):
                    agent_id = str(agent_id)
                    
                expert_team.append(
                    ExpertProfile(
                        id=agent_id,
                        name=agent.get("name", f"Expert {agent_id}"),
                        description=agent.get("describe", agent.get("description", ""))[:500]  # Support different field names and truncate long descriptions
                    )
                )
        
        # If no experts are found, create default experts
        if not expert_team:
            print("Warning: No experts found in recruiter response, creating default experts")
            for i in range(1, self.num_agents + 1):
                expert_team.append(
                    ExpertProfile(
                        id=str(i),
                        name=f"General Expert {i}",
                        description="A general expert who can solve various aspects of the problem."
                    )
                )
        
        # Create work agents based on profiles
        workers = []
        for expert in expert_team:
            workers.append(
                WorkAgent(
                    agent_id=expert.id,
                    system_prompt=expert.description,
                    format_prompt=self.format_prompt
                )
            )
        return {"workers": workers, "message": response_dict.get("message", None)}

    async def _solve_async(self, worker: WorkAgent, problem: str, feedback: str = None) -> Dict[str, Any]:
        """Solve a problem asynchronously with a worker agent"""
        return await worker.solve(problem, feedback)

    async def _async_solve_problem(self, problem: str, workers: List[WorkAgent], feedback: str = None) -> List[Dict[str, Any]]:
        """Solve a problem with multiple worker agents asynchronously"""
        # Create tasks for each worker
        tasks = [self._solve_async(worker, problem, feedback) for worker in workers]
        
        # Run all tasks concurrently
        solutions = await asyncio.gather(*tasks)
        
        return solutions

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
        
        # Initialize messages and solutions
        all_messages = []
        all_solutions = []
        feedback = None
        final_solution = None
        
        # Track previous round solutions for improvement comparison
        previous_solutions = None
        
        # Use class attributes instead of config
        min_quality = self.min_quality_threshold
        min_improvement = self.min_improvement_threshold
        early_stopping_rounds = self.early_stopping_rounds
        max_runtime = self.max_runtime
        
        # Track consecutive rounds without significant improvement
        no_improvement_count = 0
        start_runtime = time.time()
        
        # Create evaluator
        evaluator = Evaluator(
            model_name=self.model_name, 
            max_iterations=self.max_iterations,
            min_quality_threshold=min_quality,
            min_improvement_threshold=min_improvement
        )
        
        # Run iterations until evaluator is satisfied or max iterations reached
        for iteration in range(1, self.max_iterations + 1):
            # Check if exceeded maximum runtime
            current_runtime = time.time() - start_runtime
            if current_runtime > max_runtime:
                print(f"Reached maximum runtime ({max_runtime}s), stopping at iteration {iteration}")
                break
            
            print(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Create specialized agents for this problem with feedback from previous iteration
            recruiter_response = await self._create_agents(problem_text, feedback)
            agents = recruiter_response.get("workers", [])
            recruiter_message = recruiter_response.get("message", None)
            
            # Add recruiter message to all messages
            if recruiter_message:
                all_messages.append(recruiter_message)
            
            # Run agents either in parallel or sequentially
            if self.use_parallel:
                # Set up async execution
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                # Run all agents asynchronously
                agent_solutions = loop.run_until_complete(
                    self._async_solve_problem(problem_text, agents)
                )
            else:
                # Run agents sequentially
                agent_solutions = []
                for agent in agents:
                    solution = agent.solve(problem_text)
                    agent_solutions.append(solution)
            
            # Collect agent messages and solutions for this iteration
            iteration_messages = []
            for solution in agent_solutions:
                if "message" in solution:
                    iteration_messages.append(solution["message"])
                    all_messages.append(solution["message"])
            
            # Store solutions for current iteration
            all_solutions.append({
                "iteration": iteration,
                "solutions": agent_solutions
            })
            
            # Get previous round expert solutions (if any)
            if iteration > 1 and len(all_solutions) > 1:
                previous_solutions = all_solutions[iteration-2]["solutions"]
            
            # Evaluate solutions, pass previous solutions for comparison
            evaluation_result = await evaluator.evaluate(problem_text, agent_solutions, iteration, previous_solutions, self.format_prompt)
            evaluation = evaluation_result.get("evaluation", {})
            
            # Add evaluator message
            if "message" in evaluation_result:
                all_messages.append(evaluation_result["message"])
            
            # Get improvement score, increment no-improvement count if below threshold
            improvement_score = evaluation.get("improvement_score", 0)
            if iteration > 1 and improvement_score < min_improvement:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # Reset count
            
            # Check if early stopping condition met
            if no_improvement_count >= early_stopping_rounds:
                print(f"Early stopping after {no_improvement_count} rounds with no significant improvement")
                # Use current best solution
                final_solution = evaluation.get("final_solution", "")
                break
            
            # Check if we need another iteration
            status = evaluation.get("status", "need_new_experts")
            
            if status == "complete":
                final_solution = evaluation.get("final_solution", "")
                print(f"Evaluation complete after {iteration} iterations")
                break
            else:
                feedback = evaluation.get("feedback", "")
        
        # If we reached max iterations without a satisfactory solution, use the last evaluation
        if final_solution is None and all_solutions:
            last_evaluation = await evaluator.evaluate(problem_text, all_solutions[-1]["solutions"], self.max_iterations, all_solutions[:-1] if len(all_solutions) > 1 else None)
            final_solution = last_evaluation.get("evaluation", {}).get("final_solution", "No satisfactory solution found")
            # Add final evaluator message
            if "message" in last_evaluation:
                all_messages.append(last_evaluation["message"])
        
        # For math problems, ensure the final solution is properly formatted
        if isinstance(final_solution, (int, float)):
            final_solution = f"The answer is \\boxed{{{final_solution}}}"
        
        # Filter messages to only include those with usage_metadata for evaluation framework
        messages_with_metadata = [msg for msg in all_messages if hasattr(msg, 'usage_metadata') and msg.usage_metadata]
        
        # Return final answer and all messages
        return {
            "messages": messages_with_metadata,  # Only return messages with usage_metadata
            "final_answer": final_solution,
            "agent_solutions": all_solutions,
        }

# Register the agent system with default parameters
# Ensure these defaults match those in the AgentVerse class
AgentSystemRegistry.register("agentverse", AgentVerse, num_agents=3, parallel=True, max_iterations=3)

if __name__ == "__main__":
    problem = {
        "problem": "What is the sum of the first 100 natural numbers?",
        "id": "problem_1"
    }
    result = AgentVerse().run_agent(problem)
    print(result)
