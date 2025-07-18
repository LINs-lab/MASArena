"""
Base Agent System Interface

This module provides the base classes and interfaces for agent systems.
"""

import abc
from typing import Dict, Any, Optional, Type, Callable
import uuid
import os
import json
from pathlib import Path
import datetime
import time
from mas_arena.agents.format_prompts import get_format_prompt
from openai.types.completion_usage import CompletionUsage
import aiofiles

class AgentSystem(abc.ABC):
    """Base class for all agent systems in the benchmark framework
    
    To ensure compatibility with MCP Tool Integration (via ToolIntegrationWrapper):
    - If your agent system (or its sub-agents/workers) uses an LLM that needs
      tools bound to it, expose the LLM instance via an attribute named `llm`.
      For example, `self.llm = ChatOpenAI(...)` or `worker.llm = ChatOpenAI(...)`.
    - For multi-agent systems, if you implement a `_create_agents` method,
      it should return a dictionary like: `{"workers": [worker1, worker2, ...]}`.
      Each `worker` object in the list should:
        - Have a `name` attribute (e.g., `worker.name = "researcher"`).
        - Have an `llm` attribute if it's intended to use tools bound to an LLM.
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        """
        Initialize the agent system.

        Args:
            name: Name of the agent system
            config: Configuration parameters for the agent system
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.evaluator_name = self.config.get("evaluator", None)
        if self.evaluator_name is None:
            raise ValueError("Evaluator name is not set in the configuration.")
        
        self.metrics_registry = None
        self.evaluator = None
        self.metrics_collector = None
        
        # Initialize storage for agent responses
        self.agent_responses = []
        
        # Configure paths for saving agent responses and visualizations
        self.responses_dir = Path(self.config.get("responses_dir", "results/agent_responses"))
        self.visualizations_dir = Path(self.config.get("visualizations_dir", "results/visualizations"))
        
        # Create directories if they don't exist
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
       
        self.format_prompt = self.format_prompt()
        # ToolManager is now initialized by the ToolIntegrationWrapper, not the base agent
        self.tool_manager = None

    def format_prompt(self) -> str:
        """
        Format the prompt for different benchmarks.
        
        Returns:
            The appropriate format prompt string for the current evaluator
        """
        return get_format_prompt(self.evaluator_name) or ""

    def _initialize_evaluator(self, evaluator_type: Type = None):
        """
        Initialize the appropriate evaluator based on configuration.
        
        Args:
            evaluator_type: Optional evaluator class to use
        """
        if self.evaluator is not None:
            return

        
        if evaluator_type is None:
            # Import here to avoid circular imports
            try:
                from mas_arena.evaluators import AVAILABLE_EVALUATORS
                # Select evaluator_type based on evaluator_name
                evaluator_type = AVAILABLE_EVALUATORS[self.evaluator_name]
               
            except ImportError:
                raise ImportError("Could not import evaluator. Please provide evaluator_type.")
        
        # Create evaluator instance
        self.evaluator = evaluator_type(
            name=self.evaluator_name,
            config={
                "data_path": self.config.get("data_path", f"data/{self.evaluator_name}_test.jsonl"),
                "log_path": self.config.get("log_path", f"data/results/{self.evaluator_name.upper()}")
            }
        )


    def _initialize_metrics_collector(self):
        """Initialize the metrics collector"""
        if self.metrics_collector is not None:
            return
            
        try:
            from mas_arena.metrics.collector import MetricsCollector
            self.metrics_collector = MetricsCollector()
            if self.metrics_registry:
                self.metrics_collector.set_metrics_registry(self.metrics_registry)
        except ImportError:
            pass  # Metrics collector is optional

    @abc.abstractmethod
    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method should be implemented by subclasses to run the actual agent logic
        without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary of run results (e.g., messages)
        """
        pass

    def _record_token_usage(self, problem_id: str, execution_time_ms: float, messages: list):
        """
        Record token usage metrics from AI messages with usage_metadata.
        
        Args:
            problem_id: ID of the problem
            execution_time_ms: Execution time in milliseconds
            messages: List of messages from the run
            
        Returns:
            Dictionary with collected LLM usage metrics
        """
        if not self.metrics_collector:
            return {} 
        
        # Track metrics from AIMessages with usage_metadata
        total_tokens = 0
        message_count = 0
        usage_metrics = []
        
        for message in messages:
            usage_metadata = None
            if hasattr(message, 'usage_metadata'):
                usage_metadata = message.usage_metadata
            elif isinstance(message, dict) and 'usage_metadata' in message:
                usage_metadata = message['usage_metadata']

            if usage_metadata:
                message_count += 1
                agent_id = ""
                if hasattr(message, 'name'):
                    agent_id = message.name
                elif isinstance(message, dict):
                    agent_id = message.get("name") or message.get("agent_id")

                agent_id = agent_id or (message.id if hasattr(message, 'id') and message.id else f"agent_{hash(message)}")

                if isinstance(usage_metadata, CompletionUsage):
                    input_tokens = usage_metadata.prompt_tokens
                    output_tokens = usage_metadata.completion_tokens
                    reasoning_tokens = getattr(usage_metadata.completion_tokens_details, 'reasoning_tokens', None)
                    total_tokens_msg = usage_metadata.total_tokens

                    input_token_details = usage_metadata.prompt_tokens_details
                    output_token_details = usage_metadata.completion_tokens_details
                    
                else:
                    # Extract metrics from usage_metadata
                    input_tokens = usage_metadata.get('input_tokens', 0)
                    output_tokens = usage_metadata.get('output_tokens', 0)
                    reasoning_tokens = usage_metadata["output_token_details"].get("reasoning", 0)
                    total_tokens_msg = usage_metadata.get('total_tokens', input_tokens + output_tokens)
                    
                    
                    input_token_details = usage_metadata.get('input_token_details', {})
                    output_token_details = usage_metadata.get('output_token_details', {})
                
                total_tokens += total_tokens_msg  
                # Record detailed token metrics directly from the message's usage_metadata
                self.metrics_collector.record_llm_usage(
                    agent_id=agent_id,
                    model_name=os.getenv("MODEL_NAME", ""),
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=total_tokens_msg,
                    reasoning_tokens=reasoning_tokens,
                    input_token_details=input_token_details,
                    output_token_details=output_token_details,
                    latency_ms=execution_time_ms / message_count if message_count > 0 else 0,
                    tags={"agent_system": self.name, "problem_id": problem_id}
                )
                
                # Collect usage metrics
                usage_metrics.append({
                    "agent_id": agent_id,
                    "model_name": os.getenv("MODEL_NAME", "gpt-4o-mini"),
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens_msg,
                    "latency_ms": execution_time_ms / message_count if message_count > 0 else 0
                })

            # Record agent interactions from tuple-style messages
            elif isinstance(message, tuple) and len(message) > 1:
                agent_id, content = message
                
                if agent_id != "user" and self.metrics_collector:  # Skip user messages
                    self.metrics_collector.record_agent_interaction(
                        from_agent="system",
                        to_agent=agent_id,
                        message_type="response",
                        content=content,
                        tags={"agent_system": self.name, "problem_id": problem_id}
                    )
        
        # Record total tokens for this problem
        if message_count > 0:
            self.metrics_collector.record_metric(
                "problem.total_tokens",
                total_tokens,
                {
                    "problem_id": problem_id,
                    "agent_system": self.name
                }
            )
        
        # Return collected metrics
        return {
            "total_tokens": total_tokens,
            "message_count": message_count,
            "agent_usage": usage_metrics
        }

    def _record_agent_responses(self, problem_id: str, messages: list):
        """
        Record internal agent responses to be saved to a file.
        
        Args:
            problem_id: ID of the problem
            messages: List of messages from the run
            
        Returns:
            List of formatted agent responses
        """
        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            return []
        
        formatted_responses = []
        
        for i, message in enumerate(messages):
            response_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "problem_id": problem_id,
                "message_index": i,
                "agent_system": self.name,
            }
            
            # Handle different message formats
            if isinstance(message, AIMessage):
                agent_id = message.name if hasattr(message, 'name') and message.name else message.id if hasattr(message, 'id') and message.id else f"agent_{hash(message)}"
                response_data.update({
                    "agent_id": agent_id,
                    "content": message.content,
                    "role": "assistant",
                    "message_type": "ai_response"
                })
                
                # Include tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    response_data["tool_calls"] = message.tool_calls
                
                # Include additional metadata if present
                if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                    response_data["additional_metadata"] = message.additional_kwargs
                
            elif isinstance(message, HumanMessage):
                response_data.update({
                    "agent_id": "user",
                    "content": message.content,
                    "role": "user",
                    "message_type": "human_query"
                })
                
            # Handle tuple-style messages from multi-agent systems
            elif isinstance(message, tuple) and len(message) > 1:
                agent_id, content = message
                agent_type = "user" if agent_id == "user" else "agent"
                
                response_data.update({
                    "agent_id": agent_id,
                    "content": content,
                    "role": "user" if agent_id == "user" else "assistant",
                    "message_type": f"{agent_type}_message"
                })
                
            # Handle dictionary-style messages
            elif isinstance(message, dict):
                agent_id = message.get("agent_id", message.get("name", f"unknown_agent_{i}"))
                response_data.update({
                    "agent_id": agent_id,
                    "content": message.get("content", ""),
                    "role": message.get("role", "assistant"),
                    "message_type": message.get("message_type", "agent_message")
                })
                
                # Include any additional fields from the dict
                for key, value in message.items():
                    if key not in response_data:
                        # Convert CompletionUsage to a serializable dict
                        if key == "usage_metadata" and isinstance(value, CompletionUsage):
                            response_data[key] = value.model_dump()
                        else:
                            response_data[key] = value
            
            if response_data.get("content"):  # Only add if there's content
                formatted_responses.append(response_data)
                self.agent_responses.append(response_data)
        
        return formatted_responses

    async def save_agent_responses(self, problem_id: str, run_id: str = None, problem: Dict[str, Any] = None):
        """
        Save agent responses to a file.
        
        Args:
            problem_id: ID of the problem
            run_id: Optional run ID to use in the filename
            problem: Optional problem dictionary containing the question
            
        Returns:
            Path to the saved file
        """
        # Filter responses for the current problem ID
        responses_for_problem = [
            r for r in self.agent_responses if r.get("problem_id") == problem_id
        ]

        if not responses_for_problem:
            return None
            
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or self.generate_run_id()
        filename = f"{self.name}_{problem_id}_{timestamp}_{run_id[:8]}.json"
        file_path = self.responses_dir / filename
        
        # Extract question from problem
        question = ""
        if problem:
            question = problem.get("problem", problem.get("question", problem.get("prompt", "")))
        
        # Extract ground truth answer from problem
        ground_truth = ""
        if problem:
            ground_truth = problem.get("answer", problem.get("ground_truth", problem.get("solution", "")))
        
        # Save to file
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps({
                "problem_id": problem_id,
                "question": question,
                "agent_system": self.name,
                "run_id": run_id,
                "timestamp": timestamp,
                "ground_truth": ground_truth,
                "responses": responses_for_problem
            }, indent=2))
        print(f"Saved agent responses to {file_path}")
        return file_path

    def generate_visualization_data(self, problem_id=None):
        """
        Generate data for visualization of agent interactions.
        
        Args:
            problem_id: Optional problem ID to filter responses
            
        Returns:
            Dictionary with visualization data
        """
        # Filter responses by problem_id if provided
        responses = self.agent_responses
        if problem_id:
            responses = [r for r in responses if r.get("problem_id") == problem_id]
            
        if not responses:
            return {"nodes": [], "links": []}
            
        # Extract unique agent IDs
        agent_ids = set()
        for response in responses:
            agent_ids.add(response.get("agent_id", "unknown"))
            
        # Create nodes for each agent
        nodes = [{"id": agent_id, "type": "user" if agent_id == "user" else "agent"} for agent_id in agent_ids]
        
        # Create links based on message sequence
        links = []
        prev_agent_id = None
        
        for i, response in enumerate(responses):
            agent_id = response.get("agent_id", "unknown")
            
            # Skip if this is the first message or same agent talking consecutively
            if prev_agent_id and prev_agent_id != agent_id:
                links.append({
                    "source": prev_agent_id,
                    "target": agent_id,
                    "value": 1,  # Count of interactions
                    "message_indices": [i-1, i]  # Include both the source and target message indices
                })
                
            prev_agent_id = agent_id
            
        # Combine parallel links (same source and target)
        combined_links = {}
        for link in links:
            key = f"{link['source']}->{link['target']}"
            if key in combined_links:
                combined_links[key]["value"] += 1
                # Add all message indices without duplicates
                for idx in link["message_indices"]:
                    if idx not in combined_links[key]["message_indices"]:
                        combined_links[key]["message_indices"].append(idx)
            else:
                combined_links[key] = {
                    "source": link["source"],
                    "target": link["target"],
                    "value": 1,
                    "message_indices": link["message_indices"]
                }
                
        return {
            "nodes": nodes,
            "links": list(combined_links.values())
        }

    async def save_visualization_data(self, problem_id: str, run_id: str = None):
        """
        Save visualization data to a file.
        
        Args:
            problem_id: ID of the problem
            run_id: Optional run ID to use in the filename
            
        Returns:
            Path to the saved file
        """
        visualization_data = self.generate_visualization_data(problem_id)
        if not visualization_data["nodes"]:
            return None
            
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or self.generate_run_id()
        filename = f"viz_{self.name}_{problem_id}_{timestamp}_{run_id[:8]}.json"
        file_path = self.visualizations_dir / filename
        
        # Get responses for this problem
        responses = [r for r in self.agent_responses if r.get("problem_id") == problem_id]
        
        # Save to file
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps({
                "problem_id": problem_id,
                "agent_system": self.name,
                "run_id": run_id,
                "timestamp": timestamp,
                "visualization": visualization_data,
                "responses": responses
            }, indent=2))
            
        return file_path

    async def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate the agent system on a given problem.
        
        This method handles running the agent, evaluating the results,
        and collecting metrics.
        """
        metrics_registry = kwargs.get("metrics_registry", self.metrics_registry)
        if metrics_registry:
            self.metrics_registry = metrics_registry
            
        self._initialize_evaluator()
        self._initialize_metrics_collector()
        
        run_id = self.generate_run_id()
        problem_id = problem.get("id", f"problem_{hash(problem.get('problem', ''))}")

        if self.metrics_collector:
            self.metrics_collector.set_metrics_registry(self.metrics_registry)
            self.metrics_collector.record_metric(
                "problem.process", 1.0, 
                {"problem_id": problem_id, "agent_system": self.name, "evaluator": self.evaluator.name, "run_id": run_id}
            )
            self.metrics_collector.start_timer(
                "problem_evaluation", 
                {"problem_id": problem_id, "agent_system": self.name, "evaluator": self.evaluator.name, "run_id": run_id}
            )
        
        try:
            agent_run_start_time = time.perf_counter()
            run_output = await self.run_agent(problem, **kwargs)
            agent_run_end_time = time.perf_counter()
            execution_time_ms = (agent_run_end_time - agent_run_start_time) * 1000
            
            messages = run_output.get("messages", [])
            usage_metrics = self._record_token_usage(problem_id, execution_time_ms, messages)
            
            self._record_agent_responses(problem_id, messages)
            response_file = await self.save_agent_responses(problem_id, run_id, problem)
            visualization_file = await self.save_visualization_data(problem_id, run_id)
            
            if self.metrics_collector and (response_file or visualization_file):
                self.metrics_collector.record_metric(
                    "agent.response_files",
                    {"response_file": str(response_file) if response_file else None, "visualization_file": str(visualization_file) if visualization_file else None},
                    {"problem_id": problem_id, "agent_system": self.name, "run_id": run_id}
                )
            
            eval_start_time = time.perf_counter()
            eval_result = self.evaluator.evaluate(problem=problem, run_result=run_output)
            eval_end_time = time.perf_counter()
            eval_duration_ms = (eval_end_time - eval_start_time) * 1000
            
            if self.metrics_collector:
                score = eval_result.get("score", 0)
                self.metrics_collector.record_evaluation_result(
                    problem_id=problem_id, score=score, duration_ms=execution_time_ms,
                    metrics={"passed": score == 1, "agent_system": self.name, "run_id": run_id},
                    tags={"agent_system": self.name, "evaluator": self.evaluator.name, "run_id": run_id}
                )
            
            return {
                "status": "success",
                **eval_result,
                "messages": messages,
                "execution_time_ms": execution_time_ms,
                "llm_usage": usage_metrics,
                "response_file": str(response_file) if response_file else None,
                "visualization_file": str(visualization_file) if visualization_file else None,
                "run_id": run_id
            }
            
        except Exception as e:
            # Record error
            if self.metrics_collector:
                self.metrics_collector.stop_timer("problem_evaluation")
                self.metrics_collector.record_error(
                    "evaluation_error",
                    str(e),
                    {
                        "problem_id": problem_id,
                        "agent_system": self.name,
                        "error_type": type(e).__name__,
                        "run_id": run_id
                    }
                )
            
            # Return a failure result instead of re-raising
            return {
                "status": "error",
                "score": 0.0,
                "is_correct": False,
                "reasoning": f"Evaluation failed with error: {str(e)}",
                "messages": [],
                "execution_time_ms": 0,
                "llm_usage": {"total_tokens": 0, "message_count": 0, "agent_usage": []},
                "response_file": None,
                "visualization_file": None,
                "run_id": run_id
            }
    
    def with_timing(self, func_name: str, tags: Dict[str, str] = None) -> Callable:
        """
        Create a decorator for timing function execution.
        
        This is a convenience method for creating timing decorators.
        
        Args:
            func_name: Name to use for the timer
            tags: Additional tags for the timer
            
        Returns:
            A decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.metrics_collector:
                    return func(*args, **kwargs)
                    
                self.metrics_collector.start_timer(func_name, tags)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.metrics_collector.stop_timer(func_name)
            return wrapper
        return decorator
    
    def set_metrics_registry(self, metrics_registry):
        """Set the metrics registry for this agent system"""
        self.metrics_registry = metrics_registry
        if self.metrics_collector:
            self.metrics_collector.set_metrics_registry(metrics_registry)
        return self

    def record_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Record a metric if metrics registry is available"""
        if self.metrics_collector:
            self.metrics_collector.record_metric(metric_name, value, tags or {})
        elif self.metrics_registry:
            collector = self.metrics_registry.get_collector("system")
            if collector:
                collector.collect_point(metric_name, value, tags or {})

    def generate_run_id(self) -> str:
        """Generate a unique run ID"""
        return str(uuid.uuid4())

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent system"""
        return {
            "name": self.name, 
            "type": self.__class__.__name__, 
            "config": self.config,
            "evaluator": self.evaluator.name if self.evaluator else None
        }



class AgentSystemRegistry:
    """Registry for agent systems available in the benchmark"""

    _registry = {}
    _initialized = False

    @classmethod
    def _import_agent_systems(cls):
        """Import all agent system modules to trigger their registration."""
        if cls._initialized:
            return

        # Import the package to trigger registration
        cls._initialized = True

    @classmethod
    def register(cls, name: str, agent_class, **default_config):
        """Register an agent system class with the registry."""
        cls._registry[name] = {
            "class": agent_class,
            "default_config": default_config
        }

    @classmethod
    def get(cls, name: str, config: Dict[str, Any] = None) -> Optional[AgentSystem]:
        """Get an instance of an agent system by name."""
        cls._import_agent_systems()
        if name not in cls._registry:
            raise KeyError(f"Agent system '{name}' not found. Available: {', '.join(cls.get_available_system_names())}")
        
        agent_info = cls._registry[name]
        agent_config = {**agent_info["default_config"], **(config or {})}
        return agent_info["class"](name=name, config=agent_config)

    @classmethod
    def list_available(cls) -> Dict[str, Any]:
        """List all available agent systems and their descriptions."""
        cls._import_agent_systems()
        return {name: info["default_config"].get("description", "") 
                for name, info in cls._registry.items()}

    @classmethod
    def get_available_system_names(cls) -> list[str]:
        """Returns a list of names of all registered agent systems."""
        cls._import_agent_systems()
        return sorted(cls._registry.keys())

    @classmethod
    def get_all_systems(cls) -> Dict[str, Dict[str, Any]]:
        """Returns the complete dictionary of all registered agent systems."""
        cls._import_agent_systems()
        return cls._registry


def create_agent_system(name: str, config: Dict[str, Any] = None) -> Optional[AgentSystem]:
    """
    Create an agent system by name.
    If 'use_tools' or 'use_mcp_tools' is in the config, the agent system
    will be wrapped with ToolIntegrationWrapper to enable tool use.
    """
    AgentSystemRegistry._import_agent_systems()
    config = config or {}

    # Get an instance of the agent system from the registry
    try:
        agent_system = AgentSystemRegistry.get(name, config=config)
    except KeyError:
        return None

    if not agent_system:
        return None
    # Check if any tool integration should be applied
    if config and (config.get("use_tools") or config.get("use_mcp_tools")):
        try:
            from mas_arena.tools.tool_integration import ToolIntegrationWrapper
            # Extract MCP server config and mock flag
            mcp_servers = config.get("mcp_servers", {})
            mock_mode = config.get("mock_mcp", False)

            # Wrap the agent system
            # The wrapper will now be responsible for initializing the ToolManager
            agent_system = ToolIntegrationWrapper(agent_system, mcp_servers, mock=mock_mode)

        except ImportError as e:
            import traceback
            print("Warning: ToolIntegrationWrapper not found. Tool integration is disabled.")
            print("Underlying ImportError:")
            traceback.print_exc()


    return agent_system

