"""
Centralized Metrics Collection

This module provides a centralized metrics collection system that unifies
how metrics are recorded across the mas_arena.
"""

import time
from typing import Dict, Any, Callable


class MetricsCollector:
    """
    Centralized metrics collection system that handles all metric recording.
    
    This class provides a unified interface for recording metrics across different
    components of the benchmark system. It acts as a facade over various metric
    collectors (system, agent, inter_agent) and ensures consistent metric recording.
    
    Attributes:
        metrics_registry: The metrics registry containing individual collectors
        timers: Dictionary of active timers for measuring durations
        callback: Optional callback function for metrics
    """
    
    def __init__(self, metrics_registry=None, callback: Callable = None):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_registry: The metrics registry to use
            callback: Optional callback function for metrics
        """
        self.metrics_registry = metrics_registry
        self.timers = {}
        self.callback = callback
        self.disabled = False
    
    def set_metrics_registry(self, metrics_registry):
        """
        Set the metrics registry.
        
        Args:
            metrics_registry: The metrics registry to use
            
        Returns:
            Self for method chaining
        """
        self.metrics_registry = metrics_registry
        return self
    
    def disable(self):
        """Disable metrics collection"""
        self.disabled = True
        return self
    
    def enable(self):
        """Enable metrics collection"""
        self.disabled = False
        return self
    
    def record_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """
        Record a generic metric.
        
        This is the core method for recording any kind of metric. Other specialized
        methods ultimately call this method.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            tags: Additional tags for the metric
            
        Returns:
            Recorded value
        """
        if self.disabled or not self.metrics_registry:
            return value
            
        collector = self.metrics_registry.get_collector("system")
        if collector:
            collector.collect_point(metric_name, value, tags or {})
            
        if self.callback:
            self.callback(metric_name, value, tags or {})
            
        return value
    
    # Timer methods
    
    def start_timer(self, timer_name: str, tags: Dict[str, str] = None):
        """
        Start a timer for measuring durations.
        
        Args:
            timer_name: Name of the timer
            tags: Additional tags for the timer
            
        Returns:
            Self for method chaining
        """
        self.timers[timer_name] = {
            "start_time": time.time(),
            "tags": tags or {}
        }
        
        # Record the start event
        self.record_metric(
            f"{timer_name}.start", 
            1.0,
            tags or {}
        )
        
        return self
    
    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a timer and record the duration.
        
        Args:
            timer_name: Name of the timer
            
        Returns:
            Duration in milliseconds
        """
        if timer_name not in self.timers:
            return 0.0
            
        timer_info = self.timers.pop(timer_name)
        duration_ms = (time.time() - timer_info["start_time"]) * 1000
        
        # Record the duration
        self.record_metric(
            f"{timer_name}.duration_ms", 
            duration_ms,
            timer_info["tags"]
        )
        
        # Record the completion event
        self.record_metric(
            f"{timer_name}.complete", 
            1.0,
            timer_info["tags"]
        )
        
        return duration_ms
    
    def measure_execution(self, func: Callable, timer_name: str = None, tags: Dict[str, str] = None):
        """
        Measure the execution time of a function.
        
        Args:
            func: Function to measure
            timer_name: Name of the timer (defaults to function name)
            tags: Additional tags
            
        Returns:
            A decorated function that measures execution time
        """
        if not timer_name:
            timer_name = func.__name__
            
        def wrapper(*args, **kwargs):
            self.start_timer(timer_name, tags)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.stop_timer(timer_name)
                
        return wrapper
    
    # LLM metrics
    
    def record_llm_usage(
        self, 
        agent_id: str, 
        model_name: str, 
        prompt_tokens: int = 0,
        completion_tokens: int = 0, 
        reasoning_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0.0,
        input_token_details: Dict[str, int] = None,
        output_token_details: Dict[str, int] = None,
        tags: Dict[str, str] = None
    ):
        """
        Record LLM usage metrics.
        
        Args:
            agent_id: ID of the agent
            model_name: Name of the LLM model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            reasoning_tokens: Number of reasoning tokens
            total_tokens: Total number of tokens (if not calculated from prompt+completion)
            latency_ms: Latency in milliseconds
            input_token_details: Detailed breakdown of input token counts
            output_token_details: Detailed breakdown of output token counts
            tags: Additional tags
            
        Returns:
            Self for method chaining
        """
        if self.disabled or not self.metrics_registry:
            return self
            
        # Get the agent collector
        agent_collector = self.metrics_registry.get_collector("agent")
        if not agent_collector:
            return self
            
        # Calculate total tokens if not provided
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens
            
        # Record detailed metrics
        agent_collector.record_llm_usage(
            agent_id=agent_id,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            tags=tags or {}
        )
        
        # Also record summary metrics
        metrics = {
            f"llm.tokens.{agent_id}": total_tokens,
            f"llm.tokens.prompt.{agent_id}": prompt_tokens,
            f"llm.tokens.completion.{agent_id}": completion_tokens,
            f"llm.tokens.reasoning.{agent_id}": reasoning_tokens,
            f"llm.latency_ms.{agent_id}": latency_ms
        }
        
        # Add detailed token metrics if provided
        input_details = input_token_details
        if hasattr(input_token_details, 'model_dump'):
            input_details = input_token_details.model_dump()

        if input_details:
            for detail_type, count in input_details.items():
                metrics[f"llm.tokens.input.{detail_type}.{agent_id}"] = count
                
        output_details = output_token_details
        if hasattr(output_token_details, 'model_dump'):
            output_details = output_token_details.model_dump()
            
        if output_details:
            for detail_type, count in output_details.items():
                metrics[f"llm.tokens.output.{detail_type}.{agent_id}"] = count
        
        for metric_name, value in metrics.items():
            self.record_metric(
                metric_name, 
                value,
                {
                    "agent_id": agent_id,
                    "model": model_name,
                    **(tags or {})
                }
            )
            
        return self
    
    def record_agent_interaction(
        self, 
        from_agent: str, 
        to_agent: str, 
        message_type: str,
        content: str = "",
        tags: Dict[str, str] = None
    ):
        """
        Record an interaction between agents.
        
        Args:
            from_agent: ID of the sending agent
            to_agent: ID of the receiving agent
            message_type: Type of message
            content: Message content (optional)
            tags: Additional tags
            
        Returns:
            Self for method chaining
        """
        if self.disabled or not self.metrics_registry:
            return self
            
        inter_agent_collector = self.metrics_registry.get_collector("inter_agent")
        if inter_agent_collector:
            inter_agent_collector.record_interaction(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                tags=tags or {}
            )
            
        # Record a summary metric
        self.record_metric(
            f"agent.interaction.{from_agent}.{to_agent}", 
            1.0,
            {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "message_type": message_type,
                **(tags or {})
            }
        )
        
        # Record content length if provided
        if content:
            self.record_metric(
                f"agent.interaction.content_length.{from_agent}.{to_agent}", 
                len(content),
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "message_type": message_type,
                    **(tags or {})
                }
            )
            
        return self
    
    def record_evaluation_result(
        self, 
        problem_id: str, 
        score: float, 
        duration_ms: float,
        metrics: Dict[str, Any] = None,
        tags: Dict[str, str] = None
    ):
        """
        Record an evaluation result.
        
        Args:
            problem_id: ID of the problem
            score: Evaluation score
            duration_ms: Duration in milliseconds
            metrics: Additional metrics to record
            tags: Additional tags
            
        Returns:
            Self for method chaining
        """
        if self.disabled or not self.metrics_registry:
            return self
            
        system_collector = self.metrics_registry.get_collector("system")
        if system_collector:
            # Record the score
            system_collector.collect_point(
                "evaluation.score", 
                score,
                {
                    "problem_id": problem_id,
                    **(tags or {})
                }
            )
            
            # Record the duration
            system_collector.record_latency(
                "evaluation.duration", 
                duration_ms,
                {
                    "problem_id": problem_id, 
                    **(tags or {})
                }
            )
            
            # Record pass/fail
            system_collector.collect_point(
                "evaluation.passed", 
                1.0 if score == 1 else 0.0,
                {
                    "problem_id": problem_id, 
                    **(tags or {})
                }
            )
            
        # Record additional metrics if provided
        if metrics:
            for metric_name, value in metrics.items():
                self.record_metric(
                    f"evaluation.{metric_name}", 
                    value,
                    {
                        "problem_id": problem_id, 
                        **(tags or {})
                    }
                )
                
        return self
                
    def record_error(self, error_type: str, message: str, tags: Dict[str, str] = None):
        """
        Record an error.
        
        Args:
            error_type: Type of error
            message: Error message
            tags: Additional tags
            
        Returns:
            Self for method chaining
        """
        self.record_metric(
            f"error.{error_type}", 
            1.0,
            {
                "error_message": message[:100],  # Truncate long messages
                **(tags or {})
            }
        )
        
        return self
    
    def record_system_metrics(self, metrics: Dict[str, Any], tags: Dict[str, str] = None):
        """
        Record multiple system metrics at once.
        
        Args:
            metrics: Dictionary of metric name to value
            tags: Additional tags
            
        Returns:
            Self for method chaining
        """
        for metric_name, value in metrics.items():
            self.record_metric(f"system.{metric_name}", value, tags)
            
        return self 