"""
Multi-Agent System Metrics Collection Framework.

This package provides comprehensive metrics collection and analysis for LangGraph-based
multi-agent systems across system, agent, and inter-agent dimensions.

Modules:
    system_metrics: System-level performance and resource utilization metrics
    agent_metrics: Individual agent performance and behavior metrics
    inter_agent_metrics: Communication and coordination metrics between agents
    collectors: Unified metrics collection framework components
    unified_evaluator: Unified evaluation framework for comparing agent systems
    collector: Centralized metrics collection system
"""

from mas_arena.metrics.collectors import (
    MetricsRegistry,
    MetricsCollectionConfig,
    BaseMetricsCollector
)

from mas_arena.metrics.collector import MetricsCollector

__all__ = [
    'MetricsRegistry',
    'MetricsCollectionConfig',
    'BaseMetricsCollector',
    'MetricsCollector',
] 