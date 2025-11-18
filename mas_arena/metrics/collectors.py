"""
Unified Metrics Collection Framework for Multi-Agent Systems.

This module provides the core infrastructure for collecting, aggregating,
and managing metrics across the benchmark framework.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
import time
import queue
from datetime import datetime
import threading
import json
import os
import csv
import copy
import random


@dataclass
class MetricsCollectionConfig:
    """Configuration for metrics collection behavior."""
    
    # Sampling frequency in milliseconds
    sampling_interval_ms: int = 1000
    
    # Queue configuration
    metrics_queue_size: int = 10000
    metrics_batch_size: int = 100
    metrics_flush_interval_ms: int = 500
    
    # Whether to collect each category of metrics
    collect_system_metrics: bool = True
    collect_agent_metrics: bool = True
    collect_inter_agent_metrics: bool = True
    collect_llm_metrics: bool = True
    collect_tool_metrics: bool = True
    collect_memory_metrics: bool = True
    
    # Storage configuration
    metrics_storage_path: Optional[str] = None
    in_memory_retention_seconds: int = 3600
    
    # Output formats
    enable_prometheus_export: bool = False
    enable_json_export: bool = True
    
    # Filtering options
    included_metric_names: Set[str] = field(default_factory=set)
    excluded_metric_names: Set[str] = field(default_factory=set)
    
    # Aggregation settings
    default_aggregation_window_seconds: int = 60
    
    # Sampling rate (1.0 = collect all, 0.5 = collect 50%, etc.)
    sampling_rate: float = 1.0


class BaseMetricsCollector(ABC):
    """Base class for all metrics collectors in the framework."""
    
    def __init__(self, config: Optional[MetricsCollectionConfig] = None):
        """
        Initialize the metrics collector.
        
        Args:
            config: Configuration for metrics collection behavior
        """
        self.config = config or MetricsCollectionConfig()
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._collection_thread = None
        self._metrics_processor_thread = None
        self._stop_collection = False
        
        # Initialize metrics processing queue
        self._metrics_queue = queue.Queue(maxsize=self.config.metrics_queue_size)
        
    def start_collection(self) -> None:
        """Start collecting metrics."""
        if self._collection_thread is not None and self._collection_thread.is_alive():
            return
        
        self._stop_collection = False
        
        # Start the metrics processor thread
        self._metrics_processor_thread = threading.Thread(
            target=self._process_metrics_queue, 
            daemon=True,
            name=f"{self.__class__.__name__}_processor"
        )
        self._metrics_processor_thread.start()
        
        # Start the regular collection thread
        self._collection_thread = threading.Thread(
            target=self._collection_loop, 
            daemon=True,
            name=f"{self.__class__.__name__}_collector"
        )
        self._collection_thread.start()
        
        print(f"Started metrics collection threads for {self.__class__.__name__}")
    
    def stop_collection(self) -> None:
        """Stop collecting metrics."""
        self._stop_collection = True
        
        # Wait for collection thread to stop
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=2.0)
            
        # Wait for processor thread to process remaining items
        if self._metrics_processor_thread and self._metrics_processor_thread.is_alive():
            # Give it a chance to process remaining items
            time.sleep(1.0)
            self._metrics_processor_thread.join(timeout=2.0)
    
    def _collection_loop(self) -> None:
        """Background thread that collects metrics at regular intervals."""
        pass  # To be implemented by subclasses
    
    def _process_metrics_queue(self) -> None:
        """Process metrics from the queue in batches."""
        while not self._stop_collection:
            try:
                # Process metrics in batches for efficiency
                batch = []
                batch_size = min(self._metrics_queue.qsize(), self.config.metrics_batch_size)
                
                # Get up to batch_size items from the queue
                for _ in range(batch_size):
                    try:
                        item = self._metrics_queue.get_nowait()
                        batch.append(item)
                    except queue.Empty:
                        break
                
                # Process the batch if not empty
                if batch:
                    self._process_metrics_batch(batch)
                    
                    # Mark tasks as done
                    for _ in range(len(batch)):
                        self._metrics_queue.task_done()
                        
                # Sleep for a short interval to avoid busy-waiting
                # If the queue is empty, sleep longer to reduce CPU usage
                if not batch:
                    time.sleep(self.config.metrics_flush_interval_ms / 1000)
                else:
                    time.sleep(0.001)  # 1ms sleep when actively processing
                    
            except Exception as e:
                print(f"Error in metrics queue processing: {str(e)}")
                time.sleep(0.1)  # Sleep a bit longer after an error
    
    def _process_metrics_batch(self, batch: List[tuple]) -> None:
        """
        Process a batch of metrics.
        
        Args:
            batch: List of (metric_name, value, tags, timestamp) tuples
        """
        # Acquire lock once for the entire batch
        try:
            with self._lock:
                for metric_name, value, tags, timestamp in batch:
                    # Store in metrics dictionary
                    if metric_name not in self.metrics:
                        self.metrics[metric_name] = []
                    
                    self.metrics[metric_name].append({
                        'timestamp': timestamp,
                        'value': value,
                        'tags': tags
                    })
                
                # Prune old metrics after processing batch
                self._prune_old_metrics()
        except Exception as e:
            print(f"Error processing metrics batch: {str(e)}")
            
        # Notify hooks outside the lock to prevent deadlocks
        registry = MetricsRegistry()
        for metric_name, value, tags, _ in batch:
            try:
                registry.notify_hooks(metric_name, value, tags)
            except Exception as e:
                print(f"Error notifying hooks for {metric_name}: {str(e)}")
    
    def collect_point(self, metric_name: str, value: Any, tags: Dict[str, str] = None) -> None:
        """
        Collect a single data point for a metric.
        
        Args:
            metric_name: Name of the metric to collect
            value: Value of the metric
            tags: Optional tags/dimensions for the metric
        """
        # Skip excluded metrics
        if (self.config.excluded_metric_names and 
            (metric_name in self.config.excluded_metric_names)):
            return
            
        # Skip metrics not explicitly included if inclusion filter is active
        if (self.config.included_metric_names and 
            metric_name not in self.config.included_metric_names):
            return
        
        # Apply sampling rate - randomly skip some metrics based on sampling_rate
        if self.config.sampling_rate < 1.0 and random.random() > self.config.sampling_rate:
            return
            
        tags = tags or {}
        timestamp = datetime.now().isoformat()
        
        # Add to queue without blocking
        try:
            # Use put_nowait to avoid blocking if queue is full
            self._metrics_queue.put_nowait((metric_name, value, tags, timestamp))
        except queue.Full:
            # If queue is full, log a warning and drop the metric
            print(f"WARNING: Metrics queue full, dropping metric: {metric_name}")
    
    def _prune_old_metrics(self) -> None:
        """Remove metrics that are older than the retention period."""
        if self.config.in_memory_retention_seconds <= 0:
            return
            
        try:
            retention_threshold = datetime.now().timestamp() - self.config.in_memory_retention_seconds
            
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = [
                    dp for dp in self.metrics[metric_name]
                    if datetime.fromisoformat(dp['timestamp']).timestamp() > retention_threshold
                ]
        except Exception as e:
            # Don't let pruning errors affect the main function
            print(f"WARNING: Error while pruning old metrics: {str(e)}")
    
    def get_metrics(self, 
                  metric_names: Optional[List[str]] = None, 
                  time_range: Optional[tuple] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve collected metrics.
        
        Args:
            metric_names: Optional list of metric names to retrieve
            time_range: Optional tuple of (start_time, end_time) to filter by
            
        Returns:
            Dictionary of metric names to lists of metric data points
        """
        result = {}
        
        # Wait for any pending metrics to be processed
        if hasattr(self, '_metrics_queue') and not self._metrics_queue.empty():
            try:
                # Give the processor a chance to catch up
                self._metrics_queue.join()
            except Exception as e:
                print(f"WARNING: Error while waiting for metrics queue to be processed: {str(e)}")
                # If joining fails, continue with what we have
                pass
        
        # Use a lock timeout to avoid deadlocks
        lock_acquired = self._lock.acquire(timeout=1.0)  # 1-second timeout
        if not lock_acquired:
            print("WARNING: Could not acquire lock for retrieving metrics. Returning partial result.")
            return result
            
        try:
            # Determine which metrics to include
            names_to_retrieve = metric_names if metric_names is not None else self.metrics.keys()
            
            for name in names_to_retrieve:
                if name not in self.metrics:
                    continue
                    
                # Apply time range filter if specified
                if time_range is not None:
                    start_time, end_time = time_range
                    result[name] = [
                        dp for dp in self.metrics[name]
                        if start_time <= datetime.fromisoformat(dp['timestamp']).timestamp() <= end_time
                    ]
                else:
                    result[name] = copy.deepcopy(self.metrics[name])
                    
        except Exception as e:
            print(f"ERROR in get_metrics: {str(e)}")
        finally:
            self._lock.release()  # Ensure the lock is always released
            
        return result
    
    def export_metrics(self, format: str, path: Optional[str] = None) -> None:
        """
        Export collected metrics in the specified format.
        
        Args:
            format: Format to export (json, csv, prometheus, etc.)
            path: Optional path to write the export to
        """
        # Wait for any pending metrics to be processed before exporting
        if hasattr(self, '_metrics_queue') and not self._metrics_queue.empty():
            try:
                # Give the processor a chance to catch up
                self._metrics_queue.join()
            except Exception:
                # If joining fails, continue with what we have
                pass
                
        format = format.lower()
        
        if path is None:
            path = self.config.metrics_storage_path
            
        if path is None:
            raise ValueError("No export path specified")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        metrics_data = self.get_metrics()
        
        if format == 'json':
            self._export_json(metrics_data, path)
        elif format == 'csv':
            self._export_csv(metrics_data, path)
        elif format == 'prometheus':
            self._export_prometheus(metrics_data, path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, metrics_data: Dict[str, List[Dict[str, Any]]], path: str) -> None:
        """Export metrics as JSON."""
        with open(path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _export_csv(self, metrics_data: Dict[str, List[Dict[str, Any]]], path: str) -> None:
        """Export metrics as CSV."""
        # Create a flattened representation for CSV
        rows = []
        
        for metric_name, data_points in metrics_data.items():
            for dp in data_points:
                row = {
                    'metric_name': metric_name,
                    'timestamp': dp['timestamp'],
                    'value': dp['value']
                }
                
                # Add tags as columns
                for tag_name, tag_value in dp['tags'].items():
                    row[f'tag_{tag_name}'] = tag_value
                
                rows.append(row)
        
        if not rows:
            return
            
        # Write to CSV
        with open(path, 'w', newline='') as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def _export_prometheus(self, metrics_data: Dict[str, List[Dict[str, Any]]], path: str) -> None:
        """Export metrics in Prometheus text format."""
        lines = []
        
        for metric_name, data_points in metrics_data.items():
            # Add metric type help
            lines.append(f"# HELP {metric_name} {metric_name}")
            lines.append(f"# TYPE {metric_name} gauge")
            
            for dp in data_points:
                # Format tags for Prometheus
                tag_str = ','.join([f'{k}="{v}"' for k, v in dp['tags'].items()])
                timestamp_ms = int(datetime.fromisoformat(dp['timestamp']).timestamp() * 1000)
                
                if tag_str:
                    lines.append(f"{metric_name}{{{tag_str}}} {dp['value']} {timestamp_ms}")
                else:
                    lines.append(f"{metric_name} {dp['value']} {timestamp_ms}")
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))


class MetricsRegistry:
    """
    Central registry for all metrics collectors in the system.
    
    Provides a unified interface for accessing and managing all metrics collectors.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsRegistry, cls).__new__(cls)
            cls._instance._collectors = {}
            cls._instance._hooks = []
        return cls._instance
    
    def register_collector(self, name: str, collector: BaseMetricsCollector) -> None:
        """
        Register a metrics collector with the registry.
        
        Args:
            name: Name to register the collector under
            collector: Collector instance to register
        """
        self._collectors[name] = collector
        
    def get_collector(self, name: str) -> Optional[BaseMetricsCollector]:
        """
        Get a collector by name.
        
        Args:
            name: Name of the collector to retrieve
            
        Returns:
            The collector instance or None if not found
        """
        return self._collectors.get(name)
    
    def register_hook(self, hook: Callable[[str, Any, Dict[str, str]], None]) -> None:
        """
        Register a hook that will be called for every metric collection.
        
        Args:
            hook: Callable that takes (metric_name, value, tags)
        """
        self._hooks.append(hook)
    
    def notify_hooks(self, metric_name: str, value: Any, tags: Dict[str, str]) -> None:
        """
        Notify all registered hooks of a metric collection.
        
        Args:
            metric_name: Name of the metric
            value: Value of the metric
            tags: Tags/dimensions for the metric
        """
        # Make a copy of hooks to avoid issues if hooks change during iteration
        hooks = list(self._hooks)
        
        for hook in hooks:
            try:
                hook(metric_name, value, tags)
            except Exception as e:
                # Prevent hook errors from affecting metric collection
                print(f"ERROR in metrics hook: {str(e)}")
    
    def start_all_collectors(self) -> None:
        """Start all registered collectors."""
        for collector in self._collectors.values():
            collector.start_collection()
    
    def stop_all_collectors(self) -> None:
        """Stop all registered collectors."""
        for collector in self._collectors.values():
            collector.stop_collection()
    
    def export_all(self, format: str, path: Optional[str] = None) -> None:
        """
        Export metrics from all collectors.
        
        Args:
            format: Format to export (json, csv, prometheus, etc.)
            path: Optional path to write the export to
        """
        if path and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            
        for name, collector in self._collectors.items():
            collector_path = path
            if os.path.isdir(path):
                collector_path = os.path.join(path, f"{name}.{format}")
            collector.export_metrics(format, collector_path) 