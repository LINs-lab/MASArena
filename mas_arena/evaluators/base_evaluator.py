#!/usr/bin/env python3
"""
Base Evaluator

This module provides the base class for all evaluators in the benchmark framework.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABCMeta

from mas_arena.core.registry import COMPONENT_REGISTRY

class BaseEvaluator(metaclass=ABCMeta):
    """
    Abstract base class for evaluators.
    Each evaluator is responsible for scoring an agent's performance on a specific mas_arena.
    """
    # All evaluators are considered thread-safe by default.
    # Evaluators that are NOT thread-safe should override this to False.
    SUPPORTS_CONCURRENCY = True

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the base evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration dictionary for the evaluator
        """
        self.name = name
        self.config = config or {}
        
        # Set up data and log paths
        self.data_path = config.get("data_path", f"data/{name}_test.jsonl")
        self.log_path = config.get("log_path", f"data/results/{name.upper()}")
        
        # Set up logging
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(f"{self.log_path}/{name}_eval.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{name}_evaluator")
        
        # Dataset
        self.dataset = []
        self._train_data: Optional[List[dict]] = None
        self._dev_data: Optional[List[dict]] = None
        self._test_data: Optional[List[dict]] = None

        self._load_data()

    def _load_dateset_from_path(self, path: str):
        """Load dataset from a specified path."""
        if not path or not isinstance(path, str) or not os.path.exists(path):
            raise ValueError(f"Invalid or non-existent path: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except Exception as e:
            self.logger.error(f"Failed to load dataset from {path}: {e}")
            return []

    def _load_dataset(self):
        """Load the dataset for evaluation."""
        return self._load_dateset_from_path(self.data_path)

    def _load_data(self):
        """Load data for the evaluator."""
        pass

    
    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Evaluate an agent's solution to a problem.
        
        Args:
            problem: Problem dictionary
            **kwargs: Additional arguments
            
        Returns:
            Evaluation results dictionary
        """
        raise NotImplementedError("Subclasses must implement the evaluate method")
    
    def batch_evaluate(self, problems: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of problems.
        
        Args:
            problems: List of problem dictionaries
            **kwargs: Additional arguments
            
        Returns:
            List of evaluation results dictionaries
        """
        results = []
        for problem in problems:
            result = self.evaluate(problem, **kwargs)
            results.append(result)
        return results
    
    def verify_answer(self, prediction: str, reference: Union[str, Dict[str, Any]]) -> bool:
        """
        Verify if a prediction is correct according to the reference.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer (string or dictionary)
            
        Returns:
            True if the prediction is correct, False otherwise
        """
        raise NotImplementedError("Subclasses must implement the verify_answer method")
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Optional[str] = None) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: List of evaluation results dictionaries
            output_path: Optional path to save results to
        """
        path = output_path or os.path.join(self.log_path, f"{self.name}_results.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        self.logger.info(f"Saved {len(results)} evaluation results to {path}")

    async def async_evaluate(self, configured_graph, example, i):
        pass