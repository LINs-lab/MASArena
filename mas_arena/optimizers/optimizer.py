from typing import Optional, Union

from mas_arena.core.component import SerializableComponent
from mas_arena.workflow.action_graph import ActionGraph
from mas_arena.workflow.workflow_graph import WorkFlowGraph


class Optimizer(SerializableComponent):

    """Base class for optimizers in the MAS Arena framework."""

    def optimize(self, dataset: str, **kwargs):
        """
        Optimize the workflow.
        """
        raise NotImplementedError(f"``optimize`` function for {type(self).__name__} is not implemented!")

    def step(self, **kwargs):
        """
        Take a step of optimization.
        """
        raise NotImplementedError(f"``step`` function for {type(self).__name__} is not implemented!")

    def evaluate(self, dataset: str, eval_mode: str = "test", graph: Optional[Union[WorkFlowGraph, ActionGraph]] = None,
                 **kwargs) -> dict:
        raise NotImplementedError(f"``evaluate`` function for {type(self).__name__} is not implemented!")

    def convergence_check(self, *args, **kwargs) -> bool:
        """
        Check if the optimization has converged.
        """
        raise NotImplementedError(f"``convergence_check`` function for {type(self).__name__} is not implemented!")
