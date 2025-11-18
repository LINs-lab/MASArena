import threading
from collections import defaultdict
from enum import Enum
from typing import Optional, List, Union, Dict
import networkx as nx
from networkx.classes import MultiDiGraph
from pydantic import Field

from mas_arena.core_serializer.component import SerializableComponent
from .action_graph import ActionGraph
from mas_arena.core_serializer.parameter import Parameter
from mas_arena.utils.serialization_utils import generate_dynamic_class_name


class WorkFlowNodeState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkFlowNode(SerializableComponent):
    """
    Represents a node in the workflow graph.
    Each node can have multiple inputs and outputs, and can be associated with one or more agents.
    """

    name: str
    inputs: List[Parameter] = None
    outputs: List[Parameter] = None
    reason: Optional[str] = None
    agents: Optional[List[Union[str, dict]]] = None
    action_graph: Optional[ActionGraph] = None
    status: Optional[WorkFlowNodeState] = WorkFlowNodeState.PENDING

    def __init__(self, name: str, inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None,
                 agents: Optional[List[str]] = None):
        super().__init__()
        self.name = name
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.agents = agents if agents is not None else []


class WorkFlowEdge(SerializableComponent):
    """
    Represents a directed edge in a workflow graph.

    Workflow edges connect tasks (nodes) in the workflow graph, establishing
    execution dependencies and data flow relationships. Each edge has a source
    node, target node, and optional priority to influence execution order.

    Attributes:
        source: Name of the source node (where the edge starts)
        target: Name of the target node (where the edge ends)
        priority: Numeric priority value for this edge (higher means higher priority)
    """
    source: str
    target: str
    priority: Optional[int] = 0

    def __init__(self, edge_tuple: Optional[tuple] = (), **kwargs):
        data = self.init_from_tuple(edge_tuple)
        data.update(kwargs)
        super().__init__(**kwargs)

    def init_from_tuple(self, edge_tuple: tuple) -> dict:
        if not edge_tuple:
            return {}
        keys = ["source", "target", "priority"]
        data = {k: v for k, v in zip(keys, edge_tuple)}
        return data


class WorkFlowGraph(SerializableComponent):
    goal: str
    nodes: Optional[List[WorkFlowNode]] = []
    edges: Optional[List[WorkFlowEdge]] = []
    graph: Optional[Union[MultiDiGraph, "WorkFlowGraph"]] = Field(default=None, exclude=True)

    def init_component(self):
        self._lock = threading.Lock()
        if not self.graph:
            self._init_from_nodes_and_edges(self.nodes, self.edges)
        elif isinstance(self.graph, MultiDiGraph):
            self._init_from_multidigraph(self.graph, self.nodes, self.edges)
        elif isinstance(self.graph, WorkFlowGraph):
            self._init_from_workflowgraph(self.graph, self.nodes, self.edges)
        else:
            raise TypeError(
                f"{type(self.graph)} is an unknown type for graph. Supported types: [MultiDiGraph, WorkFlowGraph]")
        self._validate_workflow_structure()
        self.update_graph()

    def update_graph(self):
        self._loops = self._find_all_loops()

    def node_exists(self, node: Union[str, WorkFlowNode]) -> bool:
        if isinstance(node, str):
            return node in self.graph.nodes
        elif isinstance(node, WorkFlowNode):
            return node.name in self.graph.nodes
        else:
            raise Exception(f"Unknown node type {type(node)}, node must be a str or WorkFlowNode instance")

    def add_node(self, node: WorkFlowNode, update_graph: bool = True, **kwargs):
        if not isinstance(node, WorkFlowNode):
            raise TypeError(f"Expected WorkFlowNode, got {type(node)}")
        if self.node_exists(node.name):
            raise ValueError(f"Node {node.name} already exists in the graph")

        self.nodes.append(node)
        self.graph.add_node(node.name, ref=node)
        if update_graph:
            self.update_graph()

    def get_node(self, node_name: str) -> WorkFlowNode:
        if not self.node_exists(node_name):
            raise ValueError(f"Node {node_name} does not exist in the graph")
        return self.graph.get_node(node_name)

    def add_nodes(self, *nodes: WorkFlowNode, update_graph: bool = True, **kwargs):
        nodes: list = list(nodes)
        nodes.extend([kwargs.pop(var) for var in ["node", "nodes"] if var in kwargs])

        for node in nodes:
            if isinstance(node, (tuple, list)):
                for n in node:
                    self.add_node(n, update_graph=update_graph, **kwargs)
                else:
                    self.add_nodes(node, update_graph=update_graph, **kwargs)

        def _init_from_nodes_and_edges(self, nodes: List[WorkFlowNode] = [], edges: List[WorkFlowEdge] = []):
            """
            Initialize the workflow graph from nodes and edges.
            """
            self.nodes = nodes
            self.edges = edges
            self.graph = MultiDiGraph()
            self.add_nodes(*nodes, update_graph=False)
            self.add_edges(*edges, update_graph=False)

        def _validate_workflow_structure(self):
            isolated_nodes = list(nx.isolates(self.graph))
            if len(self.graph.nodes) > 1 and isolated_nodes:
                print(f"The workflow contains isolated nodes: {isolated_nodes}")

            initial_nodes = self.find_initial_nodes()
            if len(self.graph.nodes) > 1 and not initial_nodes:
                error_message = "There are no initial nodes in the workflow!"
                print(error_message)
                raise ValueError(error_message)

            end_nodes = self.find_end_nodes()
            if len(self.graph.nodes) > 1 and not end_nodes:
                print("There are no end nodes in the workflow")

        def find_initial_nodes(self) -> List[str]:
            initial_nodes = [node for node, in_degree in self.graph.in_degree() if in_degree == 0]
            return initial_nodes

        def find_end_nodes(self) -> List[str]:
            end_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree == 0]
            return end_nodes

        def _find_loops(self, start_node: Union[str, WorkFlowNode]) -> Dict[str, List]:
            if isinstance(start_node, str):
                start_node = self.get_node(node_name=start_node)
            start_node_name = start_node.name

            loops = defaultdict(list)

            def dfs(current_node_name: str, path: List[str]):
                ## end condition
                if current_node_name in path:
                    loops[current_node_name].append(path[path.index(current_node_name):])
                    return
                path.append(current_node_name)
                children = self.get_node_children(current_node_name)

        def get_node_children(self, node: Union[str, WorkFlowNode]) -> List[str]:
            node_name = node if isinstance(node, str) else node.name
            if not self.node_exists(node=node):
                raise ValueError(f"Node {node_name} does not exist in the graph")
            children = list(self.graph.successors(node_name))
            return children

        def get_node_predecessors(self, node: Union[str, WorkFlowNode]) -> List[str]:
            node_name = node if isinstance(node, str) else node.name
            if not self.node_exists(node=node):
                raise ValueError(f"Node {node_name} does not exist in the graph")
            predecessors = list(self.graph.predecessors(node_name))
            return predecessors

        def get_uncomplete_initial_nodes(self) -> List[str]:
            initial_nodes = self.find_initial_nodes()
            judge_initial_nodes_completed = [self.get_node(node_name).is_complete for node_name in initial_nodes]
            uncompleted_initial_nodes = []
            for node_name, is_complete in zip(initial_nodes, judge_initial_nodes_completed):
                if not is_complete:
                    uncompleted_initial_nodes.append(node_name)
            return uncompleted_initial_nodes

            def __init__(self, goal: str, tasks: List[dict], **kwargs):
                nodes = self._infer_nodes_from_tasks(tasks)
                edges = self._infer_edges_from_tasks(tasks)
                super().__init__(goal=goal, nodes=nodes, edges=edges, **kwargs)

            def _infer_nodes_from_tasks(self, tasks: List[dict]) -> List[WorkFlowNode]:
                nodes = [self._infer_node_from_task(task) for task in tasks]
                return nodes

            def _infer_node_from_task(self, task: dict) -> WorkFlowNode:

                node_name = task.get("name", None)
                if not node_name:
                    raise ValueError("Task must have a 'name' field")
                node_description = task.get("description", None)
                if not node_description:
                    raise ValueError("Task must have a 'description' field")
                agent_prompt = task.get("prompt", None)
                agent_prompt_template = task.get("prompt_template", None)
                if not agent_prompt and not agent_prompt_template:
                    raise ValueError("Task must have either 'prompt' or 'prompt_template' field")

                inputs = task.get("inputs", [])
                outputs = task.get("outputs", [])
                agent_name = generate_dynamic_class_name(node_name + " Agent")
                agent_description = node_description
                agent_system_prompt = task.get("system_prompt", "You are a helpful and highly intelligent assistant.")
                agent_output_parser = task.get("output_parser", None)
                agent_parse_mode = task.get("parse_mode", "str")
                agent_parse_func = task.get("parse_func", None)
                agent_parse_title = task.get("parse_title", None)

                node = WorkFlowNode.from_dict(
                    {
                        "name": node_name,
                        "description": node_description,
                        "inputs": inputs,
                        "outputs": outputs,
                        "agents": [
                            {
                                "name": agent_name,
                                "description": agent_description,
                                "prompt": agent_prompt,
                                "prompt_template": agent_prompt_template,
                                "system_prompt": agent_system_prompt,
                                "inputs": inputs,
                                "outputs": outputs,
                                "output_parser": agent_output_parser,
                                "parse_mode": agent_parse_mode,
                                "parse_func": agent_parse_func,
                                "parse_title": agent_parse_title
                            }
                        ]
                    }
                )
                return node

            def get_graph_info(self, **kwargs) -> dict:
                config = {
                    "class_name": self.__class__.__name__,
                    "goal": self.goal,
                    "tasks": [
                        {
                            "name": node.name,
                            "description": node.description,
                            "inputs": [param.to_dict(ignore=["class_name"]) for param in node.inputs],
                            "outputs": [param.to_dict(ignore=["class_name"]) for param in node.outputs],
                            "prompt": node.agents[0].get("prompt", None),
                            "prompt_template": node.agents[0].get("prompt_template", None).to_dict() if node.agents[
                                0].get(
                                "prompt_template", None) else None,
                            "system_prompt": node.agents[0].get("system_prompt", None),
                            "parse_mode": node.agents[0].get("parse_mode", "str"),
                            "parse_func": node.agents[0].get("parse_func", None).__name__ if node.agents[0].get(
                                "parse_func", None) else None,
                            "parse_title": node.agents[0].get("parse_title", None)
                        }
                        for node in self.nodes
                    ]
                }

                return config
