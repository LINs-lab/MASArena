from .workflow_graph import WorkFlowGraph
import regex
from typing import List

VALID_SCHEMAS = ["python", "yaml"]


class WorkFlowSchema:
    """
    The Schema of the WorkFlow
    """

    def __init__(self, graph: WorkFlowGraph, **kwargs):
        self.graph = graph
        self.kwargs = kwargs

    def convert_to_schema(self, schema_type: str) -> str:
        """
        Convert the workflow graph to the specified schema type.

        Parameters:
        - schema_type (str): The type of schema to convert to (e.g., 'python', 'yaml').

        Returns:
        - str: The workflow represented in the specified schema format.

        Raises:
        - ValueError: If the schema type is not supported.
        """
        if schema_type not in VALID_SCHEMAS:
            raise ValueError(f"Unsupported schema type: {schema_type}. Supported types are: {VALID_SCHEMAS}")

        match schema_type:
            case "python":
                repr_str = self.get_workflow_python_repr()
            case "yaml":
                repr_str = self.get_workflow_yaml_repr()
            case _:
                pass
        return repr_str

    def _get_workflow_repr_info(self) -> List[dict]:
        """
        Get the information for the workflow representation.
        """
        info = []
        for node in self.graph.nodes:
            task_name = node.name
            input_names = [param.name for param in node.inputs]
            output_names = [param.name for param in node.outputs]
            task_info = {
                "task_name": task_name,
                "input_names": input_names,
                "output_names": output_names
            }
            info.append(task_info)
        return info

    def _convert_to_func_name(self, name: str) -> str:
        """
        Convert the task name to the function name.
        """
        name = name.lower().strip()
        name = name.replace(' ', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        # Replace multiple consecutive underscores with a single underscore
        name = regex.sub(r'_+', "_", name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name

    def get_workflow_yaml_repr(self) -> str:
        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""

        yaml_workflow_info = []
        for task_info in repr_info:
            name = self._convert_to_func_name(task_info['task_name'])
            input_names = "\n".join([f'    - {input_name}' for input_name in task_info['input_names']])
            output_names = "\n".join([f'    - {output_name}' for output_name in task_info['output_names']])
            yaml_workflow_info.append(
                "- name: {name}\n  args:\n{input_names}\n  outputs:\n{output_names}".format(
                    name=name,
                    input_names=input_names,
                    output_names=output_names
                )
            )
        yaml_workflow_repr = "\n\n".join(yaml_workflow_info)
        return yaml_workflow_repr

    def get_workflow_python_repr(self) -> str:
        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""

        python_workflow_info = []
        for task_info in repr_info:
            name = self._convert_to_func_name(task_info['task_name'])
            input_names = [f'{input_name}' for input_name in task_info['input_names']]
            output_names = [f'{output_name}' for output_name in task_info['output_names']]
            python_workflow_info.append(
                "{{'name': '{name}', 'args': {args}, 'outputs': {outputs}}}".format(
                    name=name,
                    args=input_names,
                    outputs=output_names
                )
            )
        python_workflow_repr = "steps = [\n" + ",\n".join(python_workflow_info) + "\n]"
        return python_workflow_repr
