from typing import Any, Dict, List, Optional
import logging
import inspect
from smolagents import Tool

logger = logging.getLogger(__name__)


class ToolManager:
    """工具管理器"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """注册工具"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """获取所有工具"""
        return list(self.tools.values())

    def _tool_to_openai_schema(self, tool: Tool) -> Dict[str, Any]:
        """
        Convert a smolagents.Tool into OpenAI Tools API schema.

        smolagents tools usually define:
        - tool.name (str)
        - tool.description (str)
        - tool.inputs (dict[str, dict])  # each input contains type/description/nullable
        """
        # Prefer a pre-built OpenAI schema if tool.to_dict() already returns it.
        try:
            raw_schema = tool.to_dict()
            if (
                isinstance(raw_schema, dict)
                and raw_schema.get("type") == "function"
                and isinstance(raw_schema.get("function"), dict)
                and isinstance(raw_schema["function"].get("parameters"), dict)
            ):
                return raw_schema
        except Exception:
            raw_schema = None

        inputs: Dict[str, Dict[str, Any]] = getattr(tool, "inputs", {}) or {}
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for k, v in inputs.items():
            if not isinstance(v, dict):
                v = {"type": "string", "description": str(v)}
            prop = dict(v)
            # OpenAI expects JSON Schema 'type', 'description', etc under parameters.properties.
            # Keep nullable if present (OpenAI tolerates it), but also use it to compute required.
            properties[k] = prop
            if not prop.get("nullable", False):
                required.append(k)

        function_def: Dict[str, Any] = {
            "name": getattr(tool, "name", tool.__class__.__name__),
            "description": (getattr(tool, "description", "") or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        }
        if required:
            function_def["parameters"]["required"] = required

        return {"type": "function", "function": function_def}

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取所有工具的OpenAI函数调用格式"""
        return [self._tool_to_openai_schema(tool) for tool in self.tools.values()]

    def _normalize_tool_kwargs(self, tool: Tool, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort compatibility layer for historical argument names produced by prompts/models.
        Examples seen in logs:
        - inspect_file_as_csv: sometimes uses `file` instead of `file_path`
        - extract_text_content: sometimes uses `file` instead of `file_path`
        - python_interpreter: sometimes uses `input`/`python_code` instead of `code`
        """
        if not kwargs:
            return kwargs

        # Some callers may wrap args in {"arguments": {...}}
        if isinstance(kwargs.get("arguments"), dict) and len(kwargs) == 1:
            kwargs = dict(kwargs["arguments"])
        else:
            kwargs = dict(kwargs)

        alias_map = {
            # Common file path aliases
            "file": "file_path",
            "filepath": "file_path",
            "path": "file_path",
            "filename": "file_path",
            # Python code aliases
            "input": "code",
            "python_code": "code",
            "script": "code",
            "command": "code",
        }

        normalized: Dict[str, Any] = {}
        for k, v in kwargs.items():
            nk = alias_map.get(k, k)
            # Do not overwrite explicit canonical keys.
            if nk in normalized and k != nk:
                continue
            normalized[nk] = v

        # Drop unexpected kwargs if tool.forward doesn't accept **kwargs
        try:
            sig = inspect.signature(tool.forward)
            params = sig.parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if not accepts_kwargs:
                allowed = {
                    name
                    for name, p in params.items()
                    if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                }
                normalized = {k: v for k, v in normalized.items() if k in allowed}
        except Exception:
            # If introspection fails, keep best-effort normalized kwargs.
            pass

        return normalized

    def execute_tool(self, name: str, *args, **kwargs) -> str:
        """执行工具"""
        tool = self.get_tool(name)
        if not tool:
            return f"Tool '{name}' not found"

        try:
            normalized_kwargs = self._normalize_tool_kwargs(tool, kwargs)
            return tool.forward(*args, **normalized_kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {str(e)}"