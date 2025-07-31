
import logging
import json
import traceback
import os
import re
import asyncio
from typing import Dict, List, Any, Optional, get_origin, get_args
import time
import importlib.util
import inspect
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from types import NoneType, UnionType

from mas_arena.mcp_collections.base import ActionCollection

logger = logging.getLogger(__name__)


def _get_pydantic_field_type(field: FieldInfo) -> str:
    """Gets the JSON schema type for a Pydantic field."""
    annotation = field.annotation
    if annotation is None or annotation is NoneType:
        return "string"  # Treat None as string or handle as needed

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (list, List):
        return "array"
    if origin in (dict, Dict):
        return "object"
    if origin in (UnionType,): # For int | None
        # Filter out NoneType and take the first remaining type
        non_none_args = [arg for arg in args if arg is not NoneType]
        if non_none_args:
            # Recursively get the type of the first non-None argument
            # This is a simplification; you might need more complex logic
            # for Unions of multiple non-None types.
            temp_field = FieldInfo(annotation=non_none_args[0])
            return _get_pydantic_field_type(temp_field)
        else:
            return "string" # or whatever default you want if only NoneType is present

    # Handle Literal by inspecting its arguments
    if origin is not None and "Literal" in str(origin):
        if args:
            # Take the type of the first literal value as representative
            first_arg_type = type(args[0])
            temp_field = FieldInfo(annotation=first_arg_type)
            return _get_pydantic_field_type(temp_field)
        return "string"  # Default for empty Literal

    # For non-generic types or types without a clear origin handled above
    check_type = annotation if origin is None else origin

    if not inspect.isclass(check_type):
        # If it's still not a class, it could be a forward reference string or something else.
        # Handle as string as a fallback.
        return "string"

    if issubclass(check_type, str):
        return "string"
    if issubclass(check_type, int):
        return "integer"
    if issubclass(check_type, float):
        return "number"
    if issubclass(check_type, bool):
        return "boolean"
    if issubclass(check_type, list):
        return "array"
    if issubclass(check_type, dict):
        return "object"
        
    return "string"


def _discover_tools_from_collections() -> Dict[str, Dict[str, Any]]:
    """
    Dynamically discover MCP tools from the mcp_collections subdirectories.

    Scans for ActionCollection subclasses and extracts public methods
    starting with "mcp_".

    Returns:
        A dictionary mapping tool name to a dictionary of its MCP functions
        and their introspected details (docstring, parameters).
    """
    tools_info = {}
    mcp_collections_dir = os.path.join(os.path.dirname(__file__), '..', 'mcp_collections')
    subdirs_to_scan = ['documents', 'intelligence', 'media', 'tools']

    for subdir in subdirs_to_scan:
        current_dir = os.path.join(mcp_collections_dir, subdir)
        if not os.path.isdir(current_dir):
            continue

        for filename in os.listdir(current_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                server_name = filename[:-3]
                module_path = os.path.join(current_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(f"mas_arena.mcp_collections.{subdir}.{server_name}", module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, ActionCollection) and obj is not ActionCollection:
                                tool_name = getattr(obj, 'tool_name', None)
                                if not tool_name:
                                    continue

                                if tool_name not in tools_info:
                                    tools_info[tool_name] = {}
                                
                                for method_name, method_obj in inspect.getmembers(obj):
                                    if method_name.startswith("mcp_") and inspect.isfunction(method_obj):
                                        sig = inspect.signature(method_obj)
                                        docstring = inspect.getdoc(method_obj)
                                        
                                        parameters = {
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        }
                                        
                                        for param_name, param in sig.parameters.items():
                                            if param_name == 'self':
                                                continue

                                            if isinstance(param.default, FieldInfo):
                                                field_info = param.default
                                                param_type = _get_pydantic_field_type(field_info)
                                                param_desc = field_info.description
                                                param_default = field_info.default
                                                
                                                parameters["properties"][param_name] = {
                                                    "type": param_type,
                                                    "description": param_desc
                                                }
                                                if param_default is not ... and param_default is not None and param_default is not PydanticUndefined:
                                                    parameters["properties"][param_name]["default"] = param_default
                                                else:
                                                    parameters["required"].append(param_name)

                                        tools_info[tool_name][method_name] = {
                                            "description": docstring,
                                            "parameters": parameters
                                        }

                except Exception as e:
                    logger.error(f"Error discovering tools in {filename}: {e}")

    return tools_info


async def mcp_tool_desc_transform(mcp_servers: List[str], mcp_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform MCP server configurations into tool descriptions
    
    Args:
        mcp_servers: List of MCP server names (tool_name)
        mcp_config: MCP server configuration dictionary
        
    Returns:
        List of tool descriptions
    """
    if not mcp_servers or not mcp_config or "mcpServers" not in mcp_config:
        logger.warning("MCP servers or config is empty")
        return []
        
    tool_descriptions = []
    
    try:
        all_tools = _discover_tools_from_collections()

        for server_name in mcp_servers:
            if server_name not in mcp_config["mcpServers"]:
                logger.warning(f"Server {server_name} not found in MCP config")
                continue

            if mcp_config["mcpServers"].get(server_name, {}).get("disabled", False):
                logger.info(f"Server {server_name} is disabled in config, skipping.")
                continue

            server_tools = all_tools.get(server_name)

            if not server_tools:
                logger.warning(f"No MCP functions found for server {server_name}")
                continue
            
            for function_name, tool_details in server_tools.items():
                tool_desc = {
                    "name": server_name,
                    "description": tool_details.get("description", f"Function {function_name} from {server_name} server."),
                    "server_name": server_name,
                    "function_name": function_name,
                    "parameters": tool_details.get("parameters", {"type": "object", "properties": {}}),
                }
                tool_descriptions.append(tool_desc)
            
    except Exception as e:
        logger.error(f"Error transforming MCP tool descriptions: {e}")
        traceback.print_exc()
    
    return tool_descriptions


async def call_api(server_name: str, function_name: str, parameters: Dict[str, Any], mcp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call API type MCP tool
    
    Args:
        server_name: Server name
        function_name: Function name to call
        parameters: Tool parameters
        mcp_config: MCP configuration
        
    Returns:
        Tool call result
    """
    # For API type servers, send HTTP request directly
    try:
        import requests
        
        if server_name not in mcp_config.get("mcpServers", {}):
            logger.error(f"Server {server_name} not found in MCP config")
            return {"error": f"Server {server_name} not found in MCP config"}
            
        server_config = mcp_config["mcpServers"][server_name]
        base_url = server_config.get("url", "")
        
        if not base_url:
            logger.error(f"No URL configured for API server {server_name}")
            return {"error": f"No URL configured for API server {server_name}"}
            
        # Build API request
        url = f"{base_url}/{function_name}"
        headers = server_config.get("headers", {})
        timeout = server_config.get("timeout", 60.0)
        
        # Add function_name to parameters for semantic consistency
        if isinstance(parameters, dict):
            parameters = parameters.copy()  # Create a copy to avoid modifying the original
            parameters["function_name"] = function_name
        
        logger.info(f"Calling API function {function_name} on {server_name} at {url}")
        
        # Process environment variables in headers
        processed_headers = {}
        for key, value in headers.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                processed_headers[key] = os.environ.get(env_var_name, "")
            else:
                processed_headers[key] = value
        
        # Send request with timeout
        response = requests.post(
            url, 
            json=parameters, 
            headers=processed_headers,
            timeout=timeout
        )
        
        if response.status_code != 200:
            error_msg = f"API call failed with status code {response.status_code}: {response.text[:100]}..."
            logger.error(error_msg)
            return {"error": error_msg}
            
        try:
            result = response.json()
            logger.info(f"Received successful response from {server_name}.{function_name}")
            return {"result": result}
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {e}"
            logger.error(f"{error_msg}. Response text: {response.text[:100]}...")
            return {"error": error_msg}
            
    except requests.exceptions.Timeout:
        error_msg = f"API call to {server_name}.{function_name} timed out"
        logger.error(error_msg)
        return {"error": error_msg}
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error calling {server_name}.{function_name}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logger.error(f"Error calling API {function_name} on {server_name}: {e}")
        traceback.print_exc()
        return {"error": str(e)}
    
async def call_function_tool(server_name: str, function_name: str, parameters: Dict[str, Any], mcp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call function type MCP tool
    
    Args:
        server_name: Server name
        function_name: Function name to call
        parameters: Tool parameters
        mcp_config: MCP configuration
        
    Returns:
        Tool call result
    """
    # For function type servers, import and call the function directly
    try:
        if server_name not in mcp_config.get("mcpServers", {}):
            logger.error(f"Server {server_name} not found in MCP config")
            return {"error": f"Server {server_name} not found in MCP config"}
            
        server_config = mcp_config["mcpServers"][server_name]
        module_path = server_config.get("module_path", "")
        
        if not module_path:
            logger.error(f"No module path configured for function tool server {server_name}")
            return {"error": f"No module path configured for function tool server {server_name}"}
            
        logger.info(f"Calling function tool {function_name} in module {module_path}")
        
        # Import module
        module_parts = module_path.split(".")
        module_name = ".".join(module_parts)
        
        try:
            # Dynamic import with error handling
            try:
                module = __import__(module_name, fromlist=[function_name])
            except ImportError as e:
                logger.error(f"Failed to import module {module_name}: {e}")
                return {"error": f"Could not import module {module_name}: {str(e)}"}
                
            # Get function with error handling
            try:
                function = getattr(module, function_name)
            except AttributeError as e:
                logger.error(f"Function {function_name} not found in module {module_name}: {e}")
                return {"error": f"Function {function_name} not found in module {module_name}"}
            
            # Validate function is callable
            if not callable(function):
                logger.error(f"{function_name} in module {module_name} is not callable")
                return {"error": f"{function_name} in module {module_name} is not callable"}
            
            # Call function with timeout
            try:
                # Prepare arguments - handle both positional and keyword args
                if isinstance(parameters, dict):
                    # Execute function with timeout
                    result = await asyncio.wait_for(
                        asyncio.to_thread(function, **parameters),
                        timeout=server_config.get("timeout", 60.0)
                    )
                else:
                    logger.error(f"Invalid parameters type for {function_name}: expected dict, got {type(parameters)}")
                    return {"error": f"Invalid parameters type: expected dict, got {type(parameters)}"}
                
                logger.info(f"Successfully called function {function_name} in {module_name}")
                
                # Process result
                if result is None:
                    return {"result": "Function executed successfully but returned None"}
                    
                # Try to make result JSON serializable
                try:
                    # Test JSON serialization
                    json.dumps(result)
                    return {"result": result}
                except (TypeError, OverflowError) as e:
                    # If result is not JSON serializable, convert to string
                    logger.warning(f"Function result not JSON serializable: {e}. Converting to string.")
                    return {"result": str(result)}
                    
            except asyncio.TimeoutError:
                logger.error(f"Function call to {function_name} timed out")
                return {"error": f"Function call to {function_name} timed out"}
            except Exception as e:
                logger.error(f"Error executing function {function_name}: {e}")
                traceback.print_exc()
                return {"error": f"Error executing function: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Unexpected error calling function {function_name}: {e}")
            traceback.print_exc()
            return {"error": f"Unexpected error: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error calling function {function_name} on {server_name}: {e}")
        traceback.print_exc()
        return {"error": str(e)}

async def call_mcp_tool(server_name: str, function_name: str, parameters: Dict[str, Any], mcp_config: Dict[str, Any], sandbox: Any) -> Dict[str, Any]:
    """
    Call MCP tool, choosing appropriate call method based on server type
    
    Args:
        server_name: Server name
        function_name: Function name to call
        parameters: Tool parameters
        mcp_config: MCP configuration
        sandbox: The sandbox instance for process management
        
    Returns:
        Tool call result
    """
    if not mcp_config or "mcpServers" not in mcp_config or server_name not in mcp_config["mcpServers"]:
        logger.error(f"Server {server_name} not found in MCP config")
        return {"error": f"Server {server_name} not found in MCP config"}
        
    server_config = mcp_config["mcpServers"][server_name]
    server_type = server_config.get("type", "stdio")
    
    logger.info(f"Calling function {function_name} on server {server_name} with type {server_type}")
    
    try:
        if server_type == "api":
            return await call_api(server_name, function_name, parameters, mcp_config)
        elif server_type == "function_tool":
            return await call_function_tool(server_name, function_name, parameters, mcp_config)
        else: # stdio
            process = await sandbox.get_server_process(server_name)
            if not process:
                logger.error(f"Failed to get server process for {server_name} from sandbox")
                return {"error": f"Failed to get server process for {server_name}"}

            try:
                input_data = {
                    "function_name": function_name,
                    "name": function_name,
                    "arguments": parameters
                }
                input_json = json.dumps(input_data)
                logger.debug(f"Sending to {server_name}: {input_json}")
                input_bytes = (input_json + "\n").encode()
                
                process.stdin.write(input_bytes)
                await process.stdin.drain()
                
                timeout = server_config.get("timeout", 9999.0)
                
                timeout = server_config.get("timeout", 300.0)
                
                output_lines = []
                start_time = time.time()
                
                # Read lines until we have a response or timeout
                while time.time() - start_time < timeout:
                    try:
                        line_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=2.0)
                        if not line_bytes: # EOF
                            break
                        
                        line = line_bytes.decode(errors='replace').strip()
                        if line:
                            output_lines.append(line)
                            # Heuristic: if a line looks like a final JSON response, stop reading.
                            # This is an optimization to avoid waiting for the timeout.
                            if line.startswith('{') and line.endswith('}'):
                                break
                    except asyncio.TimeoutError:
                        # No new line for 2 seconds, assume the tool has finished responding
                        break

                output_text = "\n".join(output_lines)
                logger.debug(f"Received from {server_name}: {output_text[:500]}...")

                if not output_text:
                    stderr_data = await asyncio.wait_for(process.stderr.read(1024), timeout=1.0)
                    stderr_text = stderr_data.decode(errors='replace') if stderr_data else ""
                    logger.error(f"Empty response from {server_name}. Stderr: {stderr_text}")
                    return {"error": f"Empty response from {server_name}. Stderr: {stderr_text[:200]}..."}

                try:
                    # Find all non-overlapping JSON object strings (non-greedy)
                    json_matches = re.findall(r'\{.*?\}', output_text, re.DOTALL)
                    
                    if not json_matches:
                        raise json.JSONDecodeError("No JSON object found in output", output_text, 0)

                    # Try to parse the last found JSON object as it's the most likely to be the final result
                    last_json_str = json_matches[-1]
                    output_data = json.loads(last_json_str)

                    if isinstance(output_data, dict) and "error" in output_data:
                        logger.warning(f"Tool returned an error: {output_data['error']}")
                        return {"error": output_data["error"]}
                    
                    return {"result": output_data}
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from {server_name} output: {output_text[:500]}")
                    return {"error": f"Failed to parse JSON from output: {output_text[:200]}..."}

            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for response from {server_name}")
                return {"error": f"Timeout waiting for response from {server_name}"}
            except Exception as e:
                logger.error(f"Error communicating with {server_name}: {e}", exc_info=True)
                return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error calling tool {function_name} on {server_name}: {e}", exc_info=True)
        return {"error": f"Unexpected error: {str(e)}"}
