#!/usr/bin/env python3
"""
Mock MCP math server for testing MCP tool integration.
This server simulates a real MCP server providing math tools.
"""

import sys
import json
import asyncio
import math
from typing import Dict, Any, Optional


async def handle_mcp_messages():
    """Main handler for MCP messages over stdio."""
    try:
        # Send hello message with tools
        await send_hello()
        
        # Process incoming messages
        while True:
            message = await read_message()
            if not message:
                break
                
            await process_message(message)
    except Exception as e:
        sys.stderr.write(f"Error in MCP server: {e}\n")
        sys.exit(1)


async def read_message() -> Optional[Dict[str, Any]]:
    """Read a message from stdin."""
    try:
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            return None
        return json.loads(line)
    except Exception as e:
        sys.stderr.write(f"Error reading message: {e}\n")
        return None


async def send_message(message: Dict[str, Any]):
    """Send a message to stdout."""
    try:
        json_str = json.dumps(message)
        sys.stdout.write(f"{json_str}\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f"Error sending message: {e}\n")


async def send_hello():
    """Send hello message with tool definitions."""
    tool_definitions = [
        {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            "returns": {"type": "number", "description": "Sum of a and b"}
        },
        {
            "name": "subtract",
            "description": "Subtract second number from first",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number to subtract"}
                },
                "required": ["a", "b"]
            },
            "returns": {"type": "number", "description": "Result of a - b"}
        },
        {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            "returns": {"type": "number", "description": "Product of a and b"}
        },
        {
            "name": "divide",
            "description": "Divide first number by second",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator"}
                },
                "required": ["a", "b"]
            },
            "returns": {"type": "number", "description": "Result of a / b"}
        },
        {
            "name": "solve_quadratic",
            "description": "Solve a quadratic equation of form ax² + bx + c = 0",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Coefficient of x²"},
                    "b": {"type": "number", "description": "Coefficient of x"},
                    "c": {"type": "number", "description": "Constant term"}
                },
                "required": ["a", "b", "c"]
            },
            "returns": {
                "type": "object",
                "properties": {
                    "roots": {"type": "array", "items": {"type": "number"}},
                    "explanation": {"type": "string"}
                }
            }
        }
    ]
    
    hello_message = {
        "type": "hello",
        "version": "v1",
        "tools": tool_definitions
    }
    
    await send_message(hello_message)


async def process_message(message: Dict[str, Any]):
    """Process incoming MCP messages."""
    message_type = message.get("type")
    
    if message_type == "tool_call":
        await handle_tool_call(message)
    else:
        # Unsupported message type
        await send_message({
            "type": "error",
            "message": f"Unsupported message type: {message_type}"
        })


async def handle_tool_call(message: Dict[str, Any]):
    """Handle a tool call message."""
    call_id = message.get("id")
    tool_name = message.get("name")
    parameters = message.get("parameters", {})
    
    try:
        result = None
        
        if tool_name == "add":
            result = parameters.get("a", 0) + parameters.get("b", 0)
        elif tool_name == "subtract":
            result = parameters.get("a", 0) - parameters.get("b", 0)
        elif tool_name == "multiply":
            result = parameters.get("a", 0) * parameters.get("b", 0)
        elif tool_name == "divide":
            a = parameters.get("a", 0)
            b = parameters.get("b", 1)
            if b == 0:
                raise ValueError("Division by zero")
            result = a / b
        elif tool_name == "solve_quadratic":
            a = parameters.get("a", 0)
            b = parameters.get("b", 0)
            c = parameters.get("c", 0)
            
            if a == 0:
                if b == 0:
                    raise ValueError("Not a valid quadratic or linear equation")
                # Linear equation: bx + c = 0
                x = -c / b
                result = {
                    "roots": [x],
                    "explanation": f"Linear equation {b}x + {c} = 0 has single root x = {x}"
                }
            else:
                # Calculate discriminant
                discriminant = b**2 - 4*a*c
                
                if discriminant < 0:
                    # Complex roots
                    real_part = -b / (2*a)
                    imag_part = math.sqrt(abs(discriminant)) / (2*a)
                    result = {
                        "roots": [],  # Empty for complex roots
                        "explanation": f"Complex roots: {real_part} + {imag_part}i and {real_part} - {imag_part}i"
                    }
                elif discriminant == 0:
                    # One real root
                    x = -b / (2*a)
                    result = {
                        "roots": [x],
                        "explanation": f"Single root x = {x}"
                    }
                else:
                    # Two real roots
                    x1 = (-b + math.sqrt(discriminant)) / (2*a)
                    x2 = (-b - math.sqrt(discriminant)) / (2*a)
                    result = {
                        "roots": [x1, x2],
                        "explanation": f"Two roots: x = {x1} and x = {x2}"
                    }
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
            
        # Send successful result
        await send_message({
            "type": "tool_result",
            "id": call_id,
            "result": result
        })
        
    except Exception as e:
        # Send error result
        await send_message({
            "type": "tool_error",
            "id": call_id,
            "error": {
                "message": str(e),
                "type": "application_error"
            }
        })


if __name__ == "__main__":
    asyncio.run(handle_mcp_messages())