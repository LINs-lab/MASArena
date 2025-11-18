#!/usr/bin/env python3
"""
Mock MCP search server for testing MCP tool integration.
This server simulates a real MCP server providing search tools.
"""

import sys
import json
import asyncio
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
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            "returns": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        {
            "name": "document_search",
            "description": "Search within documents or papers",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return"}
                },
                "required": ["query"]
            },
            "returns": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "document": {"type": "string"},
                                "content": {"type": "string"},
                                "relevance": {"type": "number"}
                            }
                        }
                    }
                }
            }
        },
        {
            "name": "knowledge_lookup",
            "description": "Lookup facts or knowledge on a specific topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to lookup"},
                    "specific_aspect": {"type": "string", "description": "Specific aspect of topic to focus on"}
                },
                "required": ["topic"]
            },
            "returns": {"type": "string", "description": "Information about the topic"}
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
        
        if tool_name == "web_search":
            query = parameters.get("query", "")
            if not query:
                raise ValueError("Search query cannot be empty")
                
            # Mock search results
            result = {
                "results": [
                    {
                        "title": f"Mock result 1 for '{query}'",
                        "url": f"https://example.com/result1?q={query}",
                        "snippet": f"This is a mock search result for the query '{query}'. It contains relevant information."
                    },
                    {
                        "title": f"Mock result 2 for '{query}'",
                        "url": f"https://example.com/result2?q={query}",
                        "snippet": f"Another mock search result for '{query}', providing additional context and information."
                    },
                    {
                        "title": f"Mock result 3 for '{query}'",
                        "url": f"https://example.com/result3?q={query}",
                        "snippet": f"A third mock search result for '{query}', with different perspective on the topic."
                    }
                ]
            }
            
        elif tool_name == "document_search":
            query = parameters.get("query", "")
            num_results = parameters.get("num_results", 3)
            
            if not query:
                raise ValueError("Search query cannot be empty")
                
            # Limit number of results
            num_results = min(max(1, num_results), 5)
            
            # Mock document search results
            matches = []
            for i in range(num_results):
                matches.append({
                    "document": f"Document_{i+1}.pdf",
                    "content": f"Extract from document containing '{query}' and relevant context...",
                    "relevance": round(0.9 - (i * 0.15), 2)
                })
                
            result = {"matches": matches}
            
        elif tool_name == "knowledge_lookup":
            topic = parameters.get("topic", "")
            aspect = parameters.get("specific_aspect", "")
            
            if not topic:
                raise ValueError("Topic cannot be empty")
                
            # Format response based on whether specific aspect is provided
            if aspect:
                result = f"Mock knowledge about '{topic}', specifically regarding '{aspect}'. This information comes from a structured knowledge base and provides detailed facts about the requested topic and aspect."
            else:
                result = f"Mock general knowledge about '{topic}'. This includes key facts, definitions, and common understanding of the topic from a simulated knowledge base."
                
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