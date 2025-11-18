import sys
import json
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def debug_log(message):
    if DEBUG:
        print(f"Web search server: {message}", file=sys.stderr)

def web_search(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using the Tavily API.
    Returns a JSON string with either results or error.
    """
    debug_log(f"Received query: {query}")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps({"error": "TAVILY_API_KEY environment variable not set"})

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)

        # 确保返回统一结构
        return json.dumps({
            "tool_name": "web_search",
            "content": {
                "query": query,
                "results": response.get("results", [])
            }
        })

    except Exception as e:
        debug_log(f"Error during search: {str(e)}")
        return json.dumps({
            "tool_name": "web_search",
            "error": str(e),
            "content": None
        })


if __name__ == "__main__":
    try:
        raw_input = sys.stdin.read()
        debug_log(f"Raw input: {raw_input}")

        try:
            input_data = json.loads(raw_input)
        except json.JSONDecodeError as je:
            print(json.dumps({
                "tool_name": "web_search",
                "error": f"Invalid JSON input: {str(je)}",
                "content": None
            }))
            sys.exit(1)

        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})

        if tool_name == "web_search":
            query = tool_input.get("query")
            if not query:
                print(json.dumps({
                    "tool_name": "web_search",
                    "error": "Missing 'query' in tool_input",
                    "content": None
                }))
            else:
                result = web_search(query)
                print(result)
        else:
            print(json.dumps({
                "tool_name": tool_name,
                "error": f"Unknown tool_name: {tool_name}",
                "content": None
            }))

    except Exception as e:
        print(json.dumps({
            "tool_name": "web_search",
            "error": str(e),
            "content": None
        }))