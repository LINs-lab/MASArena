import os
from typing import Optional, Any
from smolagents import Tool
from dotenv import load_dotenv

load_dotenv()

class SearchTool(Tool):
    name = "web_search"
    description = "Perform a web search query (think a google search) and returns the search results."
    inputs = {
        "query": {"type": "string", "description": "The web search query to perform."}
    }
    inputs["filter_year"] = {
        "type": "string",
        "description": "[Optional parameter]: filter the search results to only include pages from a specific year. For example, '2020' will only include pages from 2020. Make sure to use this parameter if you're trying to search for articles from a specific date!",
        "nullable": True,
    }
    output_type = "string"

    def __init__(self, serp_num: int = 10, reflection: bool = False):
        super().__init__()
        self.reflection = reflection
        self.max_results = serp_num
        self.history = []
        
        # Keep Jina name if expected by system, or use generic
        self.name = "web_search"
        self.searcher = None
        
        # Try to import from tools_old INSIDE __init__ to avoid undefined name issues in sandbox
        try:
            from mas_arena.tools_old.search_api_tool import TavilySearch
            self.searcher = TavilySearch()
        except ImportError:
            # Fallback or just log/print, actual handling in forward
            pass
        except Exception:
            pass

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        # Re-import strictly required modules for sandbox execution context if needed
        # But for self.searcher, it's an instance attribute so it should be available 
        # as long as __init__ ran successfully in the main process.
        
        if not self.searcher:
            # Try initializing again if it failed or wasn't available
            try:
                from mas_arena.tools_old.search_api_tool import TavilySearch
                self.searcher = TavilySearch()
            except ImportError:
                 return "Search tool (Tavily) not available. Please install tavily-python and check mas_arena.tools_old setup."
            except Exception as e:
                return f"Failed to initialize TavilySearch: {str(e)}"

        try:
            # TavilySearch.search signature is search(self, query: str) -> str
            return self.searcher.search(query)
        except Exception as e:
            return f"Error executing search: {str(e)}"
