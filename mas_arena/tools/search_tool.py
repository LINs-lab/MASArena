import os
import time
import urllib.parse
from typing import Optional, Any

import requests
from smolagents import Tool
from dotenv import load_dotenv

load_dotenv()

class JinaSearcher:
    """Minimal Jina Search wrapper (s.jina.ai)."""

    def __init__(self, api_key: Optional[str] = None, max_results: int = 10):
        self.jina_api_key = api_key or os.getenv("JINA_API_KEY")
        self.max_results = max_results
        self.history = []

    def search(self, query: str, filter_year: Optional[int] = None) -> str:
        if not self.jina_api_key:
            raise ValueError("Missing Jina API key (JINA_API_KEY).")

        # Keep parity with other searchers: store history
        self.history.append((query, time.time()))

        # Jina's endpoint takes only q; filter_year is best-effort (ignored here).
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}"
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            # Avoid fetching full page bodies in search results
            "X-Respond-With": "no-content",
        }
        timeout_s = float(os.getenv("SEARCH_JINA_TIMEOUT", "30"))
        resp = requests.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        return resp.text

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
        
        # Keep tool name stable for all BenchAgent-based MAS
        self.name = "web_search"
        self.searcher = None  # JinaSearcher | TavilySearch
        
        # Provider selection:
        # - Default is Jina, matching project preference to swap away from Tavily.
        # - To force Tavily: WEB_SEARCH_PROVIDER=tavily
        # - To allow fallback to Tavily when Jina missing: WEB_SEARCH_ALLOW_TAVILY_FALLBACK=1
        provider = (os.getenv("WEB_SEARCH_PROVIDER") or "jina").strip().lower()
        allow_tavily_fallback = (os.getenv("WEB_SEARCH_ALLOW_TAVILY_FALLBACK") or "").strip() in {"1", "true", "yes"}

        if provider in {"jina", "jina_search"}:
            try:
                self.searcher = JinaSearcher(api_key=os.getenv("JINA_API_KEY"), max_results=serp_num)
            except Exception:
                self.searcher = None

            if not os.getenv("JINA_API_KEY") and allow_tavily_fallback:
                provider = "tavily"

        if provider == "tavily":
            try:
                from mas_arena.tools_old.search_api_tool import TavilySearch
                self.searcher = TavilySearch()
            except Exception:
                self.searcher = None

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        try:
            if not self.searcher:
                return (
                    "Search tool not available. Default provider is Jina. "
                    "Set JINA_API_KEY to enable Jina search. "
                    "If you must use Tavily, set WEB_SEARCH_PROVIDER=tavily "
                    "(or allow fallback via WEB_SEARCH_ALLOW_TAVILY_FALLBACK=1)."
                )

            # JinaSearcher.search signature includes filter_year (unused); Tavily ignores it.
            if hasattr(self.searcher, "search"):
                try:
                    return self.searcher.search(query, filter_year=filter_year)
                except TypeError:
                    return self.searcher.search(query)

            return "Search tool misconfigured: missing search() method."
        except Exception as e:
            return f"Error executing search: {str(e)}"
