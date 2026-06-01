import requests
import urllib.parse

from smolagents import Tool
import time
from typing import Any, Optional
import os
from .search_reflector import SearchReflector


class BaseSearcher:
    def __init__(self):
        self.history = []
        self.name = "search"

    def search(self, query: str, **kwargs) -> Any:
        raise NotImplementedError()


class JinaSearcher(BaseSearcher):
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 10,
    ):
        super().__init__()
        self.name = "jina_search"
        self.description = (
            "Perform a web search query using Jina API and returns the search results."
        )

        self.jina_api_key = api_key or os.getenv("JINA_API_KEY")
        self.max_results = max_results

    def search(self, query: str, filter_year: Optional[int] = None, **kwargs) -> str:
        if self.jina_api_key is None:
            raise ValueError("Missing Jina API key.")

        self.history.append((query, time.time()))

        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}"
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "X-Respond-With": "no-content",
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text

        except requests.exceptions.RequestException as e:
            raise Exception(f"Jina API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")


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

        self.searcher = JinaSearcher(
            api_key=os.getenv("JINA_API_KEY"),
            max_results=serp_num,
        )

        self.name = self.searcher.name
        self.description = self.searcher.description

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:

        if self.reflection:
            self.reflector = SearchReflector()
            _, query = self.reflector.query_reflect(query)

        results = self.searcher.search(query, filter_year)
        return results
