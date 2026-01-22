"""
Web crawler tools for smolagents.
Provides page crawling and archival capabilities.
"""

import os
import time
import asyncio

import requests
from smolagents import Tool

from dotenv import load_dotenv

load_dotenv()


class SimpleCrawler:
    """
    A simple crawler for agent to crawl pages through url, read page content, etc.
    """

    def __init__(self):
        self.history = []

    def _pre_visit(self, url: str) -> str:
        """Check if URL was previously visited."""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i][0] == url:
                return f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
        return ""

    def _check_history(self, url_or_query: str) -> str:
        """Check and update history."""
        header = ""
        for i in range(len(self.history) - 2, -1, -1):  # Start from the second last
            if self.history[i][0] == url_or_query:
                header += f"You previously visited this page {round(time.time() - self.history[i][1])} seconds ago.\n"
                return header
        self.history.append((url_or_query, time.time()))
        return header

    async def _crawl_page_async(self, url: str) -> str:
        """Crawl page using crawl4ai (async version)."""
        try:
            from crawl4ai import AsyncWebCrawler

            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                # Handle the result properly - it might be different types
                content = ""
                if hasattr(result, "markdown"):
                    content = getattr(result, "markdown", "")
                if not content and hasattr(result, "cleaned_html"):
                    content = getattr(result, "cleaned_html", "")
                if not content and hasattr(result, "text"):
                    content = getattr(result, "text", "")
                if not content:
                    content = str(result)
                return content or "No content extracted"
        except ImportError:
            return self._read_page_simple(url)
        except Exception as e:
            return f"Error crawling with crawl4ai: {str(e)}\n\nFalling back to simple method:\n{self._read_page_simple(url)}"

    def crawl_page(self, url: str) -> str:
        """Crawl page content."""
        header = self._check_history(url)
        try:
            # Try async crawling first
            pages = asyncio.run(self._crawl_page_async(url))
        except Exception:
            # Fallback to simple reading
            pages = self._read_page_simple(url)
        return header + pages

    def _read_page_simple(self, url: str) -> str:
        """Simple page reading using requests."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Simple text extraction
            from html import unescape
            import re

            content = response.text
            # Remove script and style tags
            content = re.sub(
                r"<script[^>]*>.*?</script>",
                "",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            content = re.sub(
                r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove HTML tags
            content = re.sub(r"<[^>]+>", "", content)
            # Unescape HTML entities
            content = unescape(content)
            # Clean up whitespace
            content = re.sub(r"\s+", " ", content).strip()

            return content[:8000]  # Limit content length

        except Exception as e:
            return f"Error reading page: {str(e)}"

    def read_page(self, url: str) -> str:
        """Read page using Jina AI or fallback."""

        def jina_read(url: str) -> str:
            jina_api_key = os.getenv("JINA_API_KEY")
            if not jina_api_key:
                return self._read_page_simple(url)

            jina_url = f"https://r.jina.ai/{url}"
            headers = {
                "Authorization": f"Bearer {jina_api_key}",
                "X-Engine": "browser",
                "X-Return-Format": "text",
                "X-Timeout": "10",
                "X-Token-Budget": "80000",
            }
            try:
                response = requests.get(jina_url, headers=headers, timeout=15)
                response.raise_for_status()
                return response.text
            except Exception:
                return self._read_page_simple(url)

        return jina_read(url)


class CrawlerArchiveSearchTool(Tool):
    """Archive search tool for finding archived versions of URLs."""

    name = "find_archived_url"
    description = "Given a url, searches the Wayback Machine and returns the archived version of the url that's closest in time to the desired date."

    inputs = {
        "url": {"type": "string", "description": "The url you need the archive for."},
        "date": {
            "type": "string",
            "description": "The date that you want to find the archive for. Give this date in the format 'YYYYMMDD', for instance '27 June 2008' is written as '20080627'.",
        },
    }
    output_type = "string"

    def __init__(self, crawler: SimpleCrawler = None, read_type: str = "jina_read"):
        super().__init__()
        self.crawler = crawler
        self.read_type = read_type

    def forward(self, url: str, date: str) -> str:
        """Find and read archived URL."""
        import requests
        try:
            no_timestamp_url = f"https://archive.org/wayback/available?url={url}"
            archive_url = no_timestamp_url + f"&timestamp={date}"

            response = requests.get(archive_url, timeout=20).json()
            response_notimestamp = requests.get(no_timestamp_url, timeout=20).json()

            closest = None
            if (
                "archived_snapshots" in response
                and "closest" in response["archived_snapshots"]
            ):
                closest = response["archived_snapshots"]["closest"]
            elif (
                "archived_snapshots" in response_notimestamp
                and "closest" in response_notimestamp["archived_snapshots"]
            ):
                closest = response_notimestamp["archived_snapshots"]["closest"]

            if not closest:
                return f"URL {url} was not archived on Wayback Machine, try a different url."

            target_url = closest["url"]

            if self.read_type == "crawl":
                content = self.crawler.crawl_page(target_url)
            else:
                content = self.crawler.read_page(target_url)

            return (
                f"Web archive for url {url}, snapshot taken at date {closest['timestamp'][:8]}:\n"
                + content
            )
        except Exception as e:
            return f"Error accessing archive: {str(e)}"


class CrawlerReadTool(Tool):
    """Tool for reading webpage content."""

    name = "crawl_pages"
    description = "Access a webpage using the provided URL and return completed contents of the webpage. In the case of a YouTube video URL, extract and return the video transcript."

    inputs = {
        "url": {
            "type": "string",
            "description": "The relative or absolute url of the webpage to visit.",
        },
    }
    output_type = "string"

    def __init__(self, crawler: SimpleCrawler = None, read_type: str = "jina_read"):
        super().__init__()
        self.crawler = crawler
        self.read_type = read_type

    def forward(self, url: str) -> str:
        """Read webpage content."""
        if self.read_type == "crawl":
            result = self.crawler.crawl_page(url)
        else:
            result = self.crawler.read_page(url)

        if result.strip() == "" or result == "\n":
            return (
                f"Crawling for url: {url} returned no content, maybe it is a url for .pdf file which is unable to crawl. "
                "Please try to use tool: inspect_file_as_text() to get the contents."
            )
        return result
