from typing import Optional, Dict, Any
from smolagents import Tool
import logging

logger = logging.getLogger(__name__)


class WikipediaSearchTool(Tool):
    """Wikipedia search tool for retrieving article summaries."""

    name = "wikipedia_search"
    description = "Search Wikipedia for information on a given topic and return a concise summary."

    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for Wikipedia.",
        },
        "max_sentences": {
            "type": "integer",
            "description": "Maximum number of sentences to return. Default is 3.",
            "default": 3,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._wikipedia_available = False
        self._wikipedia = None

        # Try to import wikipedia package
        try:
            import wikipedia

            self._wikipedia = wikipedia
            self._wikipedia_available = True
            wikipedia.set_lang("en")
        except ImportError:
            logger.warning(
                "wikipedia package not installed. WikipediaSearchTool will return error messages."
            )

    def forward(self, query: str, max_sentences: Optional[int] = None) -> str:
        """Search Wikipedia and return a summary."""
        max_sentences = max_sentences or 3

        if not self._wikipedia_available:
            return (
                "Wikipedia search unavailable: 'wikipedia' package not installed. "
                "Please install it with: pip install wikipedia"
            )

        try:
            search_results = self._wikipedia.search(query, results=5)
            if not search_results:
                return f"No Wikipedia articles found for '{query}'"

            for title in search_results:
                try:
                    summary = self._wikipedia.summary(title, sentences=max_sentences)
                    return f"Wikipedia article: {title}\n\n{summary}"
                except self._wikipedia.exceptions.DisambiguationError as e:
                    if e.options:
                        try:
                            summary = self._wikipedia.summary(
                                e.options[0], sentences=max_sentences
                            )
                            return f"Wikipedia article: {e.options[0]}\n\n{summary}"
                        except Exception:
                            continue
                except self._wikipedia.exceptions.PageError:
                    continue
                except Exception as inner_e:
                    logger.warning(f"Error getting summary for {title}: {inner_e}")
                    continue

            return (
                f"Found Wikipedia articles for '{query}' but couldn't retrieve content: "
                f"{', '.join(search_results)}"
            )

        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"


# 设置日志，方便查看警告信息
logging.basicConfig(level=logging.WARNING)


def test_wikipedia_tool():
    print("🔍 正在初始化 WikipediaSearchTool...")
    tool = WikipediaSearchTool()

    test_cases = [
        {"query": "Albert Einstein"},
        {"query": "Quantum mechanics", "max_sentences": 2},
        {"query": "asdfghjkl123456"},  # 应该找不到结果
        {"query": "Python (programming language)"},
        {"query": "New York City", "max_sentences": 5},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n🧪 测试 {i}: {case}")
        try:
            result = tool.forward(**case)
            print(f"✅ 结果:\n{result[:500]}{'...' if len(result) > 500 else ''}\n")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 测试无参数调用（应使用默认 max_sentences=3）
    print("\n🧪 测试默认参数调用:")
    try:
        result = tool.forward("Machine learning")
        print(f"✅ 默认调用结果（前300字符）:\n{result[:300]}...\n")
    except Exception as e:
        print(f"❌ 异常: {e}")


if __name__ == "__main__":
    test_wikipedia_tool()
