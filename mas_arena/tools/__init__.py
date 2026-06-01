"""
外部工具集成模块

这个模块统一导入和管理所有的外部工具，提供一个清晰的工具接口。
现在修改为优先使用 active tool 实现，必要时兼容封装内部 legacy 实现。
"""

# 媒体工具
# 尝试从内部 legacy 实现导入，或者使用新实现。
from .audio_inspector import AudioInspectorTool
from .visual_inspector import VisualInspectorTool

# 网络工具
from .search_tool import SearchTool # web_search：优先 Jina(s.jina.ai)，无 key 时兜底 Tavily
from .browser_tool import BrowserTool # 新增：封装内部 legacy BrowserTool
from .wikipedia_search import WikipediaSearchTool  # legacy 没有 Wiki，保留新实现
from .crawler_tools import (
    SimpleCrawler,
    CrawlerArchiveSearchTool,
    CrawlerReadTool,
)

# 文档工具
from .csv_extractor import CSVExtractorTool
from .markdown_converter import MarkdownConverterTool
from .sheet_extractor import SheetExtractorTool
from .text_extractor import TextExtractorTool
from .zip_extractor import ZipExtractorTool

# 本系统工具
from .final_answer import FinalAnswerTool

# 具体工具类
from .python_interpreter import PythonInterpreterTool

# 工具类别字典，便于管理和使用
MEDIA_TOOLS = {
    "audio_inspector": AudioInspectorTool,
    "visual_inspector": VisualInspectorTool,
}

WEB_TOOLS = {
    "search": SearchTool,
    "browser": BrowserTool,
    "wikipedia_search": WikipediaSearchTool,
    "crawler_read": CrawlerReadTool,
    "crawler_archive_search": CrawlerArchiveSearchTool,
}

DOCUMENT_TOOLS = {
    "csv_extractor": CSVExtractorTool,
    "markdown_converter": MarkdownConverterTool,
    "sheet_extractor": SheetExtractorTool,
    "text_extractor": TextExtractorTool,
    "zip_extractor": ZipExtractorTool,
}

MASARENA_TOOLS = {
    "final_answer": FinalAnswerTool,
}

LOCAL_TOOLS = {
    "python_interpreter": PythonInterpreterTool,
}


# 所有外部工具的统一字典
ALL_EXTERNAL_TOOLS = {
    **MEDIA_TOOLS,
    **WEB_TOOLS,
    **DOCUMENT_TOOLS,
    **MASARENA_TOOLS,
    **LOCAL_TOOLS,
}

# 导出所有工具类
__all__ = [
    # 媒体工具
    "AudioInspectorTool",
    "VisualInspectorTool",
    # 网络工具
    "SearchTool",
    "BrowserTool",
    "SimpleCrawler",
    "CrawlerArchiveSearchTool",
    "CrawlerReadTool",
    "WikipediaSearchTool",
    # 文档工具
    "CSVExtractorTool",
    "MarkdownConverterTool",
    "SheetExtractorTool",
    "TextExtractorTool",
    "ZipExtractorTool",
    # 工具字典
    "MEDIA_TOOLS",
    "WEB_TOOLS",
    "DOCUMENT_TOOLS",
    "MASARENA_TOOLS",
    "LOCAL_TOOLS",
    "ALL_EXTERNAL_TOOLS",
]
