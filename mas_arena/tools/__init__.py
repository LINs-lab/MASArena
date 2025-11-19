"""
外部工具集成模块

这个模块统一导入和管理所有的外部工具，提供一个清晰的工具接口。
"""

# 媒体工具
from .audio_inspector import AudioInspectorTool
from .visual_inspector import VisualInspectorTool

# 网络工具
from .search_tool import SearchTool
from .wikipedia_search import WikipediaSearchTool  # new
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
from .final_answer import FinalAnswerTool  # new

# 具体工具类
from .python_interpreter import PythonInterpreterTool  # new


# 工具类别字典，便于管理和使用
MEDIA_TOOLS = {
    "audio_inspector": AudioInspectorTool,
    "visual_inspector": VisualInspectorTool,
}

WEB_TOOLS = {
    "search": SearchTool,
    "wikipedia_search": WikipediaSearchTool,  # new
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
    "final_answer": FinalAnswerTool,  # new
}

LOCAL_TOOLS = {
    "python_interpreter": PythonInterpreterTool,  # new
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
    "SimpleCrawler",
    "CrawlerArchiveSearchTool",
    "CrawlerReadTool",
    "WikipediaSearchTool",  # new
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
    "LOCAL_TOOLS" "ALL_EXTERNAL_TOOLS",
]
