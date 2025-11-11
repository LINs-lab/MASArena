"""
外部工具集成模块

这个模块统一导入和管理所有的外部工具，提供一个清晰的工具接口。
"""

# 媒体工具
from .media.audio_inspector import AudioInspectorTool
from .media.video_inspector import VideoInspectorTool
from .media.visual_inspector import VisualInspectorTool

# 网络工具
from .web.browser import BrowserTool
from .web.download import DownloadTool
from .web.search_tool import SearchTool
from .web.text_inspector import TextInspectorTool
from .web.arxiv_tool import ArxivTool
from .web.crawler_tools import (
    SimpleCrawler,
    CrawlerArchiveSearchTool,
    CrawlerReadTool,
)

# 文档工具
from .document.csv_extractor import CSVExtractorTool
from .document.markdown_converter import MarkdownConverterTool
from .document.sheet_extractor import SheetExtractorTool
from .document.text_extractor import TextExtractorTool
from .document.zip_extractor import ZipExtractorTool

# 系统工具
from .system.terminal_tool import TerminalTool

# 工具类别字典，便于管理和使用
MEDIA_TOOLS = {
    "audio_inspector": AudioInspectorTool,
    "video_inspector": VideoInspectorTool, 
    "visual_inspector": VisualInspectorTool,
}

WEB_TOOLS = {
    "browser": BrowserTool,
    "download": DownloadTool,
    "search": SearchTool,
    "text_inspector": TextInspectorTool,
    "arxiv": ArxivTool,
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

SYSTEM_TOOLS = {
    "terminal": TerminalTool,
}

# 所有外部工具的统一字典
ALL_EXTERNAL_TOOLS = {
    **MEDIA_TOOLS,
    **WEB_TOOLS, 
    **DOCUMENT_TOOLS,
    **SYSTEM_TOOLS,
}

# 导出所有工具类
__all__ = [
    # 媒体工具
    "AudioInspectorTool",
    "VideoInspectorTool",
    "VisualInspectorTool",
    
    # 网络工具
    "BrowserTool",
    "DownloadTool", 
    "SearchTool",
    "TextInspectorTool",
    "ArxivTool",
    "SimpleCrawler",
    "CrawlerArchiveSearchTool",
    "CrawlerReadTool",
    
    # 文档工具
    "CSVExtractorTool",
    "MarkdownConverterTool",
    "SheetExtractorTool",
    "TextExtractorTool", 
    "ZipExtractorTool",
    
    # 系统工具
    "TerminalTool",
    
    # 工具字典
    "MEDIA_TOOLS",
    "WEB_TOOLS",
    "DOCUMENT_TOOLS", 
    "SYSTEM_TOOLS",
    "ALL_EXTERNAL_TOOLS",
]
