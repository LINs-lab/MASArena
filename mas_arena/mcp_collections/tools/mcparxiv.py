"""
ArXiv MCP Server

This module provides MCP server functionality for ArXiv academic paper operations.
It supports paper search, metadata extraction, and content retrieval with LLM-friendly formatting.

Key features:
- Search ArXiv papers by query, author, category, or ID
- Extract paper metadata (title, authors, abstract, etc.)
- Download and process paper PDFs
- LLM-optimized content formatting
- Comprehensive error handling and logging

Main functions:
- mcp_search_papers: Search ArXiv papers with flexible criteria
- mcp_get_paper_details: Get detailed information about specific papers
- mcp_download_paper: Download paper PDF and extract text content
- mcp_get_categories: Get available ArXiv subject categories
- mcp_get_arxiv_capabilities: Get service capabilities and configuration
"""

import json
import traceback
from datetime import datetime

import arxiv
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from mas_arena.mcp_collections.base import ActionArguments, ActionResponse, ActionCollection


class PaperResult(BaseModel):
    """Individual paper search result with structured data."""

    entry_id: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str | None = None
    categories: list[str]
    primary_category: str
    pdf_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None


class ArxivMetadata(BaseModel):
    """Metadata for ArXiv operation results."""

    operation: str
    query: str | None = None
    max_results: int | None = None
    sort_by: str | None = None
    sort_order: str | None = None
    total_results: int | None = None
    execution_time: float | None = None
    error_type: str | None = None
    paper_id: str | None = None
    download_path: str | None = None
    file_size: int | None = None


class ArxivActionCollection(ActionCollection):
    """MCP service for ArXiv academic paper operations.

    Provides comprehensive ArXiv functionality including:
    - Paper search with flexible criteria (query, author, category, ID)
    - Detailed paper metadata extraction
    - PDF download and text content extraction
    - Subject category information
    - LLM-optimized result formatting
    - Error handling and logging
    """
    tool_name = "mcparxiv"

    def __init__(self, arguments: ActionArguments) -> None:
        super().__init__(arguments)

        # Initialize supported file extensions for PDF processing
        self.supported_extensions = {".pdf"}

        # ArXiv client configuration
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,  # Be respectful to ArXiv servers
            num_retries=3,
        )

        # Create downloads directory
        self._downloads_dir = self.workspace / "arxiv_downloads"
        self._downloads_dir.mkdir(exist_ok=True)

        # ArXiv subject categories mapping
        self.subject_categories = {
            "cs": "Computer Science",
            "math": "Mathematics",
            "physics": "Physics",
            "astro-ph": "Astrophysics",
            "cond-mat": "Condensed Matter",
            "gr-qc": "General Relativity and Quantum Cosmology",
            "hep-ex": "High Energy Physics - Experiment",
            "hep-lat": "High Energy Physics - Lattice",
            "hep-ph": "High Energy Physics - Phenomenology",
            "hep-th": "High Energy Physics - Theory",
            "math-ph": "Mathematical Physics",
            "nlin": "Nonlinear Sciences",
            "nucl-ex": "Nuclear Experiment",
            "nucl-th": "Nuclear Theory",
            "quant-ph": "Quantum Physics",
            "q-bio": "Quantitative Biology",
            "q-fin": "Quantitative Finance",
            "stat": "Statistics",
            "econ": "Economics",
            "eess": "Electrical Engineering and Systems Science",
        }

        print("ArXiv service initialized")
        print(f"Downloads directory: {self._downloads_dir}")

    def _format_paper_result(self, paper: arxiv.Result) -> PaperResult:
        """Convert arxiv.Result to structured PaperResult.

        Args:
            paper: ArXiv paper result object

        Returns:
            Structured PaperResult object
        """
        return PaperResult(
            entry_id=paper.entry_id,
            title=paper.title.strip(),
            authors=[author.name for author in paper.authors],
            summary=paper.summary.strip(),
            published=paper.published.isoformat(),
            updated=paper.updated.isoformat() if paper.updated else None,
            categories=paper.categories,
            primary_category=paper.primary_category,
            pdf_url=paper.pdf_url,
            doi=paper.doi,
            journal_ref=paper.journal_ref,
            comment=paper.comment,
        )

    def _format_search_results(self, results: list[PaperResult], output_format: str = "markdown") -> str:
        """Format paper search results for LLM consumption.

        Args:
            results: List of paper results
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string suitable for LLM consumption
        """
        if not results:
            return "No papers found matching the search criteria."

        if output_format == "json":
            return json.dumps([result.model_dump() for result in results], indent=2)

        elif output_format == "text":
            output_parts = [f"Found {len(results)} papers:\n"]

            for i, paper in enumerate(results, 1):
                authors_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f" et al. ({len(paper.authors)} total)"

                output_parts.extend(
                    [
                        f"{i}. {paper.title}",
                        f"   Authors: {authors_str}",
                        f"   Published: {paper.published[:10]}",
                        f"   Categories: {', '.join(paper.categories)}",
                        f"   ArXiv ID: {paper.entry_id.split('/')[-1]}",
                        f"   Abstract: {paper.summary[:200]}...",
                        "",
                    ]
                )

            return "\n".join(output_parts)

        else:  # markdown (default)
            output_parts = [f"# ArXiv Search Results\n\nFound **{len(results)}** papers:\n"]

            for i, paper in enumerate(results, 1):
                authors_str = ", ".join(paper.authors[:3])
                if len(paper.authors) > 3:
                    authors_str += f" *et al.* ({len(paper.authors)} total)"

                arxiv_id = paper.entry_id.split("/")[-1]

                output_parts.extend(
                    [
                        f"## {i}. {paper.title}",
                        f"**Authors:** {authors_str}",
                        f"**Published:** {paper.published[:10]}",
                        f"**Categories:** {', '.join(paper.categories)}",
                        f"**ArXiv ID:** `{arxiv_id}`",
                        f"**PDF:** [Download]({paper.pdf_url})" if paper.pdf_url else "",
                        "",
                        f"**Abstract:** {paper.summary}",
                        "",
                        "---",
                        "",
                    ]
                )

            return "\n".join(output_parts)

    def _format_paper_details(self, paper: PaperResult, output_format: str = "markdown") -> str:
        """Format detailed paper information for LLM consumption.

        Args:
            paper: Paper result object
            output_format: Format type ('markdown', 'json', 'text')

        Returns:
            Formatted string with detailed paper information
        """
        if output_format == "json":
            return json.dumps(paper.model_dump(), indent=2)

        elif output_format == "text":
            output_parts = [
                f"Title: {paper.title}",
                f"Authors: {', '.join(paper.authors)}",
                f"Published: {paper.published}",
                f"Updated: {paper.updated or 'N/A'}",
                f"Primary Category: {paper.primary_category}",
                f"All Categories: {', '.join(paper.categories)}",
                f"ArXiv ID: {paper.entry_id.split('/')[-1]}",
                f"PDF URL: {paper.pdf_url or 'N/A'}",
                f"DOI: {paper.doi or 'N/A'}",
                f"Journal Reference: {paper.journal_ref or 'N/A'}",
                f"Comment: {paper.comment or 'N/A'}",
                "",
                "Abstract:",
                paper.summary,
            ]

            return "\n".join(output_parts)

        else:  # markdown (default)
            arxiv_id = paper.entry_id.split("/")[-1]

            output_parts = [
                f"# {paper.title}",
                "",
                f"**Authors:** {', '.join(paper.authors)}",
                f"**Published:** {paper.published[:10]}",
                f"**Updated:** {paper.updated[:10] if paper.updated else 'N/A'}",
                f"**Primary Category:** {paper.primary_category}",
                f"**All Categories:** {', '.join(paper.categories)}",
                f"**ArXiv ID:** `{arxiv_id}`",
                f"**PDF:** [Download]({paper.pdf_url})" if paper.pdf_url else "**PDF:** N/A",
                f"**DOI:** {paper.doi}" if paper.doi else "**DOI:** N/A",
                f"**Journal Reference:** {paper.journal_ref}" if paper.journal_ref else "**Journal Reference:** N/A",
                f"**Comment:** {paper.comment}" if paper.comment else "**Comment:** N/A",
                "",
                "## Abstract",
                "",
                paper.summary,
            ]

            return "\n".join(output_parts)

    async def mcp_search_papers(
        self,
        query: str = Field(description="Search query (keywords, title, author, etc.)"),
        sort_by: str = Field(
            default="relevance", description="Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate'"
        ),
        sort_order: str = Field(default="descending", description="Sort order: 'ascending' or 'descending'"),
        category: str | None = Field(default=None, description="Filter by ArXiv category (e.g., 'cs.AI', 'math.CO')"),
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Search ArXiv papers with flexible criteria."""
        # Handle FieldInfo objects
        if isinstance(query, FieldInfo):
            query = query.default
        if isinstance(sort_by, FieldInfo):
            sort_by = sort_by.default
        if isinstance(sort_order, FieldInfo):
            sort_order = sort_order.default
        if isinstance(category, FieldInfo):
            category = category.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            print(f"🔍 Searching ArXiv for: {query}")

            start_time = datetime.now()

            # Build search query
            search_query = query
            if category:
                search_query = f"cat:{category} AND ({query})"

            # Configure sort criteria
            sort_criterion = arxiv.SortCriterion.Relevance
            if sort_by == "lastUpdatedDate":
                sort_criterion = arxiv.SortCriterion.LastUpdatedDate
            elif sort_by == "submittedDate":
                sort_criterion = arxiv.SortCriterion.SubmittedDate

            sort_order_enum = arxiv.SortOrder.Descending
            if sort_order == "ascending":
                sort_order_enum = arxiv.SortOrder.Ascending

            # Perform search
            search = arxiv.Search(
                query=search_query, max_results=300000, sort_by=sort_criterion, sort_order=sort_order_enum
            )

            # Execute search and collect results
            results = []
            for paper in self.client.results(search):
                results.append(self._format_paper_result(paper))

            execution_time = (datetime.now() - start_time).total_seconds()

            # Format output
            formatted_output = self._format_search_results(results, output_format)

            # Create metadata
            metadata = ArxivMetadata(
                operation="search_papers",
                query=query,
                sort_by=sort_by,
                sort_order=sort_order,
                total_results=len(results),
                execution_time=execution_time,
            )

            print(f"✅ Found {len(results)} papers in {execution_time:.2f}s")

            return ActionResponse(success=True, message=formatted_output, metadata=metadata.model_dump())

        except Exception as e:
            error_msg = f"Failed to search ArXiv papers: {str(e)}"
            self.logger.error(f"ArXiv search error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=ArxivMetadata(operation="search_papers", query=query, error_type="search_error").model_dump(),
            )

    async def mcp_get_paper_details(
        self,
        paper_id: str = Field(description="ArXiv paper ID (e.g., '2301.07041' or 'arxiv:2301.07041')"),
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Get detailed information about a specific ArXiv paper."""
        # Handle FieldInfo objects
        if isinstance(paper_id, FieldInfo):
            paper_id = paper_id.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Clean paper ID
            clean_id = paper_id.replace("arxiv:", "").strip()

            print(f"📄 Getting details for paper: {clean_id}")

            start_time = datetime.now()

            # Search for the specific paper
            search = arxiv.Search(id_list=[clean_id])

            paper = next(self.client.results(search), None)
            if not paper:
                return ActionResponse(
                    success=False,
                    message=f"Paper not found: {clean_id}",
                    metadata=ArxivMetadata(
                        operation="get_paper_details", paper_id=clean_id, error_type="paper_not_found"
                    ).model_dump(),
                )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Format paper details
            paper_result = self._format_paper_result(paper)
            formatted_output = self._format_paper_details(paper_result, output_format)

            # Create metadata
            metadata = ArxivMetadata(operation="get_paper_details", paper_id=clean_id, execution_time=execution_time)

            print(f"✅ Retrieved paper details in {execution_time:.2f}s")

            return ActionResponse(success=True, message=formatted_output, metadata=metadata.model_dump())

        except Exception as e:
            error_msg = f"Failed to get paper details: {str(e)}"
            self.logger.error(f"Paper details error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=ArxivMetadata(
                    operation="get_paper_details", paper_id=paper_id, error_type="retrieval_error"
                ).model_dump(),
            )

    async def mcp_download_paper(
        self,
        paper_id: str = Field(description="ArXiv paper ID (e.g., '2301.07041' or 'arxiv:2301.07041')"),
        extract_text: bool = Field(default=True, description="Whether to extract text content from PDF"),
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Download ArXiv paper PDF and optionally extract text content."""
        # Handle FieldInfo objects
        if isinstance(paper_id, FieldInfo):
            paper_id = paper_id.default
        if isinstance(extract_text, FieldInfo):
            extract_text = extract_text.default
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            # Clean paper ID
            clean_id = paper_id.replace("arxiv:", "").strip()

            print(f"📥 Downloading paper: {clean_id}")

            start_time = datetime.now()

            # Search for the paper
            search = arxiv.Search(id_list=[clean_id])
            paper = next(self.client.results(search), None)

            if not paper:
                return ActionResponse(
                    success=False,
                    message=f"Paper not found: {clean_id}",
                    metadata=ArxivMetadata(
                        operation="download_paper", paper_id=clean_id, error_type="paper_not_found"
                    ).model_dump(),
                )

            # Download PDF
            filename = f"{clean_id.replace('/', '_')}.pdf"
            download_path = self._downloads_dir / filename

            paper.download_pdf(dirpath=str(self._downloads_dir), filename=filename)

            execution_time = (datetime.now() - start_time).total_seconds()
            file_size = download_path.stat().st_size if download_path.exists() else 0

            # Prepare response message
            if output_format == "json":
                response_data = {
                    "paper_id": clean_id,
                    "title": paper.title,
                    "download_path": str(download_path),
                    "file_size": file_size,
                    "download_time": execution_time,
                }

                if extract_text:
                    try:
                        # Basic text extraction (would need additional libraries like PyPDF2 or pdfplumber)
                        response_data["text_extraction"] = (
                            "Text extraction requires additional PDF processing libraries"
                        )
                    except Exception:
                        response_data["text_extraction"] = "Text extraction failed"

                formatted_output = json.dumps(response_data, indent=2)

            elif output_format == "text":
                output_parts = [
                    "Paper Downloaded Successfully",
                    f"Paper ID: {clean_id}",
                    f"Title: {paper.title}",
                    f"Download Path: {download_path}",
                    f"File Size: {file_size:,} bytes",
                    f"Download Time: {execution_time:.2f} seconds",
                ]

                if extract_text:
                    output_parts.append("\nNote: Text extraction requires additional PDF processing libraries")

                formatted_output = "\n".join(output_parts)

            else:  # markdown (default)
                output_parts = [
                    "# 📥 Paper Download Complete",
                    "",
                    f"**Paper ID:** `{clean_id}`",
                    f"**Title:** {paper.title}",
                    f"**Download Path:** `{download_path}`",
                    f"**File Size:** {file_size:,} bytes",
                    f"**Download Time:** {execution_time:.2f} seconds",
                ]

                if extract_text:
                    output_parts.extend(
                        [
                            "",
                            "## 📄 Text Extraction",
                            (
                                "*Note: Text extraction requires additional "
                                "PDF processing libraries like PyPDF2 or pdfplumber*"
                            ),
                        ]
                    )

                formatted_output = "\n".join(output_parts)

            # Create metadata
            metadata = ArxivMetadata(
                operation="download_paper",
                paper_id=clean_id,
                download_path=str(download_path),
                file_size=file_size,
                execution_time=execution_time,
            )

            print(f"✅ Downloaded paper in {execution_time:.2f}s ({file_size:,} bytes)")

            return ActionResponse(success=True, message=formatted_output, metadata=metadata.model_dump())

        except Exception as e:
            error_msg = f"Failed to download paper: {str(e)}"
            self.logger.error(f"Paper download error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=ArxivMetadata(
                    operation="download_paper", paper_id=paper_id, error_type="download_error"
                ).model_dump(),
            )

    def mcp_get_categories(
        self,
        output_format: str = Field(default="markdown", description="Output format: 'markdown', 'json', or 'text'"),
    ) -> ActionResponse:
        """Get available ArXiv subject categories."""
        # Handle FieldInfo objects
        if isinstance(output_format, FieldInfo):
            output_format = output_format.default

        try:
            if output_format == "json":
                formatted_output = json.dumps(self.subject_categories, indent=2)

            elif output_format == "text":
                output_parts = ["ArXiv Subject Categories:\n"]
                for code, name in self.subject_categories.items():
                    output_parts.append(f"{code}: {name}")
                formatted_output = "\n".join(output_parts)

            else:  # markdown (default)
                output_parts = [
                    "# ArXiv Subject Categories",
                    "",
                    "Available categories for filtering search results:",
                    "",
                ]

                for code, name in self.subject_categories.items():
                    output_parts.append(f"- **`{code}`**: {name}")

                output_parts.extend(
                    [
                        "",
                        "## Usage Examples",
                        "- `cs.AI` - Artificial Intelligence",
                        "- `cs.LG` - Machine Learning",
                        "- `math.CO` - Combinatorics",
                        "- `physics.gen-ph` - General Physics",
                    ]
                )

                formatted_output = "\n".join(output_parts)

            metadata = ArxivMetadata(operation="get_categories", total_results=len(self.subject_categories))

            return ActionResponse(success=True, message=formatted_output, metadata=metadata.model_dump())

        except Exception as e:
            error_msg = f"Failed to get categories: {str(e)}"
            self.logger.error(f"Categories error: {traceback.format_exc()}")

            return ActionResponse(
                success=False,
                message=error_msg,
                metadata=ArxivMetadata(operation="get_categories", error_type="internal_error").model_dump(),
            )

    def mcp_get_arxiv_capabilities(self) -> ActionResponse:
        """Get information about ArXiv service capabilities and configuration."""
        capabilities = {
            "supported_operations": [
                "Paper search with flexible criteria",
                "Detailed paper metadata retrieval",
                "PDF download and storage",
                "Subject category filtering",
                "Multiple output formats (markdown, json, text)",
                "LLM-optimized result formatting",
            ],
            "search_capabilities": [
                "Keyword and phrase search",
                "Author name search",
                "Title search",
                "Category filtering",
                "Date-based sorting",
                "Relevance-based sorting",
            ],
            "supported_formats": ["markdown", "json", "text"],
            "configuration": {
                "downloads_directory": str(self._downloads_dir),
                "client_page_size": 100,
                "client_delay_seconds": 3.0,
                "client_num_retries": 3,
                "supported_categories_count": len(self.subject_categories),
            },
            "rate_limiting": {
                "delay_between_requests": "3.0 seconds",
                "retry_attempts": 3,
                "respectful_usage": "Configured for ArXiv server guidelines",
            },
        }

        formatted_info = f"""# ArXiv Service Capabilities
        
        ## Supported Operations
        {chr(10).join(f"- {op}" for op in capabilities["supported_operations"])}

        ## Search Capabilities  
        {chr(10).join(f"- {cap}" for cap in capabilities["search_capabilities"])}

        ## Supported Output Formats
        {chr(10).join(f"- {fmt}" for fmt in capabilities["supported_formats"])}

        ## Current Configuration
        - **Downloads Directory:** {capabilities["configuration"]["downloads_directory"]}
        - **Client Page Size:** {capabilities["configuration"]["client_page_size"]}
        - **Request Delay:** {capabilities["configuration"]["client_delay_seconds"]} seconds
        - **Retry Attempts:** {capabilities["configuration"]["client_num_retries"]}
        - **Available Categories:** {capabilities["configuration"]["supported_categories_count"]}

        ## Rate Limiting & Ethics
        - **Delay Between Requests:** {capabilities["rate_limiting"]["delay_between_requests"]}
        - **Retry Policy:** {capabilities["rate_limiting"]["retry_attempts"]} attempts
        - **Server Respect:** {capabilities["rate_limiting"]["respectful_usage"]}
        """

        return ActionResponse(success=True, message=formatted_info, metadata=capabilities)


# Default arguments for testing
if __name__ == "__main__":
    import os
    import sys
    import json
    import asyncio
    
    # Determine if running in MCP tool mode (called without arguments)
    is_mcp_mode = len(sys.argv) == 1
    if is_mcp_mode:
        original_print = print
        print = lambda *args, **kwargs: original_print(*args, file=sys.stderr, **kwargs)
    
    load_dotenv()
    
    arguments = ActionArguments(
        name="arxiv",
        transport="stdio",
        workspace=os.getenv("MASARENA_WORKSPACE", "~"),
    )
    
    try:
        service = ArxivActionCollection(arguments)
        if is_mcp_mode:
            input_line = sys.stdin.readline().strip()
            try:
                input_data = json.loads(input_line)
                function_name = input_data.get("function_name", input_data.get("name", "search_papers"))
                arguments = input_data.get("arguments", {})
                
                if function_name == "mcp_search_papers":
                    result = asyncio.run(service.mcp_search_papers(
                        query=arguments.get("query", ""),
                        sort_by=arguments.get("sort_by", "relevance"),
                        sort_order=arguments.get("sort_order", "descending"),
                        category=arguments.get("category", None),
                        output_format=arguments.get("output_format", "markdown")
                    ))
                elif function_name == "mcp_get_paper_details":
                    result = asyncio.run(service.mcp_get_paper_details(
                        paper_id=arguments.get("paper_id", ""),
                        output_format=arguments.get("output_format", "markdown")
                    ))
                elif function_name == "mcp_download_paper":
                    result = asyncio.run(service.mcp_download_paper(
                        paper_id=arguments.get("paper_id", ""),
                        extract_text=arguments.get("extract_text", True),
                        output_format=arguments.get("output_format", "markdown")
                    ))
                elif function_name == "mcp_get_categories":
                    result = service.mcp_get_categories(
                        output_format=arguments.get("output_format", "markdown")
                    )
                elif function_name == "mcp_get_arxiv_capabilities":
                    result = service.mcp_get_arxiv_capabilities()
                else:
                    result = ActionResponse(
                        success=False,
                        message=f"Unknown function: {function_name}",
                        metadata={"error_type": "unknown_function"}
                    )
                
                # Write result to stdout as JSON
                sys.stdout.write(json.dumps(result.model_dump()) + "\n")
                sys.stdout.flush()
                sys.exit(0)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Error parsing input JSON: {str(e)}\n")
                sys.exit(1)
            except Exception as e:
                sys.stderr.write(f"Error processing request: {str(e)}\n{traceback.format_exc()}\n")
                sys.exit(1)
        else:
            service.run()
    except Exception as e:
        print(f"An error occurred: {e}: {traceback.format_exc()}")
        sys.exit(1)
