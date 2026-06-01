from typing import Optional, Dict, Any, List
import os
from pathlib import Path
import logging

import arxiv
from smolagents import Tool
from smolagents.models import Model

# Configure logging
logger = logging.getLogger(__name__)


class ArxivTool(Tool):
    name = "search_arxiv_papers"
    description = """
Search and retrieve academic papers from ArXiv. 
This tool can search papers by keywords, authors, categories, or specific ArXiv IDs.
It can also get detailed information about specific papers and download paper content.
"""

    inputs = {
        "action": {
            "description": "Action to perform: 'search' (search papers), 'details' (get paper details), 'categories' (list categories)",
            "type": "string",
        },
        "query": {
            "description": "Search query for papers (keywords, title, author, etc.). Required for 'search' action.",
            "type": "string",
            "nullable": True,
        },
        "paper_id": {
            "description": "ArXiv paper ID (e.g., '2301.07041'). Required for 'details' action.",
            "type": "string",
            "nullable": True,
        },
        "category": {
            "description": "Filter by ArXiv category (e.g., 'cs.AI', 'math.CO'). Optional for 'search' action.",
            "type": "string",
            "nullable": True,
        },
        "max_results": {
            "description": "Maximum number of results to return (default: 10, max: 50)",
            "type": "integer",
            "nullable": True,
        },
        "sort_by": {
            "description": "Sort by: 'relevance', 'lastUpdatedDate', 'submittedDate' (default: 'relevance')",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: Model, text_limit: int = 8000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

        # ArXiv client configuration
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3,
        )

        # Create downloads directory
        self.workspace = Path(os.path.expanduser("~")) / "arxiv_downloads"
        self.workspace.mkdir(exist_ok=True)

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

    def _format_paper_result(self, paper: arxiv.Result) -> Dict[str, Any]:
        """Convert arxiv.Result to structured dictionary."""
        try:
            return {
                "entry_id": paper.entry_id,
                "title": paper.title.strip(),
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary.strip(),
                "published": paper.published.isoformat(),
                "updated": paper.updated.isoformat() if paper.updated else None,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "pdf_url": paper.pdf_url,
                "doi": paper.doi,
                "journal_ref": paper.journal_ref,
                "comment": paper.comment,
            }
        except AttributeError as e:
            logger.warning(f"Missing attribute in paper result: {e}")
            # Safe fallback with proper type handling
            published_attr = getattr(paper, "published", None)
            updated_attr = getattr(paper, "updated", None)

            return {
                "entry_id": getattr(paper, "entry_id", ""),
                "title": getattr(paper, "title", "").strip(),
                "authors": [author.name for author in getattr(paper, "authors", [])],
                "summary": getattr(paper, "summary", "").strip(),
                "published": published_attr.isoformat() if published_attr else "",
                "updated": updated_attr.isoformat() if updated_attr else None,
                "categories": getattr(paper, "categories", []),
                "primary_category": getattr(paper, "primary_category", ""),
                "pdf_url": getattr(paper, "pdf_url", ""),
                "doi": getattr(paper, "doi", ""),
                "journal_ref": getattr(paper, "journal_ref", ""),
                "comment": getattr(paper, "comment", ""),
            }

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for display."""
        output_parts = [f"# ArXiv Search Results\n\nFound **{len(results)}** papers:\n"]

        for i, paper in enumerate(results, 1):
            # Format authors (show first 3, then indicate if there are more)
            authors_list = paper.get("authors", [])
            if len(authors_list) <= 3:
                authors_str = ", ".join(authors_list)
            else:
                authors_str = (
                    f"{', '.join(authors_list[:3])} et al. ({len(authors_list)} total)"
                )

            arxiv_id = paper["entry_id"].split("/")[-1]

            # Truncate abstract if too long
            abstract = paper.get("summary", "")
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            output_parts.extend(
                [
                    f"## {i}. {paper.get('title', 'Untitled')}",
                    f"**Authors:** {authors_str}",
                    f"**Published:** {paper.get('published', '')[:10]}",
                    f"**Categories:** {', '.join(paper.get('categories', []))}",
                    f"**ArXiv ID:** `{arxiv_id}`",
                    (
                        f"**PDF:** [Download]({paper['pdf_url']})"
                        if paper.get("pdf_url")
                        else ""
                    ),
                    "",
                    f"**Abstract:** {abstract}",
                    "",
                    "---",
                    "",
                ]
            )

        return "\n".join(filter(None, output_parts))

    def _search_papers(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> str:
        """Search ArXiv papers."""
        try:
            # Build search query
            search_query = query
            if category:
                search_query += f" AND cat:{category}"

            # Configure sort order
            sort_criteria = arxiv.SortCriterion.Relevance
            if sort_by == "lastUpdatedDate":
                sort_criteria = arxiv.SortCriterion.LastUpdatedDate
            elif sort_by == "submittedDate":
                sort_criteria = arxiv.SortCriterion.SubmittedDate

            # Perform search
            search = arxiv.Search(
                query=search_query,
                max_results=min(max_results, 50),
                sort_by=sort_criteria,
                sort_order=arxiv.SortOrder.Descending,
            )

            results = []
            for paper in self.client.results(search):
                results.append(self._format_paper_result(paper))

            if not results:
                return "No papers found matching the search criteria."

            # Format results for LLM
            return self._format_search_results(results)

        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return f"Error searching papers: {str(e)}"

    def _get_paper_details(self, paper_id: str) -> str:
        """Get detailed information about a specific ArXiv paper."""
        try:
            # Clean paper ID
            if paper_id.startswith("arxiv:"):
                paper_id = paper_id[6:]

            # Search for the specific paper
            search = arxiv.Search(id_list=[paper_id])

            try:
                paper = next(iter(self.client.results(search)))
            except StopIteration:
                return f"Paper with ID '{paper_id}' not found."

            paper_data = self._format_paper_result(paper)

            # Format detailed information
            authors_str = ", ".join(paper_data["authors"])

            output_parts = [
                f"# {paper_data['title']}",
                "",
                f"**ArXiv ID:** `{paper_data['entry_id'].split('/')[-1]}`",
                f"**Authors:** {authors_str}",
                f"**Published:** {paper_data['published'][:10]}",
                f"**Updated:** {paper_data['updated'][:10] if paper_data['updated'] else 'N/A'}",
                f"**Primary Category:** {paper_data['primary_category']}",
                f"**Categories:** {', '.join(paper_data['categories'])}",
                "",
                (
                    f"**PDF URL:** {paper_data['pdf_url']}"
                    if paper_data["pdf_url"]
                    else ""
                ),
                f"**DOI:** {paper_data['doi']}" if paper_data["doi"] else "",
                (
                    f"**Journal Reference:** {paper_data['journal_ref']}"
                    if paper_data["journal_ref"]
                    else ""
                ),
                (
                    f"**Comment:** {paper_data['comment']}"
                    if paper_data["comment"]
                    else ""
                ),
                "",
                "## Abstract",
                "",
                paper_data["summary"],
            ]

            return "\n".join(filter(None, output_parts))

        except Exception as e:
            logger.error(f"Error retrieving paper details: {str(e)}")
            return f"Error retrieving paper details: {str(e)}"

    def _get_categories(self) -> str:
        """Get available ArXiv subject categories."""
        output_parts = ["# ArXiv Subject Categories\n"]

        for code, name in self.subject_categories.items():
            output_parts.append(f"- **{code}:** {name}")

        output_parts.extend(
            [
                "",
                "## Usage Examples:",
                "- Use category codes for filtering: 'cs.AI' for Artificial Intelligence",
                "- Use broader categories: 'cs' for all Computer Science papers",
                "- Combine with search: search for 'machine learning' in category 'cs.LG'",
            ]
        )

        return "\n".join(output_parts)

    def forward(
        self,
        action: str,
        query: Optional[str] = None,
        paper_id: Optional[str] = None,
        category: Optional[str] = None,
        max_results: Optional[int] = None,
        sort_by: Optional[str] = None,
    ) -> str:
        """Main entry point for the ArXiv tool."""
        # Validate inputs
        if not action:
            return "Error: 'action' parameter is required."

        action = action.lower().strip()

        # Set defaults
        max_results = max_results or 10
        sort_by = sort_by or "relevance"

        # Validate max_results
        if max_results < 1:
            max_results = 1
        elif max_results > 50:
            max_results = 50

        # Validate sort_by
        valid_sort_options = ["relevance", "lastUpdatedDate", "submittedDate"]
        if sort_by not in valid_sort_options:
            return f"Error: Invalid sort_by option. Must be one of: {', '.join(valid_sort_options)}"

        if action == "search":
            if not query or not query.strip():
                return "Error: 'query' parameter is required and cannot be empty for search action."
            return self._search_papers(query.strip(), category, max_results, sort_by)

        elif action == "details":
            if not paper_id or not paper_id.strip():
                return "Error: 'paper_id' parameter is required and cannot be empty for details action."
            return self._get_paper_details(paper_id.strip())

        elif action == "categories":
            return self._get_categories()

        else:
            return f"Error: Unknown action '{action}'. Available actions: 'search', 'details', 'categories'"
