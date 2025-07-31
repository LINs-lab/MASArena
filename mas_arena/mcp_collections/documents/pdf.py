# import os
# import time
# import traceback
# import json
# from collections import defaultdict
# from pathlib import Path
# from typing import Any, Literal, Optional
# import markdown
# from dotenv import load_dotenv
#
# from marker.convert import convert_single_pdf
# from marker.models import load_all_models
# from pydantic import Field
# from pydantic.fields import FieldInfo
#
# from ..base import ActionArguments, ActionCollection, ActionResponse
# from ..documents.models import DocumentMetadata
#
#
# class DocumentExtractionCollection(ActionCollection):
#     tool_name = "pdf"
#     """MCP service for PDF document content extraction using marker package.
#
#     Supports extraction from PDF files only.
#     Provides LLM-friendly text output with structured metadata and media file handling.
#     """
#
#     def __init__(self, arguments: ActionArguments) -> None:
#         super().__init__(arguments)
#         self._models_loaded = False
#         self._marker_models = None
#         self._media_output_dir = self.workspace / "extracted_media"
#         self._media_output_dir.mkdir(exist_ok=True)
#         self._extracted_texts_dir = self.workspace / "extracted_texts"  # New directory for text files
#         self._extracted_texts_dir.mkdir(exist_ok=True)
#
#         self.supported_extensions = {".pdf"}
#
#         print("PDF Extraction Service initialized")
#         print(f"Media output directory: {self._media_output_dir}")
#
#     def _load_marker_models(self) -> None:
#         """Load marker models for document processing.
#
#         Lazy loading to avoid unnecessary resource consumption.
#         """
#         if not self._models_loaded:
#             try:
#                 print("Loading marker models...")
#                 self._marker_models = load_all_models()
#                 self._models_loaded = True
#                 print("Marker models loaded successfully")
#             except Exception as e:
#                 self.logger.error(f"Failed to load marker models: {str(e)}")
#                 raise
#
#     def _extract_content_with_marker(
#         self, file_path: Path, force_ocr: bool = False
#     ) -> dict[str, Any]:
#         """Extract content using marker package.
#
#         Args:
#             file_path: Path to the document file
#             force_ocr: Use OCR to extract text from images if available
#
#         Returns:
#             Dictionary containing extracted content and metadata
#         """
#         start_time = time.time()
#
#         original_ocr_all_pages = os.environ.get("OCR_ALL_PAGES")
#         if force_ocr:
#             os.environ["OCR_ALL_PAGES"] = "true"
#
#         try:
#             text, images, metadata = convert_single_pdf(str(file_path), self._marker_models)
#         finally:
#             if original_ocr_all_pages is None:
#                 if "OCR_ALL_PAGES" in os.environ:
#                     del os.environ["OCR_ALL_PAGES"]
#             else:
#                 os.environ["OCR_ALL_PAGES"] = original_ocr_all_pages
#
#         processing_time = time.time() - start_time
#         return {
#             "content": text,
#             "images": images or {},
#             "metadata": metadata or defaultdict(),
#             "processing_time": processing_time,
#         }
#
#     def _save_extracted_media(self, images: dict[str, Any], file_stem: str) -> list[dict[str, str]]:
#         """Save extracted images and return their paths.
#
#         Args:
#             images: Dictionary of extracted images from marker
#             file_stem: Base name for saving files
#
#         Returns:
#             list of dictionaries containing media type and file paths
#         """
#         saved_media = []
#
#         for idx, (page_num, image_data) in enumerate(images.items()):
#             try:
#                 # Generate unique filename
#                 image_filename = f"{file_stem}_page_{page_num}_img_{idx}.png"
#                 image_path = self._media_output_dir / image_filename
#
#                 # Save image data
#                 if hasattr(image_data, "save"):
#                     # PIL Image object
#                     image_data.save(image_path)
#                 elif isinstance(image_data, bytes):
#                     # Raw image bytes
#                     with open(image_path, "wb") as f:
#                         f.write(image_data)
#                 else:
#                     # Handle other formats
#                     self.logger.warning(f"Unknown image data type for page {page_num}: {type(image_data)}")
#                     continue
#
#                 saved_media.append(
#                     {"type": "image", "path": str(image_path), "page": str(page_num), "filename": image_filename}
#                 )
#
#                 print(f"Saved image: {image_filename}")
#
#             except Exception as e:
#                 self.logger.error(f"Failed to save image from page {page_num}: {str(e)}")
#
#         return saved_media
#
#     def _format_content_for_llm(self, content: str, output_format: str) -> str:
#         """Format extracted content to be LLM-friendly.
#
#         Args:
#             content: Raw extracted content
#             output_format: Desired output format
#
#         Returns:
#             Formatted content string
#         """
#         if output_format.lower() == "markdown":
#             # Content is already in markdown format from marker
#             return content
#         elif output_format.lower() == "json":
#             # Structure content as JSON
#
#             return json.dumps({"content": content, "format": "structured_text"}, indent=2)
#         elif output_format.lower() == "html":
#             # Convert markdown to HTML if needed
#             try:
#                 return markdown.markdown(content)
#             except ImportError:
#                 self.logger.warning("markdown package not available, returning raw content")
#                 return content
#         else:
#             return content
#
#     def mcp_extract_document_content(
#         self,
#         file_path: str = Field(description="Path to the PDF document file to extract content from"),
#         output_format: Literal["markdown", "json", "html"] = Field(
#             default="markdown", description="Output format: 'markdown', 'json', or 'html'"
#         ),
#         extract_images: bool = Field(default=True, description="Whether to extract and save images from the document"),
#         save_extracted_text_to_file: bool = Field(
#             default=False, description="Save extracted text to a local file"
#         ),  # New parameter
#         use_llm: bool = Field(default=False, description="Use LLM for enhanced accuracy (requires additional setup)"),
#         force_ocr: bool = Field(default=False, description="Force OCR processing on the entire document"),
#         format_lines: bool = Field(
#             default=False, description="Reformat lines using local OCR model for better quality"
#         ),
#     ) -> ActionResponse:
#         """Extract content from PDF documents using marker package.
#
#         This tool provides comprehensive PDF document content extraction with support for:
#         - PDF files
#         - Text extraction with proper formatting
#         - Image and media extraction
#         - Metadata collection
#         - LLM-optimized output formatting
#
#         Args:
#             args: Document extraction arguments including file path and options
#
#         Returns:
#             ActionResponse with extracted content, metadata, and media file paths
#         """
#         try:
#             if isinstance(file_path, FieldInfo):
#                 file_path = file_path.default
#             if isinstance(output_format, FieldInfo):
#                 output_format = output_format.default
#             if isinstance(extract_images, FieldInfo):
#                 extract_images = extract_images.default
#             if isinstance(save_extracted_text_to_file, FieldInfo):  # Handle new parameter
#                 save_extracted_text_to_file = save_extracted_text_to_file.default
#             if isinstance(use_llm, FieldInfo):
#                 use_llm = use_llm.default
#             if isinstance(force_ocr, FieldInfo):
#                 force_ocr = force_ocr.default
#             if isinstance(format_lines, FieldInfo):
#                 format_lines = format_lines.default
#
#             # Validate input file
#             file_path: Path = self._validate_file_path(file_path)
#             print(f"Processing document: {file_path.name}")
#
#             # Load marker models if needed
#             self._load_marker_models()
#
#             # Extract content using marker
#             extraction_result = self._extract_content_with_marker(file_path, force_ocr)
#
#             # Save extracted media if requested
#             saved_media = []
#             if extract_images and extraction_result["images"]:
#                 saved_media = self._save_extracted_media(extraction_result["images"], file_path.stem)
#
#             # Format content for LLM consumption
#             formatted_content = self._format_content_for_llm(extraction_result["content"], output_format)
#
#             # Save extracted text to file if requested
#             saved_text_path_str: Optional[str] = None
#             if save_extracted_text_to_file:
#                 text_file_name = f"{file_path.stem}_extracted_text.txt"
#                 saved_text_path = self._extracted_texts_dir / text_file_name
#                 try:
#                     with open(saved_text_path, "w", encoding="utf-8") as f:
#                         f.write(formatted_content)
#                     saved_text_path_str = str(saved_text_path.absolute())
#                     print(f"Saved extracted text to: {saved_text_path_str}")
#                 except Exception as e:
#                     self.logger.error(f"Failed to save extracted text to {saved_text_path}: {str(e)}")
#                     # Optionally, you might want to reflect this failure in the response
#
#             # Prepare metadata
#             file_stats = file_path.stat()
#             document_metadata = DocumentMetadata(
#                 file_name=file_path.name,
#                 file_size=file_stats.st_size,
#                 file_type=file_path.suffix.lower(),
#                 absolute_path=str(file_path.absolute()),
#                 page_count=extraction_result["metadata"].get("pages"),
#                 processing_time=extraction_result["processing_time"],
#                 extracted_images=[media["path"] for media in saved_media if media["type"] == "image"],
#                 extracted_media=saved_media,
#                 output_format=output_format,
#                 llm_enhanced=use_llm,
#                 ocr_applied=force_ocr or format_lines,
#                 extracted_text_file_path=saved_text_path_str,
#             )
#
#             print(
#                 f"Successfully extracted content from {file_path.name} "
#                 f"({len(formatted_content)} characters, {len(saved_media)} media files)"
#             )
#
#             return ActionResponse(success=True, message=formatted_content, metadata=document_metadata.model_dump())
#
#         except FileNotFoundError as e:
#             self.logger.error(f"File not found: {str(e)}: {traceback.format_exc()}")
#             return ActionResponse(
#                 success=False, message=f"File not found: {str(e)}", metadata={"error_type": "file_not_found"}
#             )
#         except ValueError as e:
#             self.logger.error(f"Invalid input: {str(e)}: {traceback.format_exc()}")
#             return ActionResponse(
#                 success=False,
#                 message=f"Invalid input: {str(e)}: {traceback.format_exc()}",
#                 metadata={"error_type": "invalid_input"},
#             )
#         except Exception as e:
#             self.logger.error(f"Document extraction failed: {str(e)}: {traceback.format_exc()}")
#             return ActionResponse(
#                 success=False,
#                 message=f"Document extraction failed: {str(e)}",
#                 metadata={"error_type": "extraction_error"},
#             )
#
#     def mcp_list_supported_formats(self) -> ActionResponse:
#         """list all supported document formats for extraction.
#
#         Returns:
#             ActionResponse with list of supported file formats and their descriptions
#         """
#         supported_formats = {
#             "PDF": "Portable Document Format files (.pdf)",
#         }
#
#         format_list = "\n".join(
#             [f"**{format_name}**: {description}" for format_name, description in supported_formats.items()]
#         )
#
#         return ActionResponse(
#             success=True,
#             message=f"Supported document formats:\n\n{format_list}",
#             metadata={"supported_formats": list(supported_formats.keys()), "total_formats": len(supported_formats)},
#         )
#
#
# # Example usage and entry point
# if __name__ == "__main__":
#     import sys
#
#     is_mcp_mode = len(sys.argv) == 1
#     if is_mcp_mode:
#         original_print = print
#         print = lambda *args, **kwargs: original_print(*args, file=sys.stderr, **kwargs)
#
#     load_dotenv()
#     args = ActionArguments(
#         name="document_extraction_service",
#         transport="stdio",
#         workspace=os.getenv("MASARENA_WORKSPACE", "~"),
#     )
#
#     try:
#         service = DocumentExtractionCollection(args)
#         if is_mcp_mode:
#             input_line = sys.stdin.readline().strip()
#             try:
#                 input_data = json.loads(input_line)
#                 function_name = input_data.get("function_name", input_data.get("name", "extract_document_content"))
#                 arguments = input_data.get("arguments", {})
#
#                 if function_name == "extract_document_content":
#                     result = service.mcp_extract_document_content(
#                         file_path=arguments.get("file_path", ""),
#                         output_format=arguments.get("output_format", "markdown"),
#                         extract_images=arguments.get("extract_images", True),
#                         save_extracted_text_to_file=arguments.get("save_extracted_text_to_file", False),
#                         use_llm=arguments.get("use_llm", False),
#                         force_ocr=arguments.get("force_ocr", False),
#                         format_lines=arguments.get("format_lines", False)
#                     )
#                 elif function_name == "list_supported_formats":
#                     result = service.mcp_list_supported_formats()
#                 else:
#                     result = ActionResponse(
#                         success=False,
#                         message=f"Unknown function: {function_name}",
#                         metadata={"error_type": "unknown_function"}
#                     )
#
#                 # Write result to stdout as JSON
#                 sys.stdout.write(json.dumps(result.model_dump()) + "\n")
#                 sys.stdout.flush()
#                 sys.exit(0)
#             except json.JSONDecodeError as e:
#                 sys.stderr.write(f"Error parsing input JSON: {str(e)}\n")
#                 sys.exit(1)
#             except Exception as e:
#                 sys.stderr.write(f"Error processing request: {str(e)}\n{traceback.format_exc()}\n")
#                 sys.exit(1)
#         else:
#             service.run()
#     except Exception as e:
#         print(f"An error occurred: {e}: {traceback.format_exc()}")
#         sys.exit(1)
