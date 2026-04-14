from __future__ import annotations

from langchain_core.tools import tool

from reelaigen.nodes.content_parser import ContentParser
from reelaigen.nodes.pdf_parser import PDFParser


def build_parse_pdf_tool(pdf_parser: PDFParser):
    @tool("parse_pdf")
    def parse_pdf(pdf_path: str) -> dict:
        """Parse a PDF and return text, metadata, and page data."""
        result = pdf_parser.run(pdf_path)
        return {
            "text": result.text,
            "metadata": result.metadata,
            "pages": [
                {
                    "number": page.number,
                    "text": page.text,
                    "image_path": str(page.image_path) if page.image_path else None,
                    "image_id": f"image_{page.number}",
                }
                for page in result.pages
            ],
        }

    return parse_pdf


def build_analyze_content_tool(content_parser: ContentParser):
    @tool("analyze_content")
    def analyze_content(document_text: str, pages: list[dict]) -> dict:
        """Analyze parsed PDF text and related page images."""
        result = content_parser.run(document_text, images=pages)
        return result.model_dump()

    return analyze_content
