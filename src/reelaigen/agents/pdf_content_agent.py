from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reelaigen.nodes.content_parser import ContentAnalysis, ContentParser
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig
from reelaigen.tools.agent_tools import build_analyze_content_tool, build_parse_pdf_tool


@dataclass
class PDFContentAgentResult:
    parsed_pdf: dict
    content_analysis: ContentAnalysis


class PDFContentAgent:
    def __init__(self, pdf_parser: PDFParser | None = None, content_parser: ContentParser | None = None) -> None:
        self.pdf_parser = pdf_parser or PDFParser(PDFParserConfig(save_page_images=True))
        self.content_parser = content_parser or ContentParser()
        self.parse_pdf_tool = build_parse_pdf_tool(self.pdf_parser)
        self.analyze_content_tool = build_analyze_content_tool(self.content_parser)

    def run(self, pdf_path: str | Path) -> PDFContentAgentResult:
        parsed_pdf = self.parse_pdf_tool.invoke({"pdf_path": str(pdf_path)})
        content_analysis = self.analyze_content_tool.invoke(
            {
                "document_text": parsed_pdf["text"],
                "pages": parsed_pdf["pages"],
            }
        )
        return PDFContentAgentResult(
            parsed_pdf=parsed_pdf,
            content_analysis=ContentAnalysis.model_validate(content_analysis),
        )
