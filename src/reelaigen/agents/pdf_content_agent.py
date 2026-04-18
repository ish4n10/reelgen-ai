from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import StructuredTool

from reelaigen.agents.langgraph_agent.state import (
    AlgorithmAnalysisPayload,
    PDFContentAgentState,
    ParsedPDFPayload,
)
from reelaigen.nodes.algorithm_parser import AlgorithmAnalysis, AlgorithmParser
from reelaigen.nodes.content_parser import ContentAnalysis, ContentParser
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig
from reelaigen.nodes.script_writer import ScriptPlan, ScriptWriter


@dataclass
class PDFContentAgentResult:
    parsed_pdf: ParsedPDFPayload
    algorithm_analysis: AlgorithmAnalysis
    content_analysis: ContentAnalysis
    script_plan: ScriptPlan


class PDFContentAgent:
    def __init__(
        self,
        pdf_parser: PDFParser | None = None,
        algorithm_parser: AlgorithmParser | None = None,
        content_parser: ContentParser | None = None,
        script_writer: ScriptWriter | None = None,
    ) -> None:
        self.pdf_parser = pdf_parser or PDFParser(PDFParserConfig(save_page_images=True))
        self.algorithm_parser = algorithm_parser or AlgorithmParser()
        self.content_parser = content_parser or ContentParser()
        self.script_writer = script_writer or ScriptWriter()
        self.parse_pdf_tool = StructuredTool.from_function(
            func=self.parse_pdf,
            name="parse_pdf",
            description="Parse a PDF and return text, metadata, and page data.",
        )
        self.analyze_algorithm_tool = StructuredTool.from_function(
            func=self.analyze_algorithm,
            name="analyze_algorithm",
            description="Detect and simulate algorithms from parsed PDF text.",
        )
        self.analyze_content_tool = StructuredTool.from_function(
            func=self.analyze_content,
            name="analyze_content",
            description="Analyze parsed PDF text and related page images.",
        )
        self.write_script_tool = StructuredTool.from_function(
            func=self.write_script,
            name="write_script",
            description="Write narration scripts for analyzed sections.",
        )
        self.tools = [
            self.parse_pdf_tool,
            self.analyze_algorithm_tool,
            self.analyze_content_tool,
            self.write_script_tool,
        ]

    def parse_pdf(self, pdf_path: str) -> ParsedPDFPayload:
        result = self.pdf_parser.run(pdf_path)
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

    def analyze_algorithm(self, document_text: str) -> AlgorithmAnalysisPayload:
        result = self.algorithm_parser.run(document_text)
        return result.model_dump()

    def analyze_content(self, document_text: str, pages: list[dict], algorithm_context: dict | None = None) -> dict:
        result = self.content_parser.run(document_text, images=pages, algorithm_context=algorithm_context)
        return result.model_dump()

    def write_script(
        self,
        document_text: str,
        sections: list[dict],
        pages: list[dict],
        algorithm_context: dict | None = None,
    ) -> dict:
        validated_content = ContentAnalysis.model_validate(
            {
                "parent_content_type": "unused_here",
                "sections": sections,
            }
        )
        result = self.script_writer.run(
            document_text=document_text,
            sections=validated_content.sections,
            pages=pages,
            algorithm_context=algorithm_context,
        )
        return result.model_dump()

    def _build_initial_state(self, pdf_path: str | Path) -> PDFContentAgentState:
        return {"pdf_path": str(pdf_path)}

    def run(self, pdf_path: str | Path) -> PDFContentAgentResult:
        state = self._build_initial_state(pdf_path)
        state["parsed_pdf"] = self.parse_pdf_tool.invoke({"pdf_path": state["pdf_path"]})
        state["algorithm_analysis"] = self.analyze_algorithm_tool.invoke(
            {
                "document_text": state["parsed_pdf"]["text"],
            }
        )
        state["content_analysis"] = self.analyze_content_tool.invoke(
            {
                "document_text": state["parsed_pdf"]["text"],
                "pages": state["parsed_pdf"]["pages"],
                "algorithm_context": state["algorithm_analysis"],
            }
        )
        state["script_plan"] = self.write_script_tool.invoke(
            {
                "document_text": state["parsed_pdf"]["text"],
                "sections": state["content_analysis"]["sections"],
                "pages": state["parsed_pdf"]["pages"],
                "algorithm_context": state["algorithm_analysis"],
            }
        )
        return PDFContentAgentResult(
            parsed_pdf=state["parsed_pdf"],
            algorithm_analysis=AlgorithmAnalysis.model_validate(state["algorithm_analysis"]),
            content_analysis=ContentAnalysis.model_validate(state["content_analysis"]),
            script_plan=ScriptPlan.model_validate(state["script_plan"]),
        )
