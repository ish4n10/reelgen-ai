from __future__ import annotations

from reelaigen.nodes.content_parser import ContentAnalysis, ContentParser
from reelaigen.nodes.pdf_parser import PDFParser
from reelaigen.nodes.script_writer import ScriptPlan, ScriptWriter
from reelaigen.nodes.visual_planner import VisualPlanner

from .memory import add_memory_event, create_initial_context, create_initial_memory, update_context
from .state import PDFContentAgentState


class GraphNodes:
    def __init__(
        self,
        pdf_parser: PDFParser,
        content_parser: ContentParser,
        script_writer: ScriptWriter,
        visual_planner: VisualPlanner,
        algorithm_parser=None,
    ) -> None:
        self.pdf_parser = pdf_parser
        self.content_parser = content_parser
        self.script_writer = script_writer
        self.visual_planner = visual_planner
        self.algorithm_parser = algorithm_parser

    def initialize(self, state: PDFContentAgentState) -> dict:
        thread_id = state.get("context", {}).get("thread_id", "default")
        return {
            "memory": create_initial_memory(),
            "context": create_initial_context(thread_id),
            "user_prompt": state.get("user_prompt", {}),
        }

    def parse_pdf(self, state: PDFContentAgentState) -> dict:
        result = self.pdf_parser.run(state["pdf_path"])

        pages = []
        for page in result.pages:
            pages.append(
                {
                    "number": page.number,
                    "text": page.text,
                    "image_path": str(page.image_path) if page.image_path else None,
                    "image_id": f"image_{page.number}",
                }
            )

        return {
            "context": update_context(state, "parse_pdf"),
            "memory": add_memory_event(state, "parse_pdf", "Parsed PDF text, metadata, and page images."),
            "parsed_pdf": {
                "text": result.text,
                "metadata": result.metadata,
                "pages": pages,
            },
        }

    # Keep this node for later when algorithm grounding is enabled again.
    def algorithm_parser(self, state: PDFContentAgentState) -> dict:
        result = self.algorithm_parser.run(state["parsed_pdf"]["text"])
        return {
            "context": update_context(state, "algorithm_parser"),
            "memory": add_memory_event(
                state,
                "algorithm_parser",
                f"Detected algorithm: {result.algorithm_name or 'none'}",
            ),
            "algorithm_analysis": result.model_dump(),
        }

    def content_parser_node(self, state: PDFContentAgentState) -> dict:
        result = self.content_parser.run(
            document_text=state["parsed_pdf"]["text"],
            images=state["parsed_pdf"]["pages"],
            algorithm_context=state.get("algorithm_analysis"),
        )
        return {
            "context": update_context(state, "content_parser"),
            "memory": add_memory_event(
                state,
                "content_parser",
                f"Split document into {len(result.sections)} sections.",
            ),
            "content_analysis": result.model_dump(),
        }

    def script_writer_node(self, state: PDFContentAgentState) -> dict:
        content_analysis = ContentAnalysis.model_validate(state["content_analysis"])
        result = self.script_writer.run(
            document_text=state["parsed_pdf"]["text"],
            sections=content_analysis.sections,
            pages=state["parsed_pdf"]["pages"],
            algorithm_context=state.get("algorithm_analysis"),
        )
        return {
            "context": update_context(state, "script_writer"),
            "memory": add_memory_event(
                state,
                "script_writer",
                f"Wrote scripts for {len(result.sections)} sections.",
            ),
            "script_plan": result.model_dump(),
        }

    def visual_planner_node(self, state: PDFContentAgentState) -> dict:
        content_analysis = ContentAnalysis.model_validate(state["content_analysis"])
        script_plan = ScriptPlan.model_validate(state["script_plan"])

        result = self.visual_planner.run(
            document_text=state["parsed_pdf"]["text"],
            sections=content_analysis.sections,
            script_sections=script_plan.sections,
            pages=state["parsed_pdf"]["pages"],
        )
        return {
            "context": update_context(state, "visual_planner"),
            "memory": add_memory_event(
                state,
                "visual_planner",
                f"Planned visuals for {len(result.sections)} sections.",
            ),
            "visual_plan": result.model_dump(),
        }

    def summary(self, state: PDFContentAgentState) -> dict:
        final_sections = []
        script_sections = state.get("script_plan", {}).get("sections", [])
        visual_sections = state.get("visual_plan", {}).get("sections", [])

        for script_section in script_sections:
            section_id = script_section["section_id"]
            visual_section = self._find_section_by_id(visual_sections, section_id)
            final_sections.append(
                {
                    "sectionId": section_id,
                    "script": script_section,
                    "visual": visual_section or {},
                }
            )

        return {
            "context": update_context(state, "summary"),
            "memory": add_memory_event(state, "summary", "Prepared final pipeline output."),
            "final_output": {
                "sections": final_sections,
            },
        }

    def _find_section_by_id(self, sections: list[dict], section_id: int) -> dict | None:
        for section in sections:
            if section.get("section_id") == section_id:
                return section
        return None
