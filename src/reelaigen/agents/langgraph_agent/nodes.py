from __future__ import annotations

from reelaigen.nodes.content_parser import ContentAnalysis, ContentParser
from reelaigen.nodes.pdf_parser import PDFParser
from reelaigen.nodes.script_writer import ScriptWriter

from .memory import add_memory_event, create_initial_context, create_initial_memory, update_context
from .state import PDFContentAgentState


class GraphNodes:
    def __init__(
        self,
        pdf_parser: PDFParser,
        content_parser: ContentParser,
        script_writer: ScriptWriter,
        algorithm_parser=None,
    ) -> None:
        self.pdf_parser = pdf_parser
        self.content_parser = content_parser
        self.script_writer = script_writer
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

    def summary(self, state: PDFContentAgentState) -> dict:
        return {
            "context": update_context(state, "summary"),
            "memory": add_memory_event(state, "summary", "Prepared final pipeline output."),
            "final_output": {
                "user_prompt": state.get("user_prompt", {}),
                "memory": state.get("memory", {}),
                "context": state.get("context", {}),
                "parsed_pdf": state["parsed_pdf"],
                # "algorithm_analysis": state["algorithm_analysis"],
                "content_analysis": state["content_analysis"],
                "script_plan": state["script_plan"],
            },
        }
