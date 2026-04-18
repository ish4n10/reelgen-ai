from __future__ import annotations

from pathlib import Path
from typing import Any

from reelaigen.nodes.algorithm_parser import AlgorithmParser
from reelaigen.nodes.content_parser import ContentParser
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig
from reelaigen.nodes.script_writer import ScriptWriter
from reelaigen.nodes.visual_planner import VisualPlanner

from .graph import build_graph
from .nodes import GraphNodes


class ReelAIGraphAgent:
    def __init__(
        self,
        pdf_parser: PDFParser | None = None,
        algorithm_parser: AlgorithmParser | None = None,
        content_parser: ContentParser | None = None,
        script_writer: ScriptWriter | None = None,
        visual_planner: VisualPlanner | None = None,
    ) -> None:
        self.pdf_parser = pdf_parser or PDFParser(PDFParserConfig(save_page_images=True))
        self.algorithm_parser = algorithm_parser or AlgorithmParser()
        self.content_parser = content_parser or ContentParser()
        self.script_writer = script_writer or ScriptWriter()
        self.visual_planner = visual_planner or VisualPlanner()

    def build(self):
        nodes = GraphNodes(
            pdf_parser=self.pdf_parser,
            content_parser=self.content_parser,
            script_writer=self.script_writer,
            visual_planner=self.visual_planner,
            algorithm_parser=self.algorithm_parser,
        )
        return build_graph(nodes)

    def run(
        self,
        pdf_path: str | Path,
        user_prompt: dict[str, Any] | None = None,
        thread_id: str = "default",
    ) -> dict:
        app = self.build()
        return app.invoke(
            {
                "pdf_path": str(pdf_path),
                "user_prompt": user_prompt or {},
                "context": {
                    "thread_id": thread_id,
                    "current_node": "",
                    "completed_nodes": [],
                },
            },
            config={"configurable": {"thread_id": thread_id}},
        )
