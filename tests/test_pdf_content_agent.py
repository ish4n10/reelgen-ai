from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelaigen.agents.langgraph_agent import ReelAIGraphAgent
from reelaigen.nodes.content_parser import ContentParser
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig
from reelaigen.nodes.script_writer import ScriptWriter
from reelaigen.nodes.visual_planner import VisualPlanner

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
HAS_MISTRAL = importlib.util.find_spec("langchain_mistralai") is not None
SAMPLE_PDF_PATH = Path(r"G:\reelaigen\sample-pages.pdf")


class PDFContentAgentTests(unittest.TestCase):
    def test_runs_langgraph_pipeline(self) -> None:
        if not HAS_LANGGRAPH:
            self.skipTest("langgraph is not installed")
        if not HAS_MISTRAL:
            self.skipTest("langchain_mistralai is not installed")
        if not os.getenv("MISTRAL_API_KEY"):
            self.skipTest("MISTRAL_API_KEY is not set")
        if not SAMPLE_PDF_PATH.exists():
            self.skipTest(f"Sample PDF not found: {SAMPLE_PDF_PATH}")

        with TemporaryDirectory() as tmp_dir:
            agent = ReelAIGraphAgent(
                pdf_parser=PDFParser(PDFParserConfig(save_page_images=True, image_dir=Path(tmp_dir) / "pages")),
                content_parser=ContentParser(),
                script_writer=ScriptWriter(),
                visual_planner=VisualPlanner(),
            )
            app = agent.build()
            graph = app.get_graph()

            print("\nLangGraph Mermaid:")
            print(graph.draw_mermaid())

            png_path = Path("langgraph_agent_graph.png")
            try:
                png_bytes = graph.draw_mermaid_png()
                png_path.write_bytes(png_bytes)
                print(f"\nSaved graph PNG: {png_path.resolve()}")
            except Exception:
                print("\nGraph PNG export is unavailable in this environment.")

            try:
                print("\nLangGraph ASCII:")
                graph.print_ascii()
            except Exception:
                print("\nLangGraph ASCII view is unavailable in this environment.")

            result = agent.run(
                SAMPLE_PDF_PATH,
                user_prompt={
                    "raw_prompt": "Use a clean 3Blue1Brown-style animation.",
                    "animation_style": "3blue1brown",
                    "script_style": "friendly educator",
                    "special_images": ["highlight attention diagram"],
                },
                thread_id="test-thread",
            )

        print("\nFinal agent output:")
        print(json.dumps(result["final_output"], indent=2, ensure_ascii=False))

        self.assertIn("sections", result["final_output"])
        self.assertGreaterEqual(len(result["final_output"]["sections"]), 1)
        self.assertIn("sectionId", result["final_output"]["sections"][0])
        self.assertIn("script", result["final_output"]["sections"][0])
        self.assertIn("visual", result["final_output"]["sections"][0])
        self.assertTrue(result["final_output"]["sections"][0]["script"]["narration"].strip())
        self.assertGreaterEqual(len(result["final_output"]["sections"][0]["visual"].get("scenes", [])), 1)


if __name__ == "__main__":
    unittest.main()
