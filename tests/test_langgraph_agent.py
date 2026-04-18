from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelaigen.agents.langgraph_agent import ReelAIGraphAgent
from reelaigen.nodes.algorithm_parser import AlgorithmAnalysis, AlgorithmParser, AlgorithmStep
from reelaigen.nodes.content_parser import ContentAnalysis, ContentParser, ContentSection, SectionBoundary
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig
from reelaigen.nodes.script_writer import ScriptSectionOutput, ScriptTimingBeat, ScriptWriter

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
SAMPLE_PDF_PATH = Path(r"G:\reelaigen\sample-pages.pdf")


class FakeStructuredContentLLM:
    def invoke(self, _messages):
        return ContentAnalysis(
            parent_content_type="cs_explainer",
            sections=[
                ContentSection(
                    section_id=0,
                    section_boundary=SectionBoundary(
                        start_text="Attention Is All You Need",
                        end_text="Abstract",
                    ),
                    target="Transformer paper overview",
                    images=[],
                )
            ],
        )


class FakeContentLLM:
    def with_structured_output(self, _schema, method=None):
        return FakeStructuredContentLLM()


class FakeStructuredScriptLLM:
    def invoke(self, _messages):
        return ScriptSectionOutput(
            section_id=0,
            target="Transformer paper overview",
            section_text="Attention Is All You Need ... Abstract",
            narration="This section introduces the Transformer paper and its core motivation.",
            approx_duration_seconds=30,
            min_duration_seconds=25,
            max_duration_seconds=35,
            timing_estimate=[
                ScriptTimingBeat(start_second=0, end_second=12, note="Introduce the paper"),
                ScriptTimingBeat(start_second=12, end_second=30, note="Explain the main motivation"),
            ],
        )


class FakeScriptLLM:
    def with_structured_output(self, _schema, method=None):
        return FakeStructuredScriptLLM()


class FakeAlgorithmParser(AlgorithmParser):
    def run(self, document_text: str) -> AlgorithmAnalysis:
        return AlgorithmAnalysis(
            algorithm_detected=True,
            algorithm_name="binary_search",
            pseudocode="binary search pseudocode",
            sample_input={"array": [1, 3, 5, 7], "target": 7},
            state_trace=[
                AlgorithmStep(
                    step_id=0,
                    description="Check middle element",
                    state={"left": 0, "right": 3, "mid": 1},
                )
            ],
            verification_enabled=True,
        )


class LangGraphAgentTests(unittest.TestCase):
    def test_runs_graph_pipeline(self) -> None:
        if not HAS_LANGGRAPH:
            self.skipTest("langgraph is not installed")
        if not SAMPLE_PDF_PATH.exists():
            self.skipTest(f"Sample PDF not found: {SAMPLE_PDF_PATH}")

        with TemporaryDirectory() as tmp_dir:
            agent = ReelAIGraphAgent(
                pdf_parser=PDFParser(PDFParserConfig(save_page_images=True, image_dir=Path(tmp_dir) / "pages")),
                algorithm_parser=FakeAlgorithmParser(),
                content_parser=ContentParser(llm=FakeContentLLM()),
                script_writer=ScriptWriter(llm=FakeScriptLLM()),
            )

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

        self.assertIn("parsed_pdf", result["final_output"])
        self.assertIn("content_analysis", result["final_output"])
        self.assertIn("script_plan", result["final_output"])
        self.assertIn("user_prompt", result["final_output"])
        self.assertIn("memory", result["final_output"])
        self.assertIn("context", result["final_output"])


if __name__ == "__main__":
    unittest.main()
