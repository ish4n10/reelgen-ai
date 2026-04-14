from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelaigen.agents.pdf_content_agent import PDFContentAgent
from reelaigen.nodes.content_parser import ContentAnalysis, ContentParser, ContentSection, SectionBoundary
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig

SAMPLE_PDF_PATH = Path(r"G:\reelaigen\sample.pdf")


class FakeStructuredLLM:
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


class FakeLLM:
    def with_structured_output(self, _schema, method=None):
        return FakeStructuredLLM()


class PDFContentAgentTests(unittest.TestCase):
    def test_runs_pdf_parser_then_content_parser(self) -> None:
        if not SAMPLE_PDF_PATH.exists():
            self.skipTest(f"Sample PDF not found: {SAMPLE_PDF_PATH}")

        with TemporaryDirectory() as tmp_dir:
            pdf_parser = PDFParser(
                PDFParserConfig(
                    save_page_images=True,
                    image_dir=Path(tmp_dir) / "pages",
                )
            )
            content_parser = ContentParser(llm=FakeLLM())
            agent = PDFContentAgent(pdf_parser=pdf_parser, content_parser=content_parser)

            result = agent.run(SAMPLE_PDF_PATH)

        self.assertTrue(result.parsed_pdf["text"].strip())
        self.assertGreater(result.parsed_pdf["metadata"]["page_count"], 0)
        self.assertGreaterEqual(len(result.parsed_pdf["pages"]), 1)
        self.assertEqual(result.content_analysis.parent_content_type, "cs_explainer")
        self.assertEqual(result.content_analysis.sections[0].section_id, 0)


if __name__ == "__main__":
    unittest.main()
