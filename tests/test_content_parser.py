from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelaigen.nodes.content_parser import ContentParser
from reelaigen.nodes.pdf_parser import PDFParser, PDFParserConfig

SAMPLE_PDF_PATH = Path(r"G:\reelaigen\sample.pdf")


class ContentParserTests(unittest.TestCase):
    def test_accepts_pdf_parser_text_and_images_with_real_mistral(self) -> None:
        if not os.getenv("MISTRAL_API_KEY"):
            self.skipTest("MISTRAL_API_KEY is not set")
        if not SAMPLE_PDF_PATH.exists():
            self.skipTest(f"Sample PDF not found: {SAMPLE_PDF_PATH}")

        with TemporaryDirectory() as tmp_dir:
            pdf_parser = PDFParser(
                PDFParserConfig(
                    save_page_images=True,
                    image_dir=Path(tmp_dir) / "pages",
                )
            )
            pdf_result = pdf_parser.run(SAMPLE_PDF_PATH)
            text_chunk = pdf_result.text[: max(1, len(pdf_result.text) // 6)]
            image_chunk = [page for page in pdf_result.pages if page.image_path is not None][:1]

            content_parser = ContentParser()
            result = content_parser.run(text_chunk, images=image_chunk)
            print("\nFinal output:")
            print(result.model_dump_json(indent=2))

        self.assertTrue(result.parent_content_type.strip())
        self.assertGreaterEqual(len(result.sections), 1)
        self.assertGreaterEqual(len(image_chunk), 1)
        self.assertGreaterEqual(result.sections[0].section_id, 0)
        self.assertTrue(result.sections[0].section_boundary.start_text.strip())
        self.assertTrue(result.sections[0].section_boundary.end_text.strip())
        self.assertTrue(result.sections[0].target.strip())

    def test_rejects_empty_text(self) -> None:
        parser = ContentParser(llm=object())

        with self.assertRaises(ValueError):
            parser.run("   ")


if __name__ == "__main__":
    unittest.main()
