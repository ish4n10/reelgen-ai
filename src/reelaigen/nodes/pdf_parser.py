from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reelaigen.schemas import PDFPage, PDFParseError, PDFParseResult
from reelaigen.tools.pdf import read_pdf_metadata, read_pdf_text, save_pdf_pages_as_images


@dataclass
class PDFParserConfig:
    save_page_images: bool = False
    image_dir: Path | None = None
    image_scale: float = 2.0


class PDFParser:
    def __init__(self, config: PDFParserConfig | None = None) -> None:
        self.config = config or PDFParserConfig()

    def run(self, pdf_path: str | Path) -> PDFParseResult:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise PDFParseError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise PDFParseError(f"Expected a PDF file: {pdf_path}")

        page_texts = read_pdf_text(pdf_path)
        metadata = read_pdf_metadata(pdf_path)

        image_paths: list[Path | None] = [None] * len(page_texts)
        if self.config.save_page_images:
            output_dir = self.config.image_dir or Path("artifacts/page_images")
            saved_images = save_pdf_pages_as_images(
                pdf_path,
                output_dir=output_dir,
                image_scale=self.config.image_scale,
            )
            for i, image_path in enumerate(saved_images):
                if i < len(image_paths):
                    image_paths[i] = image_path

        pages = [
            PDFPage(number=i + 1, text=text, image_path=image_paths[i])
            for i, text in enumerate(page_texts)
        ]

        return PDFParseResult(
            text="\n\n".join(page.text for page in pages),
            pages=pages,
            metadata=metadata,
        )
