from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
import pypdfium2 as pdfium


def read_pdf_text(pdf_path: str | Path, max_pages: int | None = None) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    return [(page.extract_text() or "").strip() for page in pages]


def read_pdf_metadata(pdf_path: str | Path) -> dict:
    reader = PdfReader(str(pdf_path))
    metadata = dict(reader.metadata or {})
    metadata["page_count"] = len(reader.pages)
    return metadata


def save_embedded_images(
    pdf_path: str | Path,
    output_dir: str | Path = "temp_metadata",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    saved_paths: list[Path] = []

    for page_num, page in enumerate(reader.pages, start=1):
        for image_num, image_file in enumerate(page.images, start=1):
            image_name = Path(image_file.name).name
            image_path = output_dir / f"page_{page_num:04d}_{image_num:04d}_{image_name}"
            image_path.write_bytes(image_file.data)
            saved_paths.append(image_path)

    return saved_paths


def save_pdf_pages_as_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    image_scale: float = 2.0,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(str(pdf_path))
    saved_paths: list[Path] = []

    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=image_scale)
            image_path = output_dir / f"page_{index + 1:04d}.png"
            try:
                bitmap.to_pil().save(image_path)
                saved_paths.append(image_path)
            finally:
                bitmap.close()
                page.close()
    finally:
        document.close()

    return saved_paths
