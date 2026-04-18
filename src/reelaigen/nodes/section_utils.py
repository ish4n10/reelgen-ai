from __future__ import annotations

from typing import Any

from reelaigen.nodes.content_parser import ContentSection


def extract_section_text(
    document_text: str,
    start_text: str,
    end_text: str,
    fallback_limit: int,
) -> str:
    start_index = document_text.find(start_text)
    if start_index == -1:
        start_index = 0

    end_index = document_text.find(end_text, start_index + len(start_text))
    if end_index == -1:
        end_index = len(document_text)
    else:
        end_index += len(end_text)

    extracted = document_text[start_index:end_index].strip()
    if extracted:
        return extracted

    return document_text[:fallback_limit].strip()


def collect_section_images(
    section: ContentSection,
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted_ids = {image.image_id for image in section.images}
    if not wanted_ids:
        return []

    matches = []
    for page in pages:
        if page.get("image_id") in wanted_ids and page.get("image_path"):
            matches.append(page)

    return matches
