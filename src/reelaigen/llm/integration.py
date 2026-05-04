from __future__ import annotations
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _image_to_url(image: str | Path) -> str:
    image = str(image)
    if image.startswith(("http://", "https://", "data:")):
        return image

    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"
    image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_images(images: list[Any] | None) -> list[tuple[str, str | Path]]:
    normalized: list[tuple[str, str | Path]] = []

    for index, image in enumerate(images or [], start=1):
        if isinstance(image, (str, Path)):
            normalized.append((f"image_{index}", image))
            continue

        image_path = getattr(image, "image_path", None)
        image_number = getattr(image, "number", index)
        if image_path is not None:
            normalized.append((f"image_{image_number}", image_path))
            continue

        if isinstance(image, dict):
            raw_path = image.get("image_path") or image.get("path") or image.get("url")
            image_id = image.get("image_id") or f"image_{index}"
            if raw_path is not None:
                normalized.append((image_id, raw_path))

    return normalized


def build_multimodal_content(document_text: str, images: list[Any] | None = None) -> list[dict]:
    normalized_images = _normalize_images(images)

    image_lines = []
    for block_index, (image_id, _image) in enumerate(normalized_images, start=1):
        image_lines.append(f"- {image_id}: refer to image block {block_index}")

    if image_lines:
        document_text = (
            f"{document_text}\n\n"
            "Image IDs for the following image blocks:\n"
            f"{chr(10).join(image_lines)}"
        )

    content: list[dict] = [{"type": "text", "text": document_text}]

    for _image_id, image in normalized_images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_url(image)},
            }
        )

    return content


def build_multimodal_message(document_text: str, images: list[Any] | None = None) -> HumanMessage:
    return HumanMessage(content=build_multimodal_content(document_text, images))


def get_llm():
    from langchain_deepseek import ChatDeepSeek
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Set DEEPSEEK_API_KEY before using the content parser.")

    llm = ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
   )
    return llm
    return get_mistral_llm()



def get_mistral_llm():
    from langchain_mistralai import ChatMistralAI

    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("Set MISTRAL_API_KEY in your .env file or environment.")

    return ChatMistralAI(
        model="mistral-medium-latest",
        temperature=0,
    )
