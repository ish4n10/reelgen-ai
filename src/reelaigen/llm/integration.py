from __future__ import annotations
import base64
import mimetypes
import os
from pathlib import Path

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


def build_multimodal_content(document_text: str, images: list[str | Path] | None = None) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": document_text}]

    for image in images or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_url(image)},
            }
        )

    return content


def build_multimodal_message(document_text: str, images: list[str | Path] | None = None) -> HumanMessage:
    return HumanMessage(content=build_multimodal_content(document_text, images))


def get_llm():
    from langchain_openai import ChatOpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before using the content parser.")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )


def get_mistral_llm():
    from langchain_mistralai import ChatMistralAI

    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError("Set MISTRAL_API_KEY in your .env file or environment.")

    return ChatMistralAI(
        model="mistral-medium-latest",
        temperature=0,
    )
