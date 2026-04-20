from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


load_dotenv(Path(__file__).resolve().parents[4] / ".env")

BASE_DIR = Path(__file__).parent
DEFAULT_EXAMPLES_PATH = BASE_DIR / "examples" / "manim_examples.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "examples" / "manim_examples_faiss"
DEFAULT_EMBEDDING_MODEL = "mistral-embed"


def load_examples(examples_path: str | Path = DEFAULT_EXAMPLES_PATH) -> dict:
    examples_path = Path(examples_path)
    with examples_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_embedding_text(example: dict) -> str:
    parts = [
        f"title: {example.get('title', '')}",
        f"summary: {example.get('summary', '')}",
        f"explanation: {example.get('explanation', '')}",
        f"keywords: {', '.join(example.get('keywords', []))}",
        f"example_text: {example.get('example_text', '')}",
        f"symbols: {', '.join(example.get('symbols', []))}",
        f"tags: {', '.join(example.get('tags', []))}",
        f"scene_family: {example.get('scene_family', '')}",
        f"source: {example.get('source', '')}",
    ]
    return "\n".join(parts)


def build_documents(
    examples_path: str | Path = DEFAULT_EXAMPLES_PATH,
) -> list[Document]:
    dataset = load_examples(examples_path)
    examples = dataset.get("examples", [])
    documents: list[Document] = []

    for example in examples:
        embedding_text = build_embedding_text(example)
        documents.append(
            Document(
                page_content=embedding_text,
                metadata={
                    "id": example.get("id"),
                    "title": example.get("title"),
                    "summary": example.get("summary"),
                    "explanation": example.get("explanation"),
                    "keywords": example.get("keywords", []),
                    "symbols": example.get("symbols", []),
                    "tags": example.get("tags", []),
                    "scene_family": example.get("scene_family"),
                    "source": example.get("source"),
                    "code": example.get("code", ""),
                },
            )
        )

    return documents


def build_example_vector_db(
    examples_path: str | Path = DEFAULT_EXAMPLES_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Path:
    examples_path = Path(examples_path)
    output_path = Path(output_path)
    documents = build_documents(examples_path)
    embeddings = MistralAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(documents, embeddings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_path))

    metadata_path = output_path / "index_info.json"
    payload = {
        "embedding_model": embedding_model,
        "source_file": str(examples_path),
        "document_count": len(documents),
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return output_path


def main() -> None:
    output_path = build_example_vector_db()
    print(f"Saved example vector DB to: {output_path}")


if __name__ == "__main__":
    main()
