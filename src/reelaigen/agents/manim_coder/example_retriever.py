from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings


load_dotenv(Path(__file__).resolve().parents[4] / ".env")

BASE_DIR = Path(__file__).parent
DEFAULT_INDEX_DIR = BASE_DIR / "knowledge_base" / "indexes" / "manim_examples_faiss"
DEFAULT_INDEX_INFO_PATH = DEFAULT_INDEX_DIR / "index_info.json"


def load_vector_store(index_dir: str | Path = DEFAULT_INDEX_DIR):
    from langchain_community.vectorstores import FAISS

    index_dir = Path(index_dir)
    index_info_path = index_dir / "index_info.json"

    with index_info_path.open("r", encoding="utf-8") as file:
        index_info = json.load(file)

    embeddings = MistralAIEmbeddings(model=index_info["embedding_model"])
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_examples(query: str, k: int = 4, index_dir: str | Path = DEFAULT_INDEX_DIR) -> list[dict]:
    vector_store = load_vector_store(index_dir)
    docs = vector_store.similarity_search(query, k=k)

    results = []
    for doc in docs:
        results.append(
            {
                "id": doc.metadata.get("id", ""),
                "title": doc.metadata.get("title", ""),
                "summary": doc.metadata.get("summary", ""),
                "explanation": doc.metadata.get("explanation", ""),
                "symbols": doc.metadata.get("symbols", []),
                "tags": doc.metadata.get("tags", []),
                "scene_family": doc.metadata.get("scene_family", ""),
                "code": doc.metadata.get("code", ""),
            }
        )

    return results


def format_examples_for_prompt(examples: list[dict]) -> str:
    blocks = []

    for index, example in enumerate(examples, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Example {index}",
                    f"id: {example.get('id', '')}",
                    f"title: {example.get('title', '')}",
                    f"summary: {example.get('summary', '')}",
                    f"explanation: {example.get('explanation', '')}",
                    f"symbols: {', '.join(example.get('symbols', []))}",
                    f"tags: {', '.join(example.get('tags', []))}",
                    "code:",
                    example.get("code", ""),
                ]
            )
        )

    return "\n\n".join(blocks)
