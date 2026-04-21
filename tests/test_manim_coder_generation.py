from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import MistralAIEmbeddings

from reelaigen.llm.integration import get_mistral_llm


HAS_FAISS = importlib.util.find_spec("faiss") is not None or importlib.util.find_spec("faiss_cpu") is not None
HAS_LANGCHAIN_COMMUNITY = importlib.util.find_spec("langchain_community") is not None
HAS_MISTRAL = importlib.util.find_spec("langchain_mistralai") is not None

FAISS_DIR = Path(r"G:\reelaigen\src\reelaigen\agents\manim_coder\knowledge_base\indexes\manim_examples_faiss")
INDEX_INFO_PATH = FAISS_DIR / "index_info.json"


def load_vector_store() -> FAISS:
    with INDEX_INFO_PATH.open("r", encoding="utf-8") as file:
        index_info = json.load(file)

    embeddings = MistralAIEmbeddings(model=index_info["embedding_model"])
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_context(docs: list) -> str:
    blocks = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        blocks.append(
            "\n".join(
                [
                    f"Example {index}",
                    f"id: {metadata.get('id', '')}",
                    f"title: {metadata.get('title', '')}",
                    f"summary: {metadata.get('summary', '')}",
                    f"explanation: {metadata.get('explanation', '')}",
                    f"symbols: {', '.join(metadata.get('symbols', []))}",
                    f"tags: {', '.join(metadata.get('tags', []))}",
                    "code:",
                    metadata.get("code", ""),
                ]
            )
        )

    return "\n\n".join(blocks)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


class ManimCoderGenerationTests(unittest.TestCase):
    def test_generates_manim_code_from_faiss_index(self) -> None:
        if not HAS_FAISS:
            self.skipTest("faiss is not installed")
        if not HAS_LANGCHAIN_COMMUNITY:
            self.skipTest("langchain_community is not installed")
        if not HAS_MISTRAL:
            self.skipTest("langchain_mistralai is not installed")
        if not os.getenv("MISTRAL_API_KEY"):
            self.skipTest("MISTRAL_API_KEY is not set")
        if not FAISS_DIR.exists():
            self.skipTest(f"FAISS directory not found: {FAISS_DIR}")
        if not INDEX_INFO_PATH.exists():
            self.skipTest(f"Index info file not found: {INDEX_INFO_PATH}")

        vector_store = load_vector_store()
        query = (
            "Create a clean educational 3D Manim scene showing a spacetime-style surface "
            "with a black hole well, visible axes, and slow camera rotation."
        )
        docs = vector_store.similarity_search(query, k=3)
        context = build_context(docs)

        llm = get_mistral_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are a Manim code generator.\n"
                    "Use the retrieved examples as API grounding.\n"
                    "Write one runnable Manim file.\n"
                    "The scene should feel like a spacetime curvature or black hole visualization.\n"
                    "Return code only."
                )
            ),
            HumanMessage(
                content=(
                    f"User request:\n{query}\n\n"
                    f"Retrieved examples:\n{context}\n\n"
                    "Write a complete Manim Python example with imports and one scene class.\n"
                    "Prefer a 3D surface that looks like a gravity well or curved spacetime.\n"
                    "Include axes and gentle camera movement."
                )
            ),
        ]
        response = llm.invoke(messages)
        generated_code = strip_code_fences(str(response.content))

        print("\nRetrieved examples context:\n")
        print(context)
        print("\nGenerated Manim code:\n")
        print(generated_code)

        self.assertTrue(generated_code.strip())
        self.assertIn("from manim import *", generated_code)
        self.assertIn("class ", generated_code)
        self.assertIn("def construct", generated_code)


if __name__ == "__main__":
    unittest.main()
