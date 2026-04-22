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
SECTION_TEST_PATH = Path(r"G:\reelaigen\section-test.json")


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


def load_pipeline_sections(limit: int = 4) -> list[dict]:
    with SECTION_TEST_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("sections", [])[:limit]


def build_pipeline_query(sections: list[dict]) -> str:
    parts = []

    for section in sections:
        script = section.get("script", {})
        visual = section.get("visual", {})
        concepts = ", ".join(visual.get("concepts", []))
        parts.append(
            f"Section {section.get('sectionId')}: "
            f"target={script.get('target', '')}; "
            f"concepts={concepts}"
        )

    return "Create a clean educational Manim animation for these transformer attention sections: " + " | ".join(parts)


def build_pipeline_prompt(sections: list[dict]) -> str:
    blocks = []

    for section in sections:
        script = section.get("script", {})
        visual = section.get("visual", {})
        scene_summaries = []

        for scene in visual.get("scenes", []):
            scene_summaries.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "storyboard": scene.get("storyboard"),
                    "objects": scene.get("objects", []),
                    "equations": scene.get("equations", []),
                    "transitions": scene.get("transitions", []),
                    "camera_moves": scene.get("camera_moves", []),
                    "manim_primitives": scene.get("manim_primitives", []),
                }
            )

        block = {
            "sectionId": section.get("sectionId"),
            "target": script.get("target", ""),
            "narration": script.get("narration", ""),
            "approx_duration_seconds": script.get("approx_duration_seconds"),
            "concepts": visual.get("concepts", []),
            "scenes": scene_summaries,
        }
        blocks.append(block)

    return json.dumps(blocks, indent=2, ensure_ascii=False)


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


class PipelineTests(unittest.TestCase):
    def test_generates_manim_code_from_section_pipeline_prompt(self) -> None:
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
        if not SECTION_TEST_PATH.exists():
            self.skipTest(f"Section test file not found: {SECTION_TEST_PATH}")

        sections = load_pipeline_sections(limit=4)
        query = build_pipeline_query(sections)
        pipeline_prompt = build_pipeline_prompt(sections)

        vector_store = load_vector_store()
        docs = vector_store.similarity_search(query, k=4)
        context = build_context(docs)

        llm = get_mistral_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are a Manim code generator.\n"
                    "You receive pipeline sections from an educational video system.\n"
                    "Use the retrieved examples as API grounding.\n"
                    "Write one runnable Manim Python file.\n"
                    "Return code only."
                )
            ),
            HumanMessage(
                content=(
                    f"Pipeline sections:\n{pipeline_prompt}\n\n"
                    f"Retrieval query:\n{query}\n\n"
                    f"Retrieved examples:\n{context}\n\n"
                    "Generate one Manim scene or a small sequence of scenes that explains these first sections.\n"
                    "Keep it educational, readable, and grounded in the provided pipeline data."
                )
            ),
        ]
        response = llm.invoke(messages)
        generated_code = strip_code_fences(str(response.content))

        print("\nPipeline sections prompt:\n")
        print(pipeline_prompt)
        print("\nRetrieved examples context:\n")
        print(context)
        print("\nGenerated Manim code from pipeline sections:\n")
        print(generated_code)

        self.assertTrue(generated_code.strip())
        self.assertIn("from manim import *", generated_code)
        self.assertIn("class ", generated_code)
        self.assertIn("def construct", generated_code)


if __name__ == "__main__":
    unittest.main()
