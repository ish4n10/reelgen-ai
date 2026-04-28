from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelaigen.agents.manim_coder.agent import ManimCoderAgent
from reelaigen.llm.integration import get_mistral_llm


HAS_FAISS = importlib.util.find_spec("faiss") is not None or importlib.util.find_spec("faiss_cpu") is not None
HAS_LANGCHAIN_COMMUNITY = importlib.util.find_spec("langchain_community") is not None
HAS_MISTRAL = importlib.util.find_spec("langchain_mistralai") is not None

FAISS_DIR = Path(r"G:\reelaigen\src\reelaigen\agents\manim_coder\knowledge_base\indexes\manim_examples_faiss")
INDEX_INFO_PATH = FAISS_DIR / "index_info.json"
SECTION_TEST_PATH = Path(r"G:\reelaigen\section-test.json")
GENERATED_MANIM_PATH = Path(r"G:\reelaigen\tests\manim_tests\generated_from_section_test.py")
RUNTIME_IMPORT = (
    "from reelaigen.agents.manim_coder.runtime import InstrumentedScene"
)


RUNTIME_PROMPT = (
    "Use ReelaiGen runtime middleware.\n"
    f"Include this import exactly: {RUNTIME_IMPORT}\n"
    "InstrumentedScene is observer-first. It records bbox diagnostics, collisions, off-frame objects, pacing, and scene diffs.\n"
    "Do not rely on the runtime to redesign the scene for you.\n"
    "You may use raw Manim primitives for creative, colorful, expressive visuals.\n"
    "Do not use ReelaiGen wrapper mobjects or wrapper layout helpers.\n"
    "Use plain Manim primitives such as Text, Rectangle, VGroup, Line, Arrow, Axes, NumberPlane, and Surface.\n"
    "Build formulas and matrix-like layouts with plain Text, Line, Rectangle, and VGroup.\n"
    "Avoid Manim Matrix unless LaTeX is explicitly available, because Matrix uses MathTex internally.\n"
    "The generated scene class must inherit from InstrumentedScene.\n"
    "Do not inherit directly from Scene, ThreeDScene, MovingCameraScene, or ZoomedScene.\n"
    "InstrumentedScene observes after add() and play().\n"
    "It detects overlap, off-screen objects, clutter, arrow drift, camera issues, and slow pacing.\n"
    "It records layout_repairs, connection_repairs, camera_repairs, timing_repairs, and gc_plans.\n"
    "Prefer Text for normal labels and only use MathTex when LaTeX is required.\n"
    "Keep text blocks short, centered, and scaled to fit the frame when needed.\n"
    "At the end of construct, call report = self.get_runtime_report().\n"
    "Print a compact runtime summary, not the full report.\n"
    "The summary should include snapshot_count, diff_count, bbox_collision_steps, bbox_out_of_frame_steps, layout_issue_steps, layout_repair_steps, "
    "connection_repair_steps, camera_repair_steps, timing_repair_steps, and gc_plan_count.\n"
)


def load_vector_store():
    from langchain_community.vectorstores import FAISS
    from langchain_mistralai import MistralAIEmbeddings

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


def save_generated_manim_code(code: str) -> None:
    GENERATED_MANIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    GENERATED_MANIM_PATH.write_text(code, encoding="utf-8")


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
        agent = ManimCoderAgent(llm=llm)
        system_prompt = (
            "You are a Manim code generator.\n"
            "Use the retrieved examples as API grounding.\n"
            "Write one runnable Manim file.\n"
            "The scene should feel like a spacetime curvature or black hole visualization.\n"
            "Name the scene class GeneratedBlackHoleScene.\n"
            f"{RUNTIME_PROMPT}"
            "Return code only."
        )
        user_prompt = (
            f"User request:\n{query}\n\n"
            f"Retrieved examples:\n{context}\n\n"
            "Write a complete Manim Python example with imports and one scene class.\n"
            "Prefer a 3D surface that looks like a gravity well or curved spacetime.\n"
            "Include axes and gentle camera movement.\n"
            "Because InstrumentedScene is based on Scene, avoid ThreeDScene-only methods unless you also explain them through normal mobjects."
        )
        result = agent.generate_with_repair(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            scene_name="GeneratedBlackHoleScene",
            max_retries=2,
        )
        generated_code = result["final_code"]

        print("\nRetrieved examples context:\n")
        print(context)
        print("\nGenerated Manim code:\n")
        print(generated_code)
        print("\nDiagnostics summary:\n")
        print(json.dumps(result["final_diagnostics"], indent=2))

        self.assertTrue(generated_code.strip())
        self.assertIn("from manim import *", generated_code)
        self.assertIn(RUNTIME_IMPORT, generated_code)
        self.assertIn("InstrumentedScene", generated_code)
        self.assertIn("get_runtime_report", generated_code)
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
        agent = ManimCoderAgent(llm=llm)
        system_prompt = (
            "You are a Manim code generator.\n"
            "You receive pipeline sections from an educational video system.\n"
            "Use the retrieved examples as API grounding.\n"
            "The section visual.scenes data is the source of truth for what to animate.\n"
            "Follow each scene storyboard, objects, equations, transitions, camera_moves, and manim_primitives.\n"
            "Write one runnable Manim Python file.\n"
            "Name the scene class GeneratedSectionScene.\n"
            f"{RUNTIME_PROMPT}"
            "Return code only."
        )
        user_prompt = (
            f"Pipeline sections:\n{pipeline_prompt}\n\n"
            f"Retrieval query:\n{query}\n\n"
            f"Retrieved examples:\n{context}\n\n"
            "Generate one Manim scene that explains these first sections in order.\n"
            "Use section targets as chapter titles.\n"
            "Use narration to decide what labels and animation beats should appear.\n"
            "Use visual concepts to choose objects.\n"
            "Use visual equations to decide formulas using plain Manim text-based layouts.\n"
            "Use visual transitions and camera_moves as animation instructions.\n"
            "Keep it educational, readable, and grounded in the provided pipeline data.\n"
            "Use InstrumentedScene so the runtime middleware records layout snapshots and linter reports."
        )
        result = agent.generate_with_repair(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            scene_name="GeneratedSectionScene",
            max_retries=2,
        )
        generated_code = result["final_code"]
        save_generated_manim_code(generated_code)

        print("\nPipeline sections prompt:\n")
        print(pipeline_prompt)
        print("\nRetrieved examples context:\n")
        print(context)
        print("\nGenerated Manim code from pipeline sections:\n")
        print(generated_code)
        print("\nDiagnostics summary:\n")
        print(json.dumps(result["final_diagnostics"], indent=2))
        print(f"\nSaved generated Manim code to: {GENERATED_MANIM_PATH}\n")

        self.assertTrue(generated_code.strip())
        self.assertIn("from manim import *", generated_code)
        self.assertIn(RUNTIME_IMPORT, generated_code)
        self.assertIn("InstrumentedScene", generated_code)
        self.assertIn("GeneratedSectionScene", generated_code)
        self.assertIn("get_runtime_report", generated_code)
        self.assertIn("class ", generated_code)
        self.assertIn("def construct", generated_code)


if __name__ == "__main__":
    unittest.main()
