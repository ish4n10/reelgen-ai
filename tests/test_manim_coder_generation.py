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


def build_scene_planning_context(sections: list[dict]) -> str:
    lines = [
        "Plan a production-quality educational Manim scene for the following sections.",
        "Use the section order as the teaching order.",
        "Treat the listed visual scenes as source material, not as a literal code template.",
        "Prefer clean left-to-right educational layouts, consistent object sizes, and short labels.",
        "",
        build_pipeline_prompt(sections),
    ]
    return "\n".join(lines)


def save_generated_manim_code(code: str) -> None:
    GENERATED_MANIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "sys.path.insert" not in code:
        bootstrap = (
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[2] / \"src\"))\n\n"
        )
        code = bootstrap + code
    GENERATED_MANIM_PATH.write_text(code, encoding="utf-8")


def assert_scene_ir_generation_result(test_case: unittest.TestCase, result: dict, generated_code: str, scene_name: str) -> None:
    test_case.assertIn("scene_ir", result)
    test_case.assertIn("scene_ir_validation", result)
    test_case.assertIn("planning_attempts", result)
    test_case.assertIn("repair_attempts", result)
    test_case.assertIn("planning_failed", result)

    scene_ir = result["scene_ir"]
    validation = result["scene_ir_validation"]

    test_case.assertEqual(scene_ir["scene_name"], scene_name)
    test_case.assertIn("objects", scene_ir)
    test_case.assertIn("animation_blocks", scene_ir)
    test_case.assertTrue(scene_ir["objects"])
    test_case.assertTrue(scene_ir["animation_blocks"])

    test_case.assertFalse(result["planning_failed"], msg=f"Scene IR planning failed: {validation}")
    test_case.assertTrue(validation["passed"], msg=f"Scene IR validation failed: {validation}")
    test_case.assertIn("from manim import *", generated_code)
    test_case.assertIn(RUNTIME_IMPORT, generated_code)
    test_case.assertIn("InstrumentedScene", generated_code)
    test_case.assertIn("set_runtime_block", generated_code)
    test_case.assertIn("get_runtime_report", generated_code)
    test_case.assertIn("class ", generated_code)
    test_case.assertIn("def construct", generated_code)


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
        planning_context = (
            "Build a single educational scene about spacetime curvature near a black hole.\n"
            "The scene should show axes, a curved gravity-well style surface, and gentle camera motion.\n"
            "Keep the layout readable and production-oriented.\n"
            "Use short labels and explicit scene blocks.\n"
        )

        llm = get_mistral_llm()
        agent = ManimCoderAgent(llm=llm)
        result = agent.generate_from_planning_context_with_repair(
            scene_name="GeneratedBlackHoleScene",
            scene_goal="Explain a black-hole-style spacetime curvature scene.",
            planning_context=planning_context,
            retrieval_context=context,
            max_replans=2,
            max_repairs=2,
            source_context={"query": query},
        )
        generated_code = result["final_code"]

        print("\nScene IR:\n")
        print(json.dumps(result["scene_ir"], indent=2))
        print("\nRetrieved examples context:\n")
        print(context)
        print("\nGenerated Manim code:\n")
        print(generated_code)
        print("\nDiagnostics summary:\n")
        print(json.dumps(result["final_diagnostics"], indent=2))
        print("\nScene IR validation:\n")
        print(json.dumps(result["scene_ir_validation"], indent=2))

        if result.get("planning_failed"):
            self.fail(f"Scene IR planning failed: {json.dumps(result['scene_ir_validation'], indent=2)}")

        self.assertTrue(generated_code.strip())
        assert_scene_ir_generation_result(self, result, generated_code, "GeneratedBlackHoleScene")


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
        planning_context = build_scene_planning_context(sections)

        vector_store = load_vector_store()
        docs = vector_store.similarity_search(query, k=4)
        context = build_context(docs)

        llm = get_mistral_llm()
        agent = ManimCoderAgent(llm=llm)
        result = agent.generate_from_planning_context_with_repair(
            scene_name="GeneratedSectionScene",
            scene_goal="Explain the first transformer attention sections in order.",
            planning_context=planning_context,
            retrieval_context=context,
            max_replans=2,
            max_repairs=2,
            source_context={"sections": sections, "query": query},
        )
        generated_code = result["final_code"]

        print("\nScene IR:\n")
        print(json.dumps(result["scene_ir"], indent=2))
        print("\nPipeline sections prompt:\n")
        print(pipeline_prompt)
        print("\nRetrieved examples context:\n")
        print(context)
        print("\nGenerated Manim code from pipeline sections:\n")
        print(generated_code)
        print("\nDiagnostics summary:\n")
        print(json.dumps(result["final_diagnostics"], indent=2))
        print("\nScene IR validation:\n")
        print(json.dumps(result["scene_ir_validation"], indent=2))

        if result.get("planning_failed"):
            self.fail(f"Scene IR planning failed: {json.dumps(result['scene_ir_validation'], indent=2)}")

        save_generated_manim_code(generated_code)
        print(f"\nSaved generated Manim code to: {GENERATED_MANIM_PATH}\n")

        self.assertTrue(generated_code.strip())
        self.assertIn("GeneratedSectionScene", generated_code)
        assert_scene_ir_generation_result(self, result, generated_code, "GeneratedSectionScene")


if __name__ == "__main__":
    unittest.main()
