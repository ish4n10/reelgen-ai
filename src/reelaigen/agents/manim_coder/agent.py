from __future__ import annotations

import ast
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .contracts import ManimCoderInput, SceneDiagnosticsResult, SceneIR, SceneIRValidationResult
from .scene_diagnostics import inspect_manim_code
from .scene_ir_validator import validate_scene_ir
from .scene_planner import build_scene_ir, repair_scene_ir
from .symbol_lookup import SymbolLookup
from .state import ManimCoderState, build_initial_manim_coder_state


class ManimCoderAgent:
    """
    Small entry point for the future Manim coding pipeline.

    For now this only prepares a clean state object.
    Retrieval, code generation, validation, rendering, and repair
    can be added here step by step.
    """

    def __init__(self, llm=None, symbol_lookup: SymbolLookup | None = None) -> None:
        self.llm = llm
        self.symbol_lookup = symbol_lookup or SymbolLookup()

    def build_initial_state(
        self,
        section_id: int,
        target: str,
        narration: str,
        visual_plan: dict[str, Any],
        user_prompt: dict[str, Any] | None = None,
    ) -> ManimCoderState:
        input_payload = ManimCoderInput(
            section_id=section_id,
            target=target,
            narration=narration,
            visual_plan=visual_plan,
            user_prompt=user_prompt or {},
        )
        return build_initial_manim_coder_state(input_payload.model_dump())

    def run(
        self,
        section_id: int,
        target: str,
        narration: str,
        visual_plan: dict[str, Any],
        user_prompt: dict[str, Any] | None = None,
    ) -> ManimCoderState:
        return self.build_initial_state(
            section_id=section_id,
            target=target,
            narration=narration,
            visual_plan=visual_plan,
            user_prompt=user_prompt,
        )

    def generate_with_repair(
        self,
        system_prompt: str,
        user_prompt: str,
        scene_name: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("ManimCoderAgent needs an llm for generate_with_repair().")

        attempts: list[dict[str, Any]] = []
        code = self.generate_code(system_prompt, user_prompt)
        diagnostics = inspect_manim_code(code, scene_name)
        attempts.append(self.build_attempt_record(code, diagnostics, 1, "generate"))

        attempt_number = 1
        while not diagnostics.passed and attempt_number <= max_retries:
            attempt_number += 1
            code = self.repair_code(
                original_user_prompt=user_prompt,
                current_code=code,
                repair_prompt=diagnostics.repair_prompt,
                scene_name=scene_name,
            )
            diagnostics = inspect_manim_code(code, scene_name)
            attempts.append(self.build_attempt_record(code, diagnostics, attempt_number, "repair"))

        return {
            "passed": diagnostics.passed,
            "scene_name": scene_name,
            "final_code": code,
            "final_diagnostics": diagnostics.model_dump(),
            "attempts": attempts,
        }

    def generate_from_planning_context_with_repair(
        self,
        *,
        scene_name: str,
        scene_goal: str,
        planning_context: str,
        retrieval_context: str,
        max_replans: int = 5,
        max_repairs: int = 2,
        source_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("ManimCoderAgent needs an llm for generate_from_planning_context_with_repair().")

        repair_attempts: list[dict[str, Any]] = []

        scene_ir, scene_ir_validation, planning_attempts = self.plan_scene_ir_with_validation(
            scene_name=scene_name,
            scene_goal=scene_goal,
            planning_context=planning_context,
            source_context=source_context or {},
            max_replans=max_replans,
        )

        if not scene_ir_validation.passed:
            return {
                "passed": False,
                "planning_failed": True,
                "scene_name": scene_name,
                "scene_ir": scene_ir.model_dump(),
                "scene_ir_validation": scene_ir_validation.model_dump(),
                "final_code": "",
                "final_diagnostics": {
                    "passed": False,
                    "static_validation": {"passed": False, "errors": [], "warnings": []},
                    "render_success": False,
                    "render_error": "Scene IR planning failed before code generation.",
                    "issues": [],
                    "repair_prompt": "",
                    "runtime_report": {},
                },
                "planning_attempts": planning_attempts,
                "repair_attempts": repair_attempts,
            }

        system_prompt = self.build_codegen_system_prompt(scene_name=scene_name)
        user_prompt = self.build_codegen_user_prompt(scene_ir=scene_ir, retrieval_context=retrieval_context)
        code = self.generate_code(system_prompt, user_prompt)
        diagnostics = inspect_manim_code(code, scene_name)
        repair_attempts.append(self.build_attempt_record(code, diagnostics, 1, "generate"))

        attempt_number = 1
        while not diagnostics.passed and attempt_number <= max_repairs:
            attempt_number += 1
            code = self.repair_code_from_scene_ir(
                scene_ir=scene_ir,
                current_code=code,
                diagnostics=diagnostics,
                scene_name=scene_name,
            )
            diagnostics = inspect_manim_code(code, scene_name)
            repair_attempts.append(self.build_attempt_record(code, diagnostics, attempt_number, "repair"))

        return {
            "passed": scene_ir_validation.passed and diagnostics.passed,
            "planning_failed": False,
            "scene_name": scene_name,
            "scene_ir": scene_ir.model_dump(),
            "scene_ir_validation": scene_ir_validation.model_dump(),
            "final_code": code,
            "final_diagnostics": diagnostics.model_dump(),
            "planning_attempts": planning_attempts,
            "repair_attempts": repair_attempts,
        }

    def plan_scene_ir_with_validation(
        self,
        *,
        scene_name: str,
        scene_goal: str,
        planning_context: str,
        source_context: dict[str, Any],
        max_replans: int = 5,
    ) -> tuple[SceneIR, SceneIRValidationResult, list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        scene_ir = build_scene_ir(
            self.llm,
            scene_name=scene_name,
            scene_goal=scene_goal,
            planning_context=planning_context,
            source_context=source_context,
        )
        validation = validate_scene_ir(scene_ir)
        attempts.append(
            {
                "attempt": 1,
                "phase": "plan_scene_ir",
                "passed": validation.passed,
                "errors": validation.errors,
                "warnings": validation.warnings,
                "scene_ir": scene_ir.model_dump(),
            }
        )

        replan_count = 0
        while not validation.passed and replan_count < max_replans:
            replan_count += 1
            scene_ir = repair_scene_ir(
                self.llm,
                current_scene_ir=scene_ir,
                validation_errors=validation.errors,
                validation_warnings=validation.warnings,
            )
            validation = validate_scene_ir(scene_ir)
            attempts.append(
                {
                    "attempt": replan_count + 1,
                    "phase": "repair_scene_ir",
                    "passed": validation.passed,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                    "scene_ir": scene_ir.model_dump(),
                }
            )

        return scene_ir, validation, attempts

    def build_codegen_system_prompt(self, *, scene_name: str) -> str:
        return (
            "You are a Manim code generator.\n"
            "Write one complete runnable Python scene file.\n"
            "Use raw Manim primitives and ReelaiGen InstrumentedScene.\n"
            "Include 'from manim import *' at the top of the file.\n"
            "The generated scene class must inherit from InstrumentedScene.\n"
            "Include this import exactly: from reelaigen.agents.manim_coder.runtime import InstrumentedScene\n"
            f"The scene class name must be {scene_name}.\n"
            "Follow the Scene IR coordinates exactly unless a direct API constraint requires a tiny adjustment.\n"
            "Write one function per animation block named block_<block_id>(self).\n"
            "At the start of each block function call self.set_runtime_block('<block_id>').\n"
            "Assign every planned object to self.<object_id> and also set self.<object_id>._reelaigen_id = '<object_id>'.\n"
            "Use move_to() for absolute placement.\n"
            "Use object boundary anchors for arrows; avoid freeform diagonal arrows unless explicitly requested by the Scene IR.\n"
            "Use short Text-based formulas and layouts by default. Avoid Matrix or MathTex unless absolutely necessary.\n"
            "At the end of construct, call report = self.get_runtime_report() and print a compact summary.\n"
            "Return code only."
        )

    def build_codegen_user_prompt(self, *, scene_ir: SceneIR, retrieval_context: str) -> str:
        return (
            "Generate Manim code from this validated Scene IR.\n\n"
            "Scene IR:\n"
            f"{json.dumps(scene_ir.model_dump(), indent=2, ensure_ascii=False)}\n\n"
            "Retrieved Manim examples:\n"
            f"{retrieval_context}\n\n"
            "Preserve the block order from animation_blocks.\n"
            "Use the planned ids and coordinates so runtime diagnostics can map issues back to blocks.\n"
        )

    def generate_code(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return strip_code_fences(str(response.content))

    def repair_code(
        self,
        original_user_prompt: str,
        current_code: str,
        repair_prompt: str,
        scene_name: str,
    ) -> str:
        repair_system_prompt = (
            "You are repairing a generated Manim scene.\n"
            "Keep the same scene goal and the same scene class name.\n"
            f"The scene class name must remain {scene_name}.\n"
            "Edit only what is necessary to fix the diagnostics.\n"
            "Return code only."
        )
        repair_user_prompt = (
            f"Original request:\n{original_user_prompt}\n\n"
            f"Current code:\n```python\n{current_code}\n```\n\n"
            f"Diagnostics:\n{repair_prompt}\n"
        )
        return self.generate_code(repair_system_prompt, repair_user_prompt)

    def repair_code_from_scene_ir(
        self,
        *,
        scene_ir: SceneIR,
        current_code: str,
        diagnostics: SceneDiagnosticsResult,
        scene_name: str,
    ) -> str:
        failing_blocks = self.find_failing_blocks(diagnostics)
        block_context = self.build_block_repair_context(scene_ir=scene_ir, current_code=current_code, block_ids=failing_blocks)
        repair_system_prompt = (
            "You are repairing a generated Manim scene.\n"
            f"The scene class name must remain {scene_name}.\n"
            "Keep all unaffected animation blocks unchanged.\n"
            "Rewrite only the block functions implicated by the diagnostics when possible.\n"
            "Return the full corrected code only.\n"
        )
        repair_user_prompt = (
            "Scene IR:\n"
            f"{json.dumps(scene_ir.model_dump(), indent=2, ensure_ascii=False)}\n\n"
            f"Current code:\n```python\n{current_code}\n```\n\n"
            f"Relevant block context:\n{block_context}\n\n"
            f"Diagnostics:\n{diagnostics.repair_prompt}\n"
        )
        return self.generate_code(repair_system_prompt, repair_user_prompt)

    def build_attempt_record(self, code: str, diagnostics, attempt_number: int, phase: str) -> dict[str, Any]:
        return {
            "attempt": attempt_number,
            "phase": phase,
            "passed": diagnostics.passed,
            "render_success": diagnostics.render_success,
            "issue_count": len(diagnostics.issues),
            "repair_prompt": diagnostics.repair_prompt,
            "code": code,
        }

    def find_failing_blocks(self, diagnostics: SceneDiagnosticsResult) -> list[str]:
        ordered_blocks: list[str] = []
        seen: set[str] = set()

        for issue in diagnostics.issues:
            if not issue.block_id or issue.block_id in seen:
                continue
            seen.add(issue.block_id)
            ordered_blocks.append(issue.block_id)

        return ordered_blocks

    def build_block_repair_context(self, *, scene_ir: SceneIR, current_code: str, block_ids: list[str]) -> str:
        if not block_ids:
            return "No block ids were identified from diagnostics. Repair the smallest relevant scene section."

        lines = []
        for block_id in block_ids:
            block = next((item for item in scene_ir.animation_blocks if item.block_id == block_id), None)
            if block is None:
                continue

            function_name = block_function_name(block_id)
            function_source = extract_function_source(current_code, function_name)
            lines.append(
                "\n".join(
                    [
                        f"Block {block_id}",
                        json.dumps(block.model_dump(), indent=2, ensure_ascii=False),
                        "Function source:",
                        function_source or "<function not found>",
                    ]
                )
            )

        return "\n\n".join(lines) if lines else "No matching block functions were found in the current code."


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


def block_function_name(block_id: str) -> str:
    normalized = "".join(character if character.isalnum() or character == "_" else "_" for character in block_id)
    normalized = normalized.strip("_") or "unnamed_block"
    return f"block_{normalized}"


def extract_function_source(code: str, function_name: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    lines = code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            if getattr(node, "end_lineno", None) is None:
                return "\n".join(lines[node.lineno - 1 :])
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return ""
