from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .contracts import ManimCoderInput
from .scene_diagnostics import inspect_manim_code
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
