from __future__ import annotations

from typing import Any

from .contracts import ManimCoderInput
from .state import ManimCoderState, build_initial_manim_coder_state


class ManimCoderAgent:
    """
    Small entry point for the future Manim coding pipeline.

    For now this only prepares a clean state object.
    Retrieval, code generation, validation, rendering, and repair
    can be added here step by step.
    """

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
