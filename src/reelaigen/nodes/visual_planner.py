from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reelaigen.llm.integration import build_multimodal_message
from reelaigen.llm.integration import get_llm
from reelaigen.llm.prompts import VISUAL_PLANNER_PROMPT
from reelaigen.nodes.content_parser import ContentSection
from reelaigen.nodes.section_utils import collect_section_images, extract_section_text
from reelaigen.nodes.script_writer import ScriptSectionOutput


class VisualScene(BaseModel):
    scene_id: int = Field(..., description="Scene index within the section.")
    storyboard: str = Field(..., description="Short description of what the viewer sees.")
    objects: list[str] = Field(default_factory=list, description="Main visual objects in the scene.")
    equations: list[str] = Field(default_factory=list, description="Equations or formulas shown in the scene.")
    transitions: list[str] = Field(default_factory=list, description="How the scene enters or changes.")
    camera_moves: list[str] = Field(default_factory=list, description="Camera movement or focus notes.")
    manim_primitives: list[str] = Field(default_factory=list, description="Likely Manim primitives to use.")


class VisualSectionPlan(BaseModel):
    section_id: int = Field(..., description="Section index from the content analyzer.")
    target: str = Field(..., description="Short target for this section.")
    concepts: list[str] = Field(default_factory=list, description="Core visual concepts to show in this section.")
    scenes: list[VisualScene] = Field(default_factory=list, description="Storyboard scenes for this section.")


class VisualPlan(BaseModel):
    sections: list[VisualSectionPlan] = Field(default_factory=list)


@dataclass
class VisualPlannerConfig:
    max_chars: int = 6000


class VisualPlanner:
    def __init__(self, llm=None, config: VisualPlannerConfig | None = None) -> None:
        self.llm = llm or get_llm()
        self.config = config or VisualPlannerConfig()

    def run(
        self,
        document_text: str,
        sections: list[ContentSection],
        script_sections: list[ScriptSectionOutput],
        pages: list[dict[str, Any]] | None = None,
    ) -> VisualPlan:
        results: list[VisualSectionPlan] = []

        for section in sections:
            section_text = extract_section_text(
                document_text,
                section.section_boundary.start_text,
                section.section_boundary.end_text,
                self.config.max_chars,
            )
            script_section = self._find_script_section(section.section_id, script_sections)
            section_images = collect_section_images(section, pages or [])
            prompt_text = self._build_section_prompt_text(section, section_text, script_section)

            structured_llm = self.llm.with_structured_output(VisualSectionPlan, method="json_schema")
            messages = [
                SystemMessage(content=VISUAL_PLANNER_PROMPT),
                build_multimodal_message(prompt_text[: self.config.max_chars], section_images),
            ]
            result = structured_llm.invoke(messages)
            results.append(result)

        return VisualPlan(sections=results)

    def _find_script_section(
        self,
        section_id: int,
        script_sections: list[ScriptSectionOutput],
    ) -> ScriptSectionOutput | None:
        for script_section in script_sections:
            if script_section.section_id == section_id:
                return script_section
        return None

    def _build_section_prompt_text(
        self,
        section: ContentSection,
        section_text: str,
        script_section: ScriptSectionOutput | None,
    ) -> str:
        parts = [
            f"Section ID: {section.section_id}",
            f"Target: {section.target}",
            "Section text:",
            section_text,
        ]

        if script_section is not None:
            parts.extend(
                [
                    "Script data:",
                    f"Narration: {script_section.narration}",
                    f"Approx duration seconds: {script_section.approx_duration_seconds}",
                ]
            )

        if section.images:
            parts.append("Section images:")
            for image in section.images:
                parts.append(f"- {image.image_id}: {image.explanation}")

        return "\n".join(parts)
