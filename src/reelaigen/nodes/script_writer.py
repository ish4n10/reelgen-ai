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
from reelaigen.llm.prompts import SCRIPT_WRITER_PROMPT
from reelaigen.nodes.content_parser import ContentSection
from reelaigen.nodes.section_utils import collect_section_images, extract_section_text


class ScriptTimingBeat(BaseModel):
    start_second: float = Field(..., description="Approximate start time for this narration beat.")
    end_second: float = Field(..., description="Approximate end time for this narration beat.")
    note: str = Field(..., description="What is narrated in this beat.")


class ScriptSectionOutput(BaseModel):
    section_id: int = Field(..., description="Section index from the content analyzer.")
    target: str = Field(..., description="Short target for this section.")
    section_text: str = Field(..., description="Extracted source text used to write the script.")
    narration: str = Field(..., description="Narration script for this section.")
    approx_duration_seconds: int = Field(..., description="Approximate target duration in seconds.")
    min_duration_seconds: int = Field(..., description="Minimum acceptable duration in seconds.")
    max_duration_seconds: int = Field(..., description="Maximum acceptable duration in seconds.")
    timing_estimate: list[ScriptTimingBeat] = Field(default_factory=list, description="Beat-level timing estimate.")


class ScriptPlan(BaseModel):
    sections: list[ScriptSectionOutput] = Field(default_factory=list)


@dataclass
class ScriptWriterConfig:
    max_chars: int = 6000


class ScriptWriter:
    def __init__(self, llm=None, config: ScriptWriterConfig | None = None) -> None:
        self.llm = llm or get_llm()
        self.config = config or ScriptWriterConfig()

    def run(
        self,
        document_text: str,
        sections: list[ContentSection],
        pages: list[dict[str, Any]] | None = None,
        algorithm_context: dict[str, Any] | None = None,
    ) -> ScriptPlan:
        results: list[ScriptSectionOutput] = []

        for section in sections:
            section_text = extract_section_text(
                document_text,
                section.section_boundary.start_text,
                section.section_boundary.end_text,
                self.config.max_chars,
            )
            prompt_text = self._build_section_prompt_text(section, section_text, algorithm_context)
            section_images = collect_section_images(section, pages or [])

            structured_llm = self.llm.with_structured_output(ScriptSectionOutput, method="json_schema")
            messages = [
                SystemMessage(content=SCRIPT_WRITER_PROMPT),
                build_multimodal_message(prompt_text[: self.config.max_chars], section_images),
            ]
            result = structured_llm.invoke(messages)
            results.append(result)

        return ScriptPlan(sections=results)

    def _build_section_prompt_text(
        self,
        section: ContentSection,
        section_text: str,
        algorithm_context: dict[str, Any] | None,
    ) -> str:
        parts = [
            f"Section ID: {section.section_id}",
            f"Target: {section.target}",
            "Section text:",
            section_text,
        ]

        if section.images:
            parts.append("Section images:")
            for image in section.images:
                parts.append(f"- {image.image_id}: {image.explanation}")

        if algorithm_context:
            parts.append("Algorithm context:")
            parts.append(str(algorithm_context))

        return "\n".join(parts)
