from __future__ import annotations

from pydantic import BaseModel, Field


class ManimCoderInput(BaseModel):
    section_id: int = Field(..., description="Section id from the planning pipeline.")
    target: str = Field(..., description="Short goal for the section.")
    narration: str = Field(..., description="Narration text for the section.")
    visual_plan: dict = Field(..., description="Visual planner output for the section.")
    user_prompt: dict = Field(default_factory=dict, description="Optional user instructions.")


class RetrievedContext(BaseModel):
    symbols: list[str] = Field(default_factory=list, description="Relevant Manim symbols.")
    examples: list[str] = Field(default_factory=list, description="Relevant code examples or snippets.")
    notes: list[str] = Field(default_factory=list, description="Other helpful retrieval notes.")


class CodeCandidate(BaseModel):
    code: str = Field(default="", description="Generated Manim code.")
    scene_class_name: str = Field(default="", description="Generated scene class name.")
    reasoning: str = Field(default="", description="Short note about why this code was chosen.")


class ValidationResult(BaseModel):
    passed: bool = Field(default=False, description="Whether validation passed.")
    errors: list[str] = Field(default_factory=list, description="Validation errors.")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings.")
