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


class DiagnosticIssue(BaseModel):
    step: int = Field(default=0, description="Observed runtime step.")
    event: str = Field(default="", description="Runtime event name.")
    category: str = Field(default="", description="Issue category such as bbox, layout, or connection.")
    severity: str = Field(default="warning", description="Issue severity.")
    message: str = Field(default="", description="Human-readable issue description.")
    object_ids: list[str] = Field(default_factory=list, description="Related object ids when available.")


class SceneDiagnosticsResult(BaseModel):
    passed: bool = Field(default=False, description="Whether diagnostics are acceptable.")
    static_validation: ValidationResult = Field(default_factory=ValidationResult, description="Static validation output.")
    render_success: bool = Field(default=False, description="Whether scene rendering completed successfully.")
    render_error: str = Field(default="", description="Runtime render error if any.")
    issues: list[DiagnosticIssue] = Field(default_factory=list, description="Structured scene diagnostics.")
    repair_prompt: str = Field(default="", description="Compact LLM-facing prompt describing what to repair.")
    runtime_report: dict = Field(default_factory=dict, description="Raw runtime report.")
