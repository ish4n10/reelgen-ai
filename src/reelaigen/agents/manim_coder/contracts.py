from __future__ import annotations

from typing import Literal

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
    block_id: str = Field(default="", description="Runtime block id when available.")
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


class CanvasSpec(BaseModel):
    width: float = Field(default=14.22, description="Logical scene width in Manim units.")
    height: float = Field(default=8.0, description="Logical scene height in Manim units.")


class CameraFrameIR(BaseModel):
    left: float = Field(default=-6.5, description="Left camera frame bound.")
    bottom: float = Field(default=-3.0, description="Bottom camera frame bound.")
    right: float = Field(default=6.5, description="Right camera frame bound.")
    top: float = Field(default=3.0, description="Top camera frame bound.")


class SceneObjectIR(BaseModel):
    object_id: str = Field(..., description="Stable scene object id.")
    object_type: str = Field(..., description="High-level Manim primitive such as Text, Rectangle, or Arrow.")
    label: str | None = Field(default=None, description="Short label when the object is textual or labeled.")
    center: list[float] = Field(default_factory=list, description="Target object center in scene coordinates.")
    size: list[float] = Field(default_factory=list, description="Target object width and height.")
    color: str = Field(default="WHITE", description="Requested Manim color constant.")
    semantic_role: str = Field(default="node", description="Role such as node, label, annotation, or connector-anchor.")
    semantic_group: str = Field(default="default", description="Group used for size and alignment consistency.")
    z_index: int = Field(default=0, description="Rendering order hint.")
    props: dict = Field(default_factory=dict, description="Additional object-specific planning hints.")


class ConnectorIR(BaseModel):
    connector_id: str = Field(..., description="Stable connector id.")
    source_object_id: str = Field(..., description="Source object id.")
    target_object_id: str = Field(..., description="Target object id.")
    routing: Literal["horizontal", "vertical", "orthogonal", "diagonal"] = Field(
        default="horizontal",
        description="Desired routing style for the connector.",
    )
    label: str | None = Field(default=None, description="Optional connector label.")
    color: str = Field(default="WHITE", description="Requested connector color.")
    stroke_width: float = Field(default=2.5, description="Requested connector stroke width.")
    props: dict = Field(default_factory=dict, description="Additional connector-specific hints.")


class AnimationBlockIR(BaseModel):
    block_id: str = Field(..., description="Stable animation block id.")
    title: str = Field(default="", description="Short block title.")
    description: str = Field(default="", description="What happens in this block.")
    object_ids: list[str] = Field(default_factory=list, description="Objects that appear or change in this block.")
    connector_ids: list[str] = Field(default_factory=list, description="Connectors used in this block.")
    animation_style: list[str] = Field(default_factory=list, description="Preferred animation verbs such as FadeIn or GrowArrow.")


class SceneIR(BaseModel):
    scene_name: str = Field(..., description="Python scene class name to generate.")
    scene_goal: str = Field(..., description="Short educational goal for the scene.")
    layout_strategy: str = Field(default="left_to_right", description="High-level layout strategy.")
    flow_axis: Literal["x", "y"] = Field(default="x", description="Primary scene flow axis.")
    canvas: CanvasSpec = Field(default_factory=CanvasSpec, description="Canvas specification.")
    camera_frame: CameraFrameIR = Field(default_factory=CameraFrameIR, description="Target camera frame.")
    objects: list[SceneObjectIR] = Field(default_factory=list, description="Planned scene objects.")
    connectors: list[ConnectorIR] = Field(default_factory=list, description="Planned scene connectors.")
    animation_blocks: list[AnimationBlockIR] = Field(default_factory=list, description="Ordered scene animation blocks.")
    style_notes: list[str] = Field(default_factory=list, description="High-level style constraints.")
    source_context: dict = Field(default_factory=dict, description="Original planner input retained for traceability.")


class SceneIRValidationResult(BaseModel):
    passed: bool = Field(default=False, description="Whether the Scene IR passed deterministic validation.")
    errors: list[str] = Field(default_factory=list, description="Hard validation errors.")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings.")
