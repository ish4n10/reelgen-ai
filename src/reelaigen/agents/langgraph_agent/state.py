from __future__ import annotations

from typing import Any, TypedDict


class ParsedPDFPage(TypedDict):
    number: int
    text: str
    image_path: str | None
    image_id: str


class ParsedPDFPayload(TypedDict):
    text: str
    metadata: dict[str, Any]
    pages: list[ParsedPDFPage]


class AlgorithmAnalysisPayload(TypedDict, total=False):
    algorithm_detected: bool
    algorithm_name: str | None
    pseudocode: str | None
    sample_input: dict[str, Any]
    state_trace: list[dict[str, Any]]
    verification_enabled: bool


class UserPromptPayload(TypedDict, total=False):
    raw_prompt: str
    animation_style: str
    script_style: str
    special_images: list[str]
    extra_notes: str


class MemoryEvent(TypedDict):
    node: str
    summary: str


class AgentMemoryPayload(TypedDict, total=False):
    events: list[MemoryEvent]
    decisions: list[str]
    open_questions: list[str]


class AgentRuntimeContext(TypedDict, total=False):
    thread_id: str
    current_node: str
    completed_nodes: list[str]


class PDFContentAgentState(TypedDict, total=False):
    pdf_path: str
    user_prompt: UserPromptPayload
    memory: AgentMemoryPayload
    context: AgentRuntimeContext
    parsed_pdf: ParsedPDFPayload
    algorithm_analysis: AlgorithmAnalysisPayload
    content_analysis: dict[str, Any]
    script_plan: dict[str, Any]
    final_output: dict[str, Any]
