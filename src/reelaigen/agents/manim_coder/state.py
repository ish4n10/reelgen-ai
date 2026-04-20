from __future__ import annotations

from typing import Any, TypedDict


class ManimCoderMemory(TypedDict, total=False):
    retrieved_symbols: list[str]
    retrieved_examples: list[str]
    validation_history: list[str]


class ManimCoderState(TypedDict, total=False):
    input: dict[str, Any]
    retrieval_context: dict[str, Any]
    code_candidate: dict[str, Any]
    validation: dict[str, Any]
    memory: ManimCoderMemory


def build_initial_manim_coder_state(input_payload: dict[str, Any]) -> ManimCoderState:
    return {
        "input": input_payload,
        "retrieval_context": {},
        "code_candidate": {},
        "validation": {},
        "memory": {
            "retrieved_symbols": [],
            "retrieved_examples": [],
            "validation_history": [],
        },
    }
