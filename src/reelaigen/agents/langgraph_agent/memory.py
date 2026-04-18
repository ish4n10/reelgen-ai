from __future__ import annotations

from copy import deepcopy
from typing import Any

from .state import PDFContentAgentState


def create_initial_memory() -> dict[str, Any]:
    return {
        "events": [],
        "decisions": [],
        "open_questions": [],
    }


def create_initial_context(thread_id: str) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "current_node": "initialize",
        "completed_nodes": [],
    }


def update_context(state: PDFContentAgentState, node_name: str) -> dict[str, Any]:
    current = deepcopy(state.get("context", {}))
    completed = list(current.get("completed_nodes", []))
    previous = current.get("current_node")

    if previous and previous not in completed:
        completed.append(previous)

    current["current_node"] = node_name
    current["completed_nodes"] = completed
    current.setdefault("thread_id", "default")
    return current


def add_memory_event(state: PDFContentAgentState, node_name: str, summary: str) -> dict[str, Any]:
    memory = deepcopy(state.get("memory", {}))
    events = list(memory.get("events", []))
    events.append({"node": node_name, "summary": summary})
    memory["events"] = events
    memory.setdefault("decisions", [])
    memory.setdefault("open_questions", [])
    return memory
