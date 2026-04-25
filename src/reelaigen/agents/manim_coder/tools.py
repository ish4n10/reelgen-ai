from __future__ import annotations

from langchain_core.tools import tool

from .example_retriever import format_examples_for_prompt, search_examples
from .renderer import render_manim_code
from .static_validator import validate_manim_code
from .symbol_lookup import SymbolLookup


@tool("lookup_manim_symbol")
def lookup_manim_symbol(symbol_name: str) -> dict:
    """Look up exact Manim API symbol metadata from the local symbol index."""
    lookup = SymbolLookup()
    return lookup.get_symbol(symbol_name) or {}


@tool("find_manim_symbols_by_tags")
def find_manim_symbols_by_tags(tags: list[str]) -> list[dict]:
    """Find Manim symbols matching all requested tags."""
    lookup = SymbolLookup()
    return lookup.find_symbols_by_tags(tags)


@tool("retrieve_manim_examples")
def retrieve_manim_examples(query: str, k: int = 4) -> dict:
    """Retrieve relevant Manim examples from the FAISS example index."""
    examples = search_examples(query, k=k)
    return {
        "examples": examples,
        "prompt_context": format_examples_for_prompt(examples),
    }


@tool("validate_manim_code")
def validate_manim_code_tool(code: str) -> dict:
    """Run static validation on generated Manim code."""
    return validate_manim_code(code).model_dump()


@tool("render_manim_code")
def render_manim_code_tool(code: str, scene_name: str) -> dict:
    """Render generated Manim code with the local Manim CLI."""
    return render_manim_code(code, scene_name)
