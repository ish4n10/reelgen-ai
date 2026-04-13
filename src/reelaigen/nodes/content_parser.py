from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reelaigen.llm.integration import get_mistral_llm
from reelaigen.llm.prompts import CONTENT_ANALYZER_PROMPT


class SectionBoundary(BaseModel):
    start_text: str = Field(..., description="Short snippet where the section starts.")
    end_text: str = Field(..., description="Short snippet where the section ends.")


class ContentSection(BaseModel):
    section_id: int = Field(..., description="Section index starting from 0.")
    section_boundary: SectionBoundary
    target: str = Field(..., description="Main idea of the section.")


class ContentAnalysis(BaseModel):
    parent_content_type: str = Field(..., description="Main document type such as math_explainer, cs_explainer, finance, or general_education.")
    sections: list[ContentSection] = Field(default_factory=list)


@dataclass
class ContentParserConfig:
    max_chars: int = 12000


class ContentParser:
    def __init__(self, llm=None, config: ContentParserConfig | None = None) -> None:
        self.llm = llm or get_mistral_llm()
        self.config = config or ContentParserConfig()

    def run(self, document_text: str) -> ContentAnalysis:
        if not document_text.strip():
            raise ValueError("document_text is empty")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    CONTENT_ANALYZER_PROMPT
                ),
                (
                    "human",
                    "{document_text}",
                ),
            ]
        )

        structured_llm = self.llm.with_structured_output(ContentAnalysis, method="json_schema")
        messages = prompt.invoke({"document_text": document_text[: self.config.max_chars]})
        return structured_llm.invoke(messages)


