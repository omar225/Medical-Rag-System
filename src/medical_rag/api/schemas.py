"""Public MTI API models and structural validation rules."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator


ChatStatus = Literal["answered", "out_of_scope", "escalated"]
AnswerLanguage = Literal["ar", "en", "mixed"]
NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class APIModel(BaseModel):
    """Reject unexpected fields and implicit type conversions."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ChatRequest(APIModel):
    session_id: str | None = None
    message: NonBlankText


class Citation(APIModel):
    doc_id: NonBlankText
    title: NonBlankText
    section: NonBlankText
    snippet: NonBlankText


class ChatResponse(APIModel):
    answer: NonBlankText
    citations: list[Citation]
    status: ChatStatus
    language: AnswerLanguage

    @model_validator(mode="after")
    def validate_citations(self) -> Self:
        if self.status == "answered" and not self.citations:
            raise ValueError("answered requires at least one citation")
        if self.status != "answered" and self.citations:
            raise ValueError("out_of_scope and escalated require empty citations")
        return self


class ErrorResponse(APIModel):
    error: NonBlankText


class HealthResponse(APIModel):
    status: Literal["ok"]
    llm: NonBlankText
    embedding: NonBlankText
