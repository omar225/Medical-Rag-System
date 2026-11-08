"""Structural response guarantees; fixture text is not medical evidence."""

import pytest
from pydantic import ValidationError

from medical_rag.api.schemas import ChatRequest, ChatResponse, Citation


@pytest.fixture
def citation():
    return {
        "doc_id": "TEST-01",
        "title": "Synthetic test document",
        "section": "Test section",
        "snippet": "Synthetic excerpt for schema validation only.",
    }


def test_session_is_optional_and_message_is_trimmed():
    request = ChatRequest(message="  Hello  ")

    assert request.session_id is None
    assert request.message == "Hello"
    assert ChatRequest(session_id="abc123", message="Hello").session_id == "abc123"


@pytest.mark.parametrize("language", ["ar", "en", "mixed"])
def test_answered_accepts_valid_citation_and_language(citation, language):
    response = ChatResponse(
        answer="Synthetic test answer", citations=[citation], status="answered", language=language
    )

    assert response.model_dump() == {
        "answer": "Synthetic test answer",
        "citations": [citation],
        "status": "answered",
        "language": language,
    }


@pytest.mark.parametrize("status", ["unknown", "Answered", "", None])
def test_invalid_status_is_rejected(status):
    with pytest.raises(ValidationError):
        ChatResponse(answer="Test", citations=[], status=status, language="en")


@pytest.mark.parametrize("language", ["fr", "EN", "", None])
def test_invalid_language_is_rejected(language):
    with pytest.raises(ValidationError):
        ChatResponse(answer="Test", citations=[], status="out_of_scope", language=language)


def test_answered_requires_a_citation():
    with pytest.raises(ValidationError, match="answered requires at least one citation"):
        ChatResponse(answer="Test", citations=[], status="answered", language="en")


@pytest.mark.parametrize("status", ["out_of_scope", "escalated"])
def test_non_answered_status_requires_empty_citations(citation, status):
    response = ChatResponse(answer="Test", citations=[], status=status, language="en")
    assert response.citations == []

    with pytest.raises(ValidationError, match="require empty citations"):
        ChatResponse(answer="Test", citations=[citation], status=status, language="en")


@pytest.mark.parametrize("field", ["doc_id", "title", "section", "snippet"])
@pytest.mark.parametrize("invalid_value", ["", " \t\n ", None, 123])
def test_answered_rejects_invalid_citation_fields(citation, field, invalid_value):
    citation[field] = invalid_value

    with pytest.raises(ValidationError):
        ChatResponse(answer="Test", citations=[citation], status="answered", language="en")


@pytest.mark.parametrize("field", ["doc_id", "title", "section", "snippet"])
def test_answered_rejects_missing_citation_fields(citation, field):
    del citation[field]

    with pytest.raises(ValidationError):
        ChatResponse(answer="Test", citations=[citation], status="answered", language="en")


def test_citation_rejects_extra_fields(citation):
    with pytest.raises(ValidationError):
        Citation(**citation, url="unexpected")


def test_response_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ChatResponse(
            answer="Test", citations=[], status="out_of_scope", language="en", session_id="extra"
        )


@pytest.mark.parametrize("answer", ["", " \t\n ", None, 123])
def test_response_rejects_blank_or_non_text_answer(answer):
    with pytest.raises(ValidationError):
        ChatResponse(answer=answer, citations=[], status="out_of_scope", language="en")
