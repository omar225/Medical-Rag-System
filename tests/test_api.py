"""HTTP contract and public schemas; no external services."""

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from medical_rag.api.main import create_app, get_chat_service
from medical_rag.api.schemas import ChatRequest, ChatResponse, Citation
from medical_rag.generation.ollama import GenerationError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    application = create_app()

    class UnavailableService:
        def run(self, text, history=()):
            raise GenerationError("Synthetic unavailable model")

    application.dependency_overrides[get_chat_service] = lambda: UnavailableService()
    with TestClient(application) as test_client:
        yield test_client


def test_health_exact_contract(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm": "test-llm",
        "embedding": "test-embedding",
    }


def test_health_default_configuration(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "llm": "qwen3.5:4b",
            "embedding": "bge-m3",
        }


@pytest.mark.parametrize("session", [{"session_id": "abc123"}, {}, {"session_id": None}])
def test_valid_chat_returns_operational_error(client, session):
    response = client.post(
        "/chat",
        json={**session, "message": "What is the active ingredient in Cataflam?"},
    )

    assert response.status_code == 500
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"error": "Chat service failed or timed out; please retry later."}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"session_id": "abc123"},
        {"message": ""},
        {"message": " \t\n "},
        {"message": None},
        {"message": 123},
        {"message": True},
        {"message": ["text"]},
        {"message": {"text": "hello"}},
        {"message": "hello", "session_id": 123},
        {"message": "hello", "unexpected": "field"},
    ],
)
def test_invalid_chat_input_returns_422(client, payload):
    response = client.post("/chat", json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_openapi_documents_exact_public_contract(client):
    specification = client.get("/openapi.json").json()
    operation = specification["paths"]["/chat"]["post"]
    schemas = specification["components"]["schemas"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ChatResponse"
    }
    assert operation["responses"]["500"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ErrorResponse"
    }
    assert set(schemas["ChatRequest"]["required"]) == {"message"}
    assert set(schemas["ChatResponse"]["properties"]) == {
        "answer", "citations", "status", "language"
    }
    assert set(schemas["Citation"]["properties"]) == {
        "doc_id", "title", "section", "snippet"
    }


class RecordingService:
    """Answers out_of_scope and records the history it was given."""

    def __init__(self):
        self.histories = []

    def run(self, text, history=()):
        from medical_rag.api.schemas import ChatResponse
        from medical_rag.services.chat import ChatRun
        self.histories.append(history)
        return ChatRun(ChatResponse(answer=f"reply to {text}", citations=[], status="out_of_scope",
                                    language="en"), "test", 0.01)


def test_session_history_is_kept_per_session_and_web_ui_prefix_works():
    service = RecordingService()
    application = create_app()
    application.dependency_overrides[get_chat_service] = lambda: service
    with TestClient(application) as client:
        assert client.post("/chat", json={"session_id": "s1", "message": "first"}).status_code == 200
        assert client.post("/api/chat", json={"session_id": "s1", "message": "second"}).status_code == 200
        client.post("/chat", json={"session_id": "s2", "message": "other"})
        client.post("/chat", json={"message": "anonymous"})
        assert client.get("/api/health").status_code == 200
    first, second, other, anonymous = service.histories
    assert first == () and other == () and anonymous == ()
    assert [(t.question, t.answer) for t in second] == [("first", "reply to first")]


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
