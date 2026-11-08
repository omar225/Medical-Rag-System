"""Stage 1 HTTP contract checks; no external services or source files needed."""

import pytest
from fastapi.testclient import TestClient

from medical_rag.api.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-llm")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    with TestClient(create_app()) as test_client:
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
            "llm": "qwen2.5:7b",
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
    assert response.json() == {"error": "Chat pipeline is not implemented yet"}


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
