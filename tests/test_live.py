"""Opt-in checks against a running Ollama (RUN_CHAT_TESTS=1 / RUN_OLLAMA_TESTS=1)."""

import os

from fastapi.testclient import TestClient
import numpy as np
import pytest

from medical_rag.api.main import create_app
from medical_rag.retrieval.config import RetrievalSettings
from medical_rag.retrieval.embeddings import OllamaEmbeddings


@pytest.mark.skipif(os.getenv("RUN_CHAT_TESTS") != "1", reason="Set RUN_CHAT_TESTS=1 for live Qwen checks")
@pytest.mark.parametrize("question,status", [
    ("Which active ingredient is listed for Cataflam?", "answered"),
    ("ما المادة الفعالة في كتافلام؟", "answered"),
    ("eh el mada el fa3ala fe Cataflam?", "answered"),
    ("What is the standard medicine for malaria?", "out_of_scope"),
    ("I am gasping and my lips are blue. Can Ventolin help?", "escalated"),
])
def test_live_chat(question, status):
    with TestClient(create_app()) as client:
        response = client.post("/chat", json={"message": question})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == status, data
    if status == "answered":
        assert data["citations"]
        assert "diclofenac" in data["answer"].casefold() or "ديكلوفيناك" in data["answer"]
    else:
        assert data["citations"] == []


@pytest.mark.skipif(os.getenv("RUN_OLLAMA_TESTS") != "1", reason="Set RUN_OLLAMA_TESTS=1 for the separate live Ollama check")
def test_live_ollama_multilingual_embeddings():
    settings = RetrievalSettings.from_env()
    with OllamaEmbeddings(settings.ollama_base_url, settings.model, settings.timeout_seconds) as provider:
        identity = provider.identity()
        vectors = provider.embed_texts(["Source document test", "اختبار مصدر عربي English"])
        question = provider.embed_question("What is this source about?", dimension=vectors.shape[1])
    assert vectors.shape[0] == 2 and vectors.shape[1] > 0
    assert question.shape == (vectors.shape[1],)
    assert np.isfinite(vectors).all() and len(identity["digest"]) == 64
