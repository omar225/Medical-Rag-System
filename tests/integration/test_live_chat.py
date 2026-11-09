"""Opt-in real Qwen + BGE-M3 checks. Never downloads or rebuilds anything."""

import os

import pytest
from fastapi.testclient import TestClient

from medical_rag.api.main import create_app


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
