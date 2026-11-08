"""Opt-in live check; the default suite never requires Ollama or a model download."""

import os

import numpy as np
import pytest

from medical_rag.retrieval.config import RetrievalSettings
from medical_rag.retrieval.embeddings.ollama import OllamaEmbeddings


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
