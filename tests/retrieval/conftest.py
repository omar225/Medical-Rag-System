from copy import deepcopy

import numpy as np
import pytest

from medical_rag.retrieval.corpus import prepare_approved_corpus


class SyntheticEmbeddings:
    model = "bge-m3:latest"

    def __init__(self):
        self.digest = "d" * 64
        self.batches = []
        self.questions = []
        self.values = {"Alpha fixture": [3, 0, 0], "Beta fixture": [0.6, 0.8, 0], "Gamma fixture": [0, 2, 0]}

    def identity(self):
        return {"provider": "ollama", "model": self.model, "digest": self.digest}

    def embed_texts(self, texts, dimension=None):
        self.batches.append(list(texts))
        return np.asarray([self.values.get(text, [1, 0, 0]) for text in texts], dtype=np.float32)

    def embed_question(self, question, dimension=None):
        self.questions.append(question)
        return np.asarray([2, 0, 0], dtype=np.float32)


@pytest.fixture
def synthetic_provider():
    return SyntheticEmbeddings()


@pytest.fixture
def synthetic_corpus():
    documents = {
        "test-A": {"doc_id": "test-A", "title": "Synthetic A", "filename": "a.docx", "build_id": "a" * 64, "source_sha256": "1" * 64, "config_id": "c" * 64, "chunk_count": 2, "applied_corrections": []},
        "test-B": {"doc_id": "test-B", "title": "Synthetic B", "filename": "b.docx", "build_id": "b" * 64, "source_sha256": "2" * 64, "config_id": "c" * 64, "chunk_count": 1, "applied_corrections": []},
    }
    chunks = []
    for doc_id, chunk_id, text in (("test-A", "a-2", "Beta fixture"), ("test-B", "b-1", "Gamma fixture"), ("test-A", "a-1", "Alpha fixture")):
        record = documents[doc_id]
        chunks.append({
            **{key: record[key] for key in ("doc_id", "title", "filename", "build_id", "source_sha256", "config_id")},
            "chunk_id": chunk_id, "section": "Source topic", "section_path": ["Source topic"],
            "text": text, "source_spans": [{"block_position": 1}], "section_start": 1,
            "context_topics": ["Source topic"], "scope_uncertain": False,
            "correction_positions": [], "extra_provenance": {"preserve": "عربي English"},
        })
    return prepare_approved_corpus(deepcopy(chunks), documents)
