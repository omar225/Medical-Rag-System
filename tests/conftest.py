"""Shared fixtures: the real approved corpus with fake retrieval/generation, and a synthetic corpus."""

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from medical_rag.generation.config import ChatSettings
from medical_rag.generation.ollama import GenerationUsage, GroundedDraft
from medical_rag.knowledge.base import KnowledgeBase
from medical_rag.retrieval.corpus import load_bootstrap_corpus
from medical_rag.retrieval.corpus import prepare_approved_corpus
from medical_rag.retrieval.normalization import AliasRegistry
from medical_rag.services.chat import ChatService


# App tests inject their own service; do not load the knowledge base or contact Ollama at startup.
os.environ.setdefault("PRELOAD_ON_STARTUP", "0")


ROOT = Path(__file__).resolve().parents[1]


class FakeRetriever:
    """Hybrid stand-in: returns the chunks whose section contains the configured text."""

    def __init__(self, corpus):
        self.corpus, self.sections, self.calls = corpus, [], []

    def search(self, query, top_k):
        self.calls.append(query)
        chunks = [c for c in self.corpus.chunks if any(s in c["section"] for s in self.sections)]
        return [{"chunk": c} for c in chunks[:top_k]]


class FakeGenerator:
    def __init__(self):
        self.draft = GroundedDraft(category="out_of_scope", answer="Not covered.", citations=[])
        self.calls = []

    def generate(self, messages, timeout):
        self.calls.append(messages)
        return self.draft, GenerationUsage(1200, 150, 0.0, 0.3, 1.5, 1.9)


@pytest.fixture(scope="session")
def corpus():
    return load_bootstrap_corpus(ROOT / "data/knowledge_base")


@pytest.fixture(scope="session")
def aliases():
    return AliasRegistry.from_file(ROOT / "data/knowledge_base/metadata/retrieval_aliases.json")


@pytest.fixture(scope="session")
def knowledge_base(corpus, aliases):
    return KnowledgeBase(corpus, aliases, FakeRetriever(corpus))


@pytest.fixture
def service(knowledge_base):
    knowledge_base.retriever.sections = []
    knowledge_base.retriever.calls = []
    return ChatService(knowledge_base, FakeGenerator(), ChatSettings())


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
