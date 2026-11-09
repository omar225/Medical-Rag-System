"""Approved-source fixtures, mocked retrieval and generation; no model calls."""

from copy import deepcopy
from pathlib import Path

import pytest

from medical_rag.generation.config import ChatSettings
from medical_rag.generation.ollama import GenerationUsage, GroundedDraft
from medical_rag.retrieval.corpus import prepare_approved_corpus
from medical_rag.retrieval.normalization import AliasRegistry
from medical_rag.services.chat import ChatService


@pytest.fixture
def corpus():
    document = {"doc_id": "TEST-01", "title": "Synthetic medication table", "filename": "synthetic.docx",
                "build_id": "a" * 64, "source_sha256": "b" * 64, "config_id": "c" * 64,
                "chunk_count": 1}
    chunk = {**document, "chunk_id": "TEST-01-chunk-1", "section": "NSAIDs",
             "section_path": ["NSAIDs"], "section_start": 1,
             "source_spans": [{"block_position": 1}],
             "text": "Active ingredient\nCataflam | Diclofenac potassium\nCommon side effects:\nGastric irritation.\n"}
    return prepare_approved_corpus([chunk], {document["doc_id"]: document})


@pytest.fixture
def aliases():
    path = Path(__file__).resolve().parents[2] / "data/knowledge_base/metadata/retrieval_aliases.json"
    return AliasRegistry.from_file(path)


class FakeRetriever:
    def __init__(self, corpus):
        self.results = [{"score": .03, "chunk": deepcopy(corpus.chunks[0]),
                         "document": deepcopy(corpus.documents["TEST-01"]),
                         "retrieval": {"corpus_sha256": corpus.fingerprint}}]
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return deepcopy(self.results)


class FakeGenerator:
    def __init__(self):
        answer = "Cataflam contains Diclofenac potassium."
        self.draft = GroundedDraft(answer=answer, evidence_sufficient=True,
                                   support=[{"label": "E1", "quote": "Cataflam | Diclofenac potassium"}])
        self.calls = []

    def generate(self, messages, timeout, *, quote_choices=None):
        self.calls.append((messages, timeout))
        return self.draft, GenerationUsage(200, 50, .1, .2, .3, .6)


@pytest.fixture
def service(corpus, aliases):
    return ChatService(FakeRetriever(corpus), FakeGenerator(), lambda: corpus, lambda: aliases, ChatSettings())
