"""Exact cosine retrieval with source freshness and model identity checks."""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from medical_rag.retrieval.corpus import ApprovedCorpus
from medical_rag.retrieval.embeddings import OllamaEmbeddings, validate_text, validate_vectors
from medical_rag.retrieval.numpy_index import check_model, load_index


class SearchResult(TypedDict):
    score: float
    chunk: dict[str, Any]
    document: dict[str, Any]
    embedding: dict[str, str]


class DenseRetriever:
    def __init__(
        self, provider: OllamaEmbeddings, index_dir: Path,
        corpus_loader: Callable[[], ApprovedCorpus],
    ):
        self.provider = provider
        self.index_dir = Path(index_dir)
        self.corpus_loader = corpus_loader
        self._cache: tuple[str, Any] | None = None

    def _index(self, corpus: ApprovedCorpus):
        """Load and model-check the index once per corpus build; a changed corpus forces a reload."""
        if self._cache is None or self._cache[0] != corpus.fingerprint:
            index = load_index(self.index_dir, corpus, self.provider.model)
            check_model(index, self.provider.identity())
            self._cache = (corpus.fingerprint, index)
        return self._cache[1]

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        question = validate_text(question).strip()
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        corpus = self.corpus_loader()
        index = self._index(corpus)
        question_vector = self.provider.embed_question(question, dimension=index.vectors.shape[1])
        question_vector = validate_vectors([question_vector.tolist()], 1, index.vectors.shape[1])[0]
        norm = np.linalg.norm(question_vector.astype(np.float64))
        query = (question_vector / norm).astype(np.float32)
        scores = np.clip(index.vectors @ query, -1.0, 1.0)
        ranking = np.argsort(-scores, kind="stable")[:top_k]
        results = []
        for row in ranking:
            chunk = index.metadata["chunks"][int(row)]
            results.append({
                "score": float(scores[row]),
                "chunk": deepcopy(chunk),
                "document": deepcopy(index.metadata["documents"][chunk["doc_id"]]),
                "embedding": dict(index.metadata["embedding"]),
            })
        return results
