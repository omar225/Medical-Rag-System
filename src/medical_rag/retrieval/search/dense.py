"""Exact cosine retrieval with source freshness and model identity checks."""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from medical_rag.retrieval.corpus import ApprovedCorpus
from medical_rag.retrieval.embeddings.ollama import OllamaEmbeddings, validate_text, validate_vectors
from medical_rag.retrieval.vector_store.numpy_index import check_model, load_index


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

    def search(self, question: str, top_k: int = 5) -> list[SearchResult]:
        question = validate_text(question).strip()
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        # Reload current selection so replacements/deletions cannot silently use an old index.
        corpus = self.corpus_loader()
        index = load_index(self.index_dir, corpus, self.provider.model)
        check_model(index, self.provider.identity())
        question_vector = self.provider.embed_question(question, dimension=index.vectors.shape[1])
        question_vector = validate_vectors([question_vector.tolist()], 1, index.vectors.shape[1])[0]
        check_model(index, self.provider.identity())
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
