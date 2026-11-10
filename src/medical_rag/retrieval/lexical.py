"""Local BM25 over derived views of the same validated, published corpus."""

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from medical_rag.retrieval.corpus import ApprovedCorpus
from medical_rag.retrieval.embeddings import validate_text
from medical_rag.retrieval.normalization import AliasRegistry, tokenize
from medical_rag.retrieval.numpy_index import load_index


class BM25:
    def __init__(self, corpus: ApprovedCorpus, aliases: AliasRegistry, k1: float = 1.2, b: float = .75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.aliases = aliases
        self.available = aliases.available_for(corpus)
        self.k1, self.b = k1, b
        self.counts = []
        for chunk in corpus.chunks:
            text = chunk["text"]
            # Only source headings/topics; filename-derived titles can be overbroad.
            extras = [value for value in [*chunk["section_path"], *chunk.get("context_topics", [])] if value not in text]
            view = aliases.normalize_query("\n".join([text, *extras]), self.available).rewritten
            self.counts.append(Counter(tokenize(view)))
        self.lengths = np.asarray([sum(counts.values()) for counts in self.counts], dtype=np.float64)
        self.average_length = float(self.lengths.mean()) or 1.0
        self.document_frequency = Counter(term for counts in self.counts for term in counts)

    def scores(self, question: str) -> np.ndarray:
        result = np.zeros(len(self.counts), dtype=np.float64)
        for term in set(tokenize(question)):
            frequency = self.document_frequency[term]
            if frequency == 0:
                continue
            idf = math.log(1 + (len(self.counts) - frequency + .5) / (frequency + .5))
            for row, counts in enumerate(self.counts):
                tf = counts[term]
                if tf:
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[row] / self.average_length)
                    result[row] += idf * tf * (self.k1 + 1) / denominator
        return result


class LexicalRetriever:
    def __init__(
        self, index_dir: Path, model: str, corpus_loader: Callable[[], ApprovedCorpus],
        alias_loader: Callable[[], AliasRegistry],
    ):
        self.index_dir, self.model = Path(index_dir), model
        self.corpus_loader, self.alias_loader = corpus_loader, alias_loader
        self._key = None
        self._bm25 = None

    def search_with_normalization(self, question: str, top_k: int = 5):
        validate_text(question)
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        corpus = self.corpus_loader()
        aliases = self.alias_loader()
        key = (corpus.fingerprint, aliases.fingerprint)
        if key != self._key:
            # Same published snapshot gate as dense search, checked once per corpus/alias build.
            self._embedding = dict(load_index(self.index_dir, corpus, self.model).metadata["embedding"])
            self._bm25 = BM25(corpus, aliases)
            self._key = key
        bm25 = self._bm25
        query = aliases.normalize_query(question, bm25.available)
        scores = bm25.scores(query.rewritten)
        ranking = [int(row) for row in np.argsort(-scores, kind="stable") if scores[row] > 0][:top_k]
        results = []
        for row in ranking:
            chunk = corpus.chunks[row]
            results.append({
                "score": float(scores[row]), "chunk": deepcopy(chunk),
                "document": deepcopy(corpus.documents[chunk["doc_id"]]),
                "embedding": dict(self._embedding),
                "retrieval": {"mode": "lexical", "score_type": "bm25", "query": query.metadata(), "corpus_sha256": corpus.fingerprint},
            })
        return results, query, corpus

    def search(self, question: str, top_k: int = 5):
        return self.search_with_normalization(question, top_k)[0]
