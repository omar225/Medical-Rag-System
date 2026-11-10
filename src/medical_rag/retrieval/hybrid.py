"""One normalized dense query plus BM25, combined by reciprocal rank fusion."""

from copy import deepcopy

from medical_rag.retrieval.dense import DenseRetriever
from medical_rag.retrieval.lexical import LexicalRetriever
from medical_rag.retrieval.numpy_index import StaleIndexError


RRF_K = 60
CANDIDATE_COUNT = 20


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    if type(k) is not int or k <= 0:
        raise ValueError("RRF constant must be a positive integer")
    scores = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("A ranking must not contain duplicate chunk IDs")
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, lexical: LexicalRetriever):
        self.dense, self.lexical = dense, lexical

    def search(self, question: str, top_k: int = 5):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        pool = max(CANDIDATE_COUNT, top_k)
        lexical, query, corpus = self.lexical.search_with_normalization(question, pool)
        # The baseline object/code is unchanged; only this mode supplies rewritten input.
        dense = self.dense.search(query.rewritten, pool)
        canonical = {chunk["chunk_id"]: chunk for chunk in corpus.chunks}
        for result in dense:
            chunk = result["chunk"]
            if chunk != canonical.get(chunk["chunk_id"]) or result["document"] != corpus.documents.get(chunk["doc_id"]):
                raise StaleIndexError("Corpus changed between hybrid branches; retry against one build")
        components = {"dense": dense, "lexical": lexical}
        lookups = {mode: {result["chunk"]["chunk_id"]: (rank, result) for rank, result in enumerate(results, 1)} for mode, results in components.items()}
        ranking = reciprocal_rank_fusion([[result["chunk"]["chunk_id"] for result in results] for results in components.values()])
        output = []
        for chunk_id, score in ranking[:top_k]:
            source = lookups["dense"].get(chunk_id) or lookups["lexical"][chunk_id]
            result = deepcopy(source[1])
            result["score"] = score
            result["embedding"] = dict(dense[0]["embedding"])
            result["retrieval"] = {
                "mode": "hybrid", "score_type": "rrf", "query": query.metadata(),
                "corpus_sha256": corpus.fingerprint, "rrf_k": RRF_K, "candidate_count": pool,
                "ranks": {mode: lookup[chunk_id][0] if chunk_id in lookup else None for mode, lookup in lookups.items()},
                "scores": {mode: lookup[chunk_id][1]["score"] if chunk_id in lookup else None for mode, lookup in lookups.items()},
            }
            output.append(result)
        return output
