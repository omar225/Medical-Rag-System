"""Reproducible labeled-evidence evaluation, independent of retrieval rankings."""

from collections import defaultdict
from statistics import mean, median
from time import perf_counter
from typing import Any

from medical_rag.retrieval.corpus import ApprovedCorpus


def resolve_evidence(dataset: dict[str, Any], corpus: ApprovedCorpus) -> dict[str, set[str]]:
    if dataset.get("corpus_sha256") != corpus.fingerprint:
        raise ValueError("Benchmark labels belong to a different corpus snapshot; review labels before evaluation")
    resolved = {}
    for topic_id, topic in dataset["topics"].items():
        ids = set()
        for evidence in topic["evidence"]:
            matches = [c for c in corpus.chunks if c["doc_id"] == evidence["doc_id"] and c.get("section_start") == evidence["section_start"]]
            if len(matches) != 1 or any(snippet not in matches[0]["text"] for snippet in evidence["snippets"]):
                raise ValueError(f"Benchmark evidence is missing or changed for {topic_id}")
            ids.add(matches[0]["chunk_id"])
        if not ids:
            raise ValueError("Answerable benchmark topics require labeled evidence")
        resolved[topic_id] = ids
    return resolved


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: set[str]) -> dict[str, dict[str, float]]:
    if not relevant_ids:
        raise ValueError("Recall is undefined for a query without relevant evidence")
    recall, hit = {}, {}
    for k in (1, 3, 5):
        found = len(set(retrieved_ids[:k]) & relevant_ids)
        recall[str(k)] = found / len(relevant_ids)
        hit[str(k)] = float(found > 0)
    return {"recall": recall, "hit": hit}


def evaluate_retrieval(
    retriever: Any, corpus: ApprovedCorpus, dataset: dict[str, Any],
) -> dict[str, Any]:
    labels = resolve_evidence(dataset, corpus)
    rows = []
    for case in dataset["queries"]:
        topic_id = case.get("topic_id")
        if topic_id is not None and topic_id not in labels:
            raise ValueError("Query refers to an unlabeled benchmark topic")
        started = perf_counter()
        results = retriever.search(case["question"], top_k=5)
        latency = perf_counter() - started
        expected = labels[topic_id] if topic_id is not None else set()
        metrics = retrieval_metrics([r["chunk"]["chunk_id"] for r in results], expected) if expected else None
        rows.append({
            **case, "latency_seconds": latency, "expected_chunk_ids": sorted(expected),
            "metrics": metrics, "results": results,
        })
    answerable = [row for row in rows if row["metrics"] is not None]
    if not answerable:
        raise ValueError("Benchmark has no answerable queries")
    groups = defaultdict(list)
    for row in answerable:
        groups[row["language"]].append(row)

    def aggregate(group):
        return {
            metric: {str(k): mean(row["metrics"][metric][str(k)] for row in group) for k in (1, 3, 5)}
            for metric in ("recall", "hit")
        }

    return {
        "schema_version": 1,
        "corpus_sha256": corpus.fingerprint,
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "out_of_kb_count": len(rows) - len(answerable),
        "metrics": aggregate(answerable),
        "by_language": {language: {"count": len(group), **aggregate(group)} for language, group in groups.items()},
        "latency_seconds": {
            "mean": mean(row["latency_seconds"] for row in rows),
            "median": median(row["latency_seconds"] for row in rows),
            "max": max(row["latency_seconds"] for row in rows),
        },
        "queries": rows,
    }
