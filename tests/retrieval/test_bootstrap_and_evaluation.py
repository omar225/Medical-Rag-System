from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from medical_rag.retrieval.corpus import CorpusError, load_bootstrap_corpus
from medical_rag.retrieval.evaluation import evaluate_retrieval, resolve_evidence, retrieval_metrics


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bootstrap_copy(tmp_path):
    source = ROOT / "data/knowledge_base"
    target = tmp_path / "kb"
    for name in ("source", "processed", "chunks", "metadata"):
        shutil.copytree(source / name, target / name)
    return target


def test_bootstrap_indexes_only_the_selected_nine_documents(bootstrap_copy):
    candidate = bootstrap_copy / "chunks/candidate-tenth.json"
    candidate.write_text('[{"doc_id":"candidate-tenth","text":"Not approved"}]', encoding="utf-8")
    corpus = load_bootstrap_corpus(bootstrap_copy)
    assert len(corpus.documents) == 9 and len(corpus.chunks) == 113
    assert all(c["doc_id"] != "candidate-tenth" for c in corpus.chunks)
    manifest_path = bootstrap_copy / "metadata/generated/ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"].append({"doc_id": "candidate-tenth"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CorpusError, match="exactly the approved"):
        load_bootstrap_corpus(bootstrap_copy)


@pytest.mark.parametrize("change", ["source", "processed_build", "chunk_build", "corrections", "missing_chunks"])
def test_bootstrap_stale_source_artifacts_are_rejected(bootstrap_copy, change):
    if change == "source":
        path = next((bootstrap_copy / "source/KB-01").glob("*.docx"))
        path.write_bytes(path.read_bytes() + b"Changed source snapshot")
    elif change == "processed_build":
        path = bootstrap_copy / "processed/KB-01.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["build_id"] = "f" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
    elif change == "chunk_build":
        path = bootstrap_copy / "chunks/KB-01.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data[0]["build_id"] = "f" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
    elif change == "corrections":
        path = bootstrap_copy / "metadata/structure_corrections.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["KB-01"]["rules"][0]["reason"] = "Changed structural review"
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        (bootstrap_copy / "chunks/KB-01.json").unlink()
    with pytest.raises(CorpusError):
        load_bootstrap_corpus(bootstrap_copy)


def test_recall_is_fraction_of_all_labeled_evidence_not_hit_rate():
    metrics = retrieval_metrics(["a", "irrelevant", "b", "c", "d"], {"a", "b", "c", "d"})
    assert metrics["recall"] == {"1": .25, "3": .5, "5": 1.0}
    assert metrics["hit"] == {"1": 1.0, "3": 1.0, "5": 1.0}


def test_frozen_benchmark_resolves_all_manually_labeled_evidence():
    corpus = load_bootstrap_corpus(ROOT / "data/knowledge_base")
    dataset = json.loads((ROOT / "tests/evaluation/datasets/retrieval_queries.json").read_text(encoding="utf-8"))
    labels = resolve_evidence(dataset, corpus)
    assert {key: len(ids) for key, ids in labels.items()} == {
        "cataflam_ingredient": 5, "brufen_same_ingredient": 7, "arb_contraindications": 1,
        "salbutamol_effects": 2, "metformin_contraindications": 1, "metronidazole_alcohol": 1,
    }
    changed = deepcopy(dataset)
    changed["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="different corpus"):
        resolve_evidence(changed, corpus)


def test_out_of_kb_results_are_recorded_without_becoming_medical_classification(synthetic_corpus):
    chunk = synthetic_corpus.chunks[0]
    dataset = {
        "corpus_sha256": synthetic_corpus.fingerprint,
        "topics": {"answerable": {"evidence": [{"doc_id": chunk["doc_id"], "section_start": 1, "snippets": [chunk["text"]]}]}},
        "queries": [
            {"id": "answerable", "language": "en", "topic_id": "answerable", "question": "Question"},
            {"id": "unrelated", "language": "en", "topic_id": None, "question": "Unrelated question"},
        ],
    }
    # Give the gold record an unambiguous test-only source anchor.
    from medical_rag.retrieval.corpus import prepare_approved_corpus
    altered = deepcopy(synthetic_corpus.chunks)
    altered[0]["section_start"] = 99
    corpus = prepare_approved_corpus(altered, synthetic_corpus.documents)
    dataset["corpus_sha256"] = corpus.fingerprint
    dataset["topics"]["answerable"]["evidence"][0]["section_start"] = 99

    class StubRetriever:
        def search(self, question, top_k=5):
            return [{"score": .9, "chunk": altered[0], "document": {}, "embedding": {}}]

    report = evaluate_retrieval(StubRetriever(), corpus, dataset)
    assert report["out_of_kb_count"] == 1
    assert report["queries"][1]["metrics"] is None
    assert report["queries"][1]["results"]  # Dense retrieval still returned a chunk.
    assert report["metrics"]["recall"]["1"] == 1
