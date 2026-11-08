from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from medical_rag.retrieval.corpus import CorpusError, prepare_approved_corpus
from medical_rag.retrieval.embeddings.ollama import EmbeddingError
from medical_rag.retrieval.search.dense import DenseRetriever
from medical_rag.retrieval.vector_store.numpy_index import (
    DenseIndexError, StaleIndexError, WrongModelError, build_index, load_index,
)


def read_metadata(directory):
    return json.loads((directory / "index.json").read_text(encoding="utf-8"))


def write_metadata(directory, metadata):
    (directory / "index.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_build_alignment_metadata_preservation_and_cosine_ranking(tmp_path, synthetic_corpus, synthetic_provider):
    metadata = build_index(synthetic_corpus, synthetic_provider, tmp_path, batch_size=2)
    assert synthetic_provider.batches == [["Alpha fixture", "Beta fixture"], ["Gamma fixture"]]
    assert metadata["row_chunk_ids"] == ["a-1", "a-2", "b-1"]
    index = load_index(tmp_path, synthetic_corpus, "bge-m3")
    np.testing.assert_allclose(index.vectors, [[1, 0, 0], [.6, .8, 0], [0, 1, 0]], atol=1e-6)
    assert index.metadata["chunks"] == synthetic_corpus.chunks
    assert index.metadata["documents"] == synthetic_corpus.documents
    assert index.metadata["chunks"][0]["extra_provenance"] == {"preserve": "عربي English"}
    retriever = DenseRetriever(synthetic_provider, tmp_path, lambda: synthetic_corpus)
    for question in ("English question", "سؤال عربي", "Mixed سؤال"):
        results = retriever.search(question, 5)
        assert [r["chunk"]["chunk_id"] for r in results] == ["a-1", "a-2", "b-1"]
        np.testing.assert_allclose([r["score"] for r in results], [1, .6, 0], atol=1e-6)
        assert results[0]["document"] == synthetic_corpus.documents["test-A"]
        assert results[0]["embedding"] == synthetic_provider.identity()
    assert synthetic_provider.questions == ["English question", "سؤال عربي", "Mixed سؤال"]
    assert len(retriever.search("Question", 1)) == 1


def test_ties_and_input_order_are_deterministic(tmp_path, synthetic_corpus, synthetic_provider):
    synthetic_provider.values["Beta fixture"] = [1, 0, 0]
    shuffled = prepare_approved_corpus(list(reversed(synthetic_corpus.chunks)), synthetic_corpus.documents)
    assert shuffled.fingerprint == synthetic_corpus.fingerprint
    build_index(shuffled, synthetic_provider, tmp_path)
    results = DenseRetriever(synthetic_provider, tmp_path, lambda: shuffled).search("Query", 2)
    assert [r["chunk"]["chunk_id"] for r in results] == ["a-1", "a-2"]


def test_future_document_is_supported_only_when_explicitly_selected(synthetic_corpus):
    chunks = deepcopy(synthetic_corpus.chunks)
    candidate = deepcopy(chunks[0])
    candidate.update(doc_id="uploaded-tenth", chunk_id="candidate-1", build_id="e" * 64)
    with pytest.raises(CorpusError, match="not in the explicitly approved"):
        prepare_approved_corpus([*chunks, candidate], synthetic_corpus.documents)
    documents = deepcopy(synthetic_corpus.documents)
    documents["uploaded-tenth"] = {key: candidate[key] for key in ("doc_id", "title", "filename", "build_id", "source_sha256", "config_id")}
    accepted = prepare_approved_corpus([*chunks, candidate], documents)
    assert len(accepted.chunks) == 4


@pytest.mark.parametrize("field", ["build_id", "source_sha256", "config_id"])
def test_mismatched_document_build_identity_is_rejected(synthetic_corpus, field):
    chunks = deepcopy(synthetic_corpus.chunks)
    chunks[0][field] = "f" * 64
    with pytest.raises(CorpusError, match=field):
        prepare_approved_corpus(chunks, synthetic_corpus.documents)


def test_missing_index_is_clear(tmp_path, synthetic_corpus):
    with pytest.raises(DenseIndexError, match="Missing or invalid index"):
        load_index(tmp_path, synthetic_corpus, "bge-m3")


@pytest.mark.parametrize("payload", ["not JSON", "[]", "{}", '{"schema_version":99}'])
def test_invalid_index_manifest(tmp_path, synthetic_corpus, payload):
    (tmp_path / "index.json").write_text(payload, encoding="utf-8")
    with pytest.raises(DenseIndexError):
        load_index(tmp_path, synthetic_corpus, "bge-m3")


def test_stale_build_replacement_and_document_deletion(tmp_path, synthetic_corpus, synthetic_provider):
    build_index(synthetic_corpus, synthetic_provider, tmp_path)
    chunks, documents = deepcopy(synthetic_corpus.chunks), deepcopy(synthetic_corpus.documents)
    documents["test-A"]["build_id"] = "f" * 64
    for c in chunks:
        if c["doc_id"] == "test-A":
            c["build_id"] = "f" * 64
    changed = prepare_approved_corpus(chunks, documents)
    with pytest.raises(StaleIndexError):
        load_index(tmp_path, changed, "bge-m3")
    deleted = prepare_approved_corpus([c for c in chunks if c["doc_id"] != "test-B"], {"test-A": documents["test-A"]})
    with pytest.raises(StaleIndexError):
        DenseRetriever(synthetic_provider, tmp_path, lambda: deleted).search("Query")


def test_wrong_model_name_and_same_name_changed_digest(tmp_path, synthetic_corpus, synthetic_provider):
    build_index(synthetic_corpus, synthetic_provider, tmp_path)
    with pytest.raises(WrongModelError):
        load_index(tmp_path, synthetic_corpus, "another-model")
    synthetic_provider.digest = "e" * 64
    with pytest.raises(WrongModelError, match="digest"):
        DenseRetriever(synthetic_provider, tmp_path, lambda: synthetic_corpus).search("Query")


def test_model_change_during_build_does_not_publish(tmp_path, synthetic_corpus, synthetic_provider, monkeypatch):
    identity = synthetic_provider.identity()
    identities = iter([identity, {**identity, "digest": "e" * 64}])
    monkeypatch.setattr(synthetic_provider, "identity", lambda: next(identities))
    with pytest.raises(WrongModelError):
        build_index(synthetic_corpus, synthetic_provider, tmp_path)
    assert not (tmp_path / "index.json").exists()


def test_bad_provider_vector_count_does_not_publish(tmp_path, synthetic_corpus, synthetic_provider, monkeypatch):
    monkeypatch.setattr(synthetic_provider, "embed_texts", lambda *args, **kwargs: np.asarray([[1, 0, 0]], dtype=np.float32))
    with pytest.raises(EmbeddingError, match="count"):
        build_index(synthetic_corpus, synthetic_provider, tmp_path)
    assert not (tmp_path / "index.json").exists()


@pytest.mark.parametrize("corruption", ["chunk_order", "row_ids", "text", "count", "dimension", "path", "digest", "missing_vectors", "vector_bytes"])
def test_index_corruption_fails_clearly(tmp_path, synthetic_corpus, synthetic_provider, corruption):
    build_index(synthetic_corpus, synthetic_provider, tmp_path)
    metadata = read_metadata(tmp_path)
    if corruption == "chunk_order":
        metadata["chunks"].reverse()
    elif corruption == "row_ids":
        metadata["row_chunk_ids"].reverse()
    elif corruption == "text":
        metadata["chunks"][0]["text"] = "Changed source text"
    elif corruption == "count":
        metadata["chunk_count"] = 4
    elif corruption == "dimension":
        metadata["dimension"] = 4
    elif corruption == "path":
        metadata["vectors_file"] = "../../escape.npy"
    elif corruption == "digest":
        metadata["embedding"]["digest"] = "not-a-digest"
    elif corruption == "missing_vectors":
        (tmp_path / metadata["vectors_file"]).unlink()
    elif corruption == "vector_bytes":
        (tmp_path / metadata["vectors_file"]).write_bytes(b"invalid vector file")
    write_metadata(tmp_path, metadata)
    with pytest.raises(DenseIndexError):
        load_index(tmp_path, synthetic_corpus, "bge-m3")


@pytest.mark.parametrize("matrix", [
    np.asarray([[1, 0, 0]] * 3, dtype=np.float64),
    np.asarray([[0, 0, 0]] * 3, dtype=np.float32),
    np.asarray([[float("nan"), 0, 0]] * 3, dtype=np.float32),
])
def test_numerical_vector_file_validation(tmp_path, synthetic_corpus, synthetic_provider, matrix):
    build_index(synthetic_corpus, synthetic_provider, tmp_path)
    metadata = read_metadata(tmp_path)
    temporary = tmp_path / "changed.npy"
    np.save(temporary, matrix, allow_pickle=False)
    checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
    filename = f"vectors-{checksum}.npy"
    temporary.replace(tmp_path / filename)
    metadata.update(vectors_file=filename, vectors_sha256=checksum)
    write_metadata(tmp_path, metadata)
    with pytest.raises(DenseIndexError):
        load_index(tmp_path, synthetic_corpus, "bge-m3")


@pytest.mark.parametrize("query,top_k", [("", 5), (" \n", 5), (123, 5), ("Query", 0), ("Query", -1), ("Query", True)])
def test_invalid_queries_fail_before_loading_or_embedding(tmp_path, synthetic_provider, query, top_k):
    retriever = DenseRetriever(synthetic_provider, tmp_path, lambda: pytest.fail("Invalid query must not read the corpus"))
    with pytest.raises(ValueError):
        retriever.search(query, top_k)
