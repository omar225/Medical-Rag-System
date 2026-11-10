"""Embedding client and NumPy vector index (synthetic vectors, no Ollama)."""

from copy import deepcopy
import hashlib
import json

import httpx
import numpy as np
import pytest

from medical_rag.retrieval.corpus import CorpusError, prepare_approved_corpus
from medical_rag.retrieval.dense import DenseRetriever
from medical_rag.retrieval.embeddings import (
    ContextLimitError, EmbeddingError, ModelUnavailableError, OllamaEmbeddings,
)
from medical_rag.retrieval.embeddings import EmbeddingError
from medical_rag.retrieval.numpy_index import (
    DenseIndexError, StaleIndexError, WrongModelError, build_index, load_index,
)


def response(payload, status=200):
    return httpx.Response(status, content=json.dumps(payload).encode("utf-8"), headers={"content-type": "application/json"})


def make_provider(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaEmbeddings("http://mock-ollama", client=client), client


def test_batch_and_single_question_preserve_unicode_and_disable_truncation():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/api/embed"
        return response({"model": "bge-m3", "embeddings": [[1, 2, 3] for _ in payload["input"]]})

    provider, client = make_provider(handler)
    with client:
        vectors = provider.embed_texts(["English source", "مصدر عربي English"])
        question = provider.embed_question("إيه الأعراض الجانبية؟", dimension=3)
    assert vectors.shape == (2, 3) and vectors.dtype == np.float32
    assert question.shape == (3,)
    assert requests == [
        {"model": "bge-m3:latest", "input": ["English source", "مصدر عربي English"], "truncate": False},
        {"model": "bge-m3:latest", "input": ["إيه الأعراض الجانبية؟"], "truncate": False},
    ]


def test_installed_model_digest_identity():
    provider, client = make_provider(lambda request: response({"models": [{"name": "bge-m3:latest", "digest": "a" * 64}]}))
    with client:
        assert provider.identity() == {"provider": "ollama", "model": "bge-m3:latest", "digest": "a" * 64}


@pytest.mark.parametrize("payload", [{"models": []}, {"models": [{"name": "another-model", "digest": "a" * 64}]}])
def test_missing_model_reports_setup_without_downloading(payload):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return response(payload)

    provider, client = make_provider(handler)
    with client, pytest.raises(ModelUnavailableError, match="ollama pull bge-m3:latest"):
        provider.identity()
    assert paths == ["/api/tags"]


@pytest.mark.parametrize("payload", [{}, {"models": "bad"}, {"models": [None]}, {"models": [{"name": "bge-m3", "digest": "bad"}]}])
def test_invalid_model_listing_is_rejected(payload):
    provider, client = make_provider(lambda request: response(payload))
    with client, pytest.raises(EmbeddingError):
        provider.identity()


@pytest.mark.parametrize("kind", ["timeout", "unavailable"])
def test_network_failures_are_clear(kind):
    def handler(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("mock timeout", request=request)
        raise httpx.ConnectError("mock refused connection", request=request)

    provider, client = make_provider(handler)
    with client, pytest.raises(EmbeddingError, match="timed out" if kind == "timeout" else "unavailable"):
        provider.embed_question("Query")


@pytest.mark.parametrize("status,error,exception", [
    (404, "model bge-m3 not found", ModelUnavailableError),
    (500, "runner failed", EmbeddingError),
    (400, "input length exceeds maximum context length", ContextLimitError),
])
def test_failed_requests_and_context_limit(status, error, exception):
    text = "long source " * 10000

    def handler(request):
        payload = json.loads(request.content)
        assert payload["truncate"] is False and payload["input"] == [text]
        return response({"error": error}, status)

    provider, client = make_provider(handler)
    with client, pytest.raises(exception):
        provider.embed_question(text)


@pytest.mark.parametrize("embeddings", [
    [], [[1, 2], [3, 4]], [[]], [[1, 2], [3]],
    [[True, 1]], [["1", 2]], [[None, 2]], [[float("nan"), 2]],
    [[float("inf"), 2]], [[1e300, 1]], [[0, 0]], {"vector": [1, 2]},
])
def test_invalid_vectors_are_rejected(embeddings):
    provider, client = make_provider(lambda request: response({"model": "bge-m3", "embeddings": embeddings}))
    with client, pytest.raises(EmbeddingError):
        provider.embed_question("Query")


def test_dimensions_are_checked_across_batches_and_against_the_index():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return response({"model": "bge-m3", "embeddings": [[1, 2] if calls == 1 else [1, 2, 3]]})

    provider, client = make_provider(handler)
    with client:
        provider.embed_question("First")
        with pytest.raises(EmbeddingError, match="dimensions"):
            provider.embed_question("Second")
    provider, client = make_provider(lambda request: response({"model": "bge-m3", "embeddings": [[1, 2]]}))
    with client, pytest.raises(EmbeddingError, match="dimensions"):
        provider.embed_question("Query", dimension=3)


@pytest.mark.parametrize("payload", [{"embeddings": [[1, 2]]}, {"model": "different", "embeddings": [[1, 2]]}, []])
def test_invalid_embedding_response_model(payload):
    provider, client = make_provider(lambda request: response(payload))
    with client, pytest.raises(EmbeddingError):
        provider.embed_question("Query")


def test_non_json_response_is_rejected():
    provider, client = make_provider(lambda request: httpx.Response(200, text="not JSON"))
    with client, pytest.raises(EmbeddingError, match="invalid JSON"):
        provider.embed_question("Query")


@pytest.mark.parametrize("text", [None, 123, "", " \n\t ", ["query"]])
def test_invalid_input_fails_before_http(text):
    provider, client = make_provider(lambda request: pytest.fail("Invalid input must not call Ollama"))
    with client, pytest.raises(ValueError):
        provider.embed_question(text)


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
