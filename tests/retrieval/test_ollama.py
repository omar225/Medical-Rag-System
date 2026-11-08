import json

import httpx
import numpy as np
import pytest

from medical_rag.retrieval.embeddings.ollama import (
    ContextLimitError, EmbeddingError, ModelUnavailableError, OllamaEmbeddings,
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
