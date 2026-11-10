"""Small HTTPX client for Ollama embeddings, with truncation explicitly disabled."""

import math
import re
from typing import Any

import httpx
import numpy as np


class EmbeddingError(RuntimeError):
    """The configured service did not produce a usable embedding."""


class ModelUnavailableError(EmbeddingError):
    pass


class ContextLimitError(EmbeddingError):
    pass


def canonical_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Embedding model must be a nonblank string")
    name = model.strip()
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a nonblank plain-text string")
    return text


def validate_vectors(values: Any, count: int, dimension: int | None = None) -> np.ndarray:
    if not isinstance(values, list) or len(values) != count or count == 0:
        raise EmbeddingError("Embedding vector count does not match the input count")
    width = dimension
    for row in values:
        if not isinstance(row, list) or not row:
            raise EmbeddingError("Each embedding must be a nonempty numeric vector")
        width = len(row) if width is None else width
        if len(row) != width:
            raise EmbeddingError("Embedding dimensions do not match")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
            raise EmbeddingError("Embedding values must be finite numbers, excluding booleans")
    with np.errstate(over="ignore", invalid="ignore"):
        vectors = np.asarray(values, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise EmbeddingError("Embedding values cannot be represented as finite float32 values")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise EmbeddingError("Zero-length embeddings cannot be used for cosine similarity")
    return vectors


class OllamaEmbeddings:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "bge-m3",
        timeout_seconds: float = 180.0,
        *,
        client: httpx.Client | None = None,
    ):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("Ollama endpoint must be an HTTP or HTTPS URL")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be finite and positive")
        self.model = canonical_model(model)
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds, trust_env=False)
        self._dimension: int | None = None

    def __enter__(self) -> "OllamaEmbeddings":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = self._client.request(
                method, self.base_url + path, timeout=self.timeout_seconds, **kwargs
            )
        except httpx.TimeoutException as error:
            raise EmbeddingError(f"Ollama request timed out after {self.timeout_seconds:g} seconds") from error
        except httpx.RequestError as error:
            raise EmbeddingError(f"Ollama is unavailable at {self.base_url}: {error}") from error
        try:
            data = response.json()
        except ValueError as error:
            raise EmbeddingError(f"Ollama returned invalid JSON (HTTP {response.status_code})") from error
        message = data.get("error", "request failed") if isinstance(data, dict) else "request failed"
        if response.status_code >= 400 or (isinstance(data, dict) and "error" in data):
            if response.status_code == 404 and "model" in str(message).lower():
                raise ModelUnavailableError(f"Ollama model {self.model} is unavailable. Required setup: ollama pull {self.model}")
            if any(term in str(message).lower() for term in ("context length", "context window", "input length")):
                raise ContextLimitError(f"Ollama rejected over-context input; no text was truncated: {message}")
            raise EmbeddingError(f"Ollama HTTP {response.status_code}: {message}")
        if not isinstance(data, dict):
            raise EmbeddingError("Ollama response must be a JSON object")
        return data

    def identity(self) -> dict[str, str]:
        """Resolve the installed model digest without pulling or creating a model."""
        data = self._request("GET", "/api/tags")
        models = data.get("models")
        if not isinstance(models, list):
            raise EmbeddingError("Ollama model listing has no valid models array")
        for entry in models:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise EmbeddingError("Ollama model listing contains an invalid model entry")
            if canonical_model(entry["name"]) == self.model:
                digest = entry.get("digest")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise EmbeddingError("Ollama model identity has no valid SHA-256 digest")
                return {"provider": "ollama", "model": self.model, "digest": digest}
        raise ModelUnavailableError(f"Ollama model {self.model} is unavailable. Required setup: ollama pull {self.model}")

    def embed_texts(self, texts: list[str], dimension: int | None = None) -> np.ndarray:
        if not isinstance(texts, list) or not texts:
            raise ValueError("Embedding input must be a nonempty list of strings")
        for text in texts:
            validate_text(text)
        data = self._request(
            "POST", "/api/embed",
            json={"model": self.model, "input": texts, "truncate": False},
        )
        response_model = data.get("model")
        if not isinstance(response_model, str) or canonical_model(response_model) != self.model:
            raise EmbeddingError("Ollama returned embeddings for a different or missing model identity")
        expected = dimension if dimension is not None else self._dimension
        vectors = validate_vectors(data.get("embeddings"), len(texts), expected)
        self._dimension = vectors.shape[1]
        return vectors

    def embed_question(self, question: str, dimension: int | None = None) -> np.ndarray:
        return self.embed_texts([validate_text(question)], dimension)[0]
