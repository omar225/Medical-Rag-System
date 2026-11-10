"""A small exact index: immutable float32 NPY data and an atomic JSON manifest."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np

from medical_rag.retrieval.corpus import ApprovedCorpus, prepare_approved_corpus
from medical_rag.retrieval.embeddings import OllamaEmbeddings, canonical_model, validate_vectors


INPUT_VERSION = "raw-chunk-text-v1"


class DenseIndexError(RuntimeError):
    pass


class StaleIndexError(DenseIndexError):
    pass


class WrongModelError(DenseIndexError):
    pass


@dataclass(frozen=True)
class DenseIndex:
    vectors: np.ndarray
    metadata: dict[str, Any]


def check_model(index: DenseIndex, identity: dict[str, str]) -> None:
    if index.metadata["embedding"] != identity:
        raise WrongModelError("Index embedding model name/digest differs from Ollama; rebuild the index")


def build_index(
    corpus: ApprovedCorpus,
    provider: OllamaEmbeddings,
    index_dir: Path,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Embed only the explicitly approved input snapshot, in canonical row order."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    started = perf_counter()
    checked = prepare_approved_corpus(corpus.chunks, corpus.documents)
    if checked.fingerprint != corpus.fingerprint:
        raise DenseIndexError("Approved corpus was modified after selection")
    identity = provider.identity()
    batches = []
    dimension = None
    embedding_seconds = 0.0
    for offset in range(0, len(checked.chunks), batch_size):
        texts = [chunk["text"] for chunk in checked.chunks[offset:offset + batch_size]]
        batch_started = perf_counter()
        vectors = provider.embed_texts(texts, dimension=dimension)
        embedding_seconds += perf_counter() - batch_started
        # Validate again at the storage boundary, including for alternative callers/providers.
        vectors = validate_vectors(vectors.tolist(), len(texts), dimension)
        dimension = vectors.shape[1]
        batches.append(vectors)
    if provider.identity() != identity:
        raise WrongModelError("Ollama model identity changed while indexing; build was not published")
    matrix = np.concatenate(batches)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    matrix = (matrix / norms).astype(np.float32)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary = index_dir / f".vectors-{uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            np.save(stream, matrix, allow_pickle=False)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        filename = f"vectors-{checksum}.npy"
        temporary.replace(index_dir / filename)
        metadata = {
            "schema_version": 1,
            "embedding_input_version": INPUT_VERSION,
            "embedding": identity,
            "dimension": dimension,
            "chunk_count": len(checked.chunks),
            "document_count": len(checked.documents),
            "documents": checked.documents,
            "corpus_sha256": checked.fingerprint,
            "row_chunk_ids": [chunk["chunk_id"] for chunk in checked.chunks],
            "vectors_file": filename,
            "vectors_sha256": checksum,
            "chunks": checked.chunks,
            "batch_size": batch_size,
            "embedding_seconds": embedding_seconds,
            "indexing_seconds": perf_counter() - started,
        }
        manifest_tmp = index_dir / f".index-{uuid4().hex}.tmp"
        try:
            manifest_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            manifest_tmp.replace(index_dir / "index.json")
        finally:
            manifest_tmp.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def load_index(index_dir: Path, corpus: ApprovedCorpus, expected_model: str) -> DenseIndex:
    """Reject stale sources, different models, damaged vectors, or row misalignment."""
    checked = prepare_approved_corpus(corpus.chunks, corpus.documents)
    if checked.fingerprint != corpus.fingerprint:
        raise StaleIndexError("Approved corpus was modified after selection")
    path = Path(index_dir) / "index.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DenseIndexError(f"Missing or invalid index at {path}; run the retrieval build command") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise DenseIndexError("Unsupported or invalid index manifest")
    if metadata.get("embedding_input_version") != INPUT_VERSION:
        raise StaleIndexError("Index text-input format changed; rebuild the index")
    identity = metadata.get("embedding")
    if not isinstance(identity, dict) or identity.get("model") != canonical_model(expected_model):
        raise WrongModelError("Configured embedding model differs from the index model")
    if identity.get("provider") != "ollama" or not isinstance(identity.get("digest"), str) or not re.fullmatch(r"[0-9a-f]{64}", identity["digest"]):
        raise DenseIndexError("Index has invalid embedding provider/digest metadata")
    if metadata.get("corpus_sha256") != corpus.fingerprint:
        raise StaleIndexError("Index is stale: approved chunks, metadata, or document builds changed; rebuild it")
    try:
        stored = prepare_approved_corpus(metadata["chunks"], metadata["documents"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise DenseIndexError("Index contains invalid source/chunk metadata") from error
    if stored.fingerprint != corpus.fingerprint:
        raise DenseIndexError("Index chunk metadata does not match its source fingerprint")
    ids = [chunk["chunk_id"] for chunk in stored.chunks]
    if metadata.get("row_chunk_ids") != ids or metadata["chunks"] != stored.chunks:
        raise DenseIndexError("Vector row order and chunk metadata are not aligned")
    if metadata.get("chunk_count") != len(ids) or metadata.get("document_count") != len(stored.documents):
        raise DenseIndexError("Index count metadata is inconsistent")
    checksum = metadata.get("vectors_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum) or metadata.get("vectors_file") != f"vectors-{checksum}.npy":
        raise DenseIndexError("Invalid vector filename/checksum in index manifest")
    try:
        vector_path = Path(index_dir) / metadata["vectors_file"]
        if hashlib.sha256(vector_path.read_bytes()).hexdigest() != checksum:
            raise DenseIndexError("Vector file checksum mismatch")
        vectors = np.load(vector_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise DenseIndexError("Missing or invalid numerical vector data") from error
    dimension = metadata.get("dimension")
    if type(dimension) is not int or dimension <= 0 or not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32 or vectors.shape != (len(ids), dimension):
        raise DenseIndexError("Vector dimensions/count/dtype do not match chunk metadata")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1.0, atol=1e-5):
        raise DenseIndexError("Index contains nonfinite, zero, or unnormalized vectors")
    return DenseIndex(vectors, metadata)
