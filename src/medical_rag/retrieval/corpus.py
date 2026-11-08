"""Explicit corpus selection and provenance checks, separate from ingestion."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from medical_rag.ingestion.loaders.docx_reader import APPROVED_SOURCES
from medical_rag.ingestion.processing.pipeline import PIPELINE_VERSION


class CorpusError(RuntimeError):
    pass


def canonical_hash(value: Any) -> str:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ApprovedCorpus:
    chunks: list[dict[str, Any]]
    documents: dict[str, dict[str, Any]]
    fingerprint: str


def prepare_approved_corpus(
    chunks: list[dict[str, Any]], approved_documents: dict[str, dict[str, Any]]
) -> ApprovedCorpus:
    """Validate a caller-selected approved set; parsing success is never approval."""
    if not isinstance(chunks, list) or not chunks or not isinstance(approved_documents, dict) or not approved_documents:
        raise CorpusError("An explicitly selected, nonempty approved corpus is required")
    if any(not isinstance(chunk, dict) for chunk in chunks) or any(not isinstance(record, dict) for record in approved_documents.values()):
        raise CorpusError("Chunks and approved document records must be JSON objects")
    ordered = sorted(deepcopy(chunks), key=lambda chunk: (chunk.get("doc_id", ""), chunk.get("chunk_id", "")))
    documents = deepcopy(approved_documents)
    counts: Counter = Counter()
    ids = set()
    for chunk in ordered:
        if not isinstance(chunk, dict):
            raise CorpusError("Each chunk must be a JSON object")
        for field in ("chunk_id", "doc_id", "title", "section", "text", "filename", "build_id", "source_sha256", "config_id"):
            if not isinstance(chunk.get(field), str) or not chunk[field].strip():
                raise CorpusError(f"Chunk has missing or invalid {field}")
        doc_id = chunk["doc_id"]
        if doc_id not in documents:
            raise CorpusError(f"Document {doc_id} is not in the explicitly approved corpus")
        record = documents[doc_id]
        for field in ("build_id", "source_sha256", "config_id"):
            if not re.fullmatch(r"[0-9a-f]{64}", chunk[field]):
                raise CorpusError(f"Invalid {field} identity for document {doc_id}")
        for field in ("build_id", "source_sha256", "config_id", "title", "filename"):
            if chunk[field] != record.get(field):
                raise CorpusError(f"Stale or mismatched {field} for document {doc_id}; rebuild ingestion artifacts")
        path = chunk.get("section_path")
        if not isinstance(path, list) or not all(isinstance(part, str) and part.strip() for part in path) or " > ".join(path) != chunk["section"]:
            raise CorpusError("Chunk section path does not match its section")
        spans = chunk.get("source_spans")
        if not isinstance(spans, list) or not spans or any(
            not isinstance(span, dict) or type(span.get("block_position")) is not int or span["block_position"] <= 0
            for span in spans
        ):
            raise CorpusError("Chunk has invalid source spans")
        for span in spans:
            if "row_start" in span or "row_end" in span:
                if type(span.get("row_start")) is not int or type(span.get("row_end")) is not int or not 0 < span["row_start"] <= span["row_end"]:
                    raise CorpusError("Chunk has an invalid table row range")
        if chunk.get("scope_uncertain", False):
            raise CorpusError("Unresolved section scope requires review before indexing")
        if chunk["chunk_id"] in ids:
            raise CorpusError("Duplicate chunk ID in approved corpus")
        ids.add(chunk["chunk_id"])
        counts[doc_id] += 1
    if set(counts) != set(documents):
        raise CorpusError("Approved document set and chunk document set differ")
    for doc_id, record in documents.items():
        if "chunk_count" in record and record["chunk_count"] != counts[doc_id]:
            raise CorpusError(f"Chunk count differs from the approved build for {doc_id}")
    fingerprint = canonical_hash({"documents": documents, "chunks": ordered})
    return ApprovedCorpus(ordered, documents, fingerprint)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CorpusError(f"Missing or invalid ingestion artifact: {path}. Run scripts/ingestion/rebuild.py") from error


def load_bootstrap_corpus(knowledge_base: Path) -> ApprovedCorpus:
    """Select exactly the approved-nine bootstrap; extra files never join by discovery."""
    manifest = _read_json(knowledge_base / "metadata/generated/ingestion_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2 or manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise CorpusError("Ingestion manifest is stale or unsupported; rebuild ingestion artifacts")
    records = manifest.get("documents")
    if not isinstance(records, list) or any(not isinstance(record, dict) or not isinstance(record.get("doc_id"), str) for record in records):
        raise CorpusError("Invalid ingestion manifest document list")
    documents = {record.get("doc_id"): record for record in records}
    if len(records) != len(documents) or set(documents) != set(APPROVED_SOURCES):
        raise CorpusError("Bootstrap indexing requires exactly the approved source catalog")
    corrections = _read_json(knowledge_base / "metadata/structure_corrections.json")
    if not isinstance(corrections, dict):
        raise CorpusError("Invalid bootstrap structure correction catalog")
    chunks = []
    for doc_id, filename in APPROVED_SOURCES.items():
        record = documents[doc_id]
        if record.get("filename") != filename or record.get("source_path") != f"{doc_id}/{filename}":
            raise CorpusError("Bootstrap source filename or path differs from the approved catalog")
        source = knowledge_base / "source" / doc_id / filename
        try:
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as error:
            raise CorpusError(f"Approved source is missing: {source}") from error
        if source_hash != record.get("source_sha256"):
            raise CorpusError(f"Source {doc_id} changed; rebuild ingestion artifacts before indexing/search")
        document = _read_json(knowledge_base / "processed" / f"{doc_id}.json")
        if not isinstance(document, dict) or any(document.get(key) != record.get(key) for key in (
            "doc_id", "build_id", "source_sha256", "config_id", "title", "filename", "applied_corrections"
        )):
            raise CorpusError(f"Processed source/build identity mismatch for {doc_id}")
        reviewed = corrections.get(doc_id, {})
        if not isinstance(reviewed, dict) or reviewed.get("source_sha256") != source_hash or reviewed.get("rules") != record.get("applied_corrections"):
            raise CorpusError(f"Reviewed corrections changed for {doc_id}; rebuild ingestion artifacts")
        document_chunks = _read_json(knowledge_base / "chunks" / f"{doc_id}.json")
        if not isinstance(document_chunks, list) or any(not isinstance(c, dict) or c.get("doc_id") != doc_id for c in document_chunks):
            raise CorpusError(f"Chunk file does not belong to the approved document {doc_id}")
        chunks.extend(document_chunks)
    if manifest.get("document_count") != len(documents) or manifest.get("chunk_count") != len(chunks):
        raise CorpusError("Ingestion manifest totals do not match the selected chunks")
    return prepare_approved_corpus(chunks, documents)
