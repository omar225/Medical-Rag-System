"""Rebuild per-document JSON and an auditable, deterministic manifest."""

from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from typing import Any
from zipfile import BadZipFile

from lxml.etree import XMLSyntaxError

from medical_rag.ingestion.sections import DEFAULT_MAX_CHARS, chunk_document
from medical_rag.ingestion.docx_reader import UnsupportedDocxStructure, discover_approved_sources, read_docx
from medical_rag.ingestion.models import IngestionResult, SourceMetadata, StructureCorrections
from medical_rag.ingestion.structure import issue


# Bump when extraction, structure resolution, or rendering semantics change.
PIPELINE_VERSION = "docx-ingestion-2.1-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def ingest_document(
    document_path: Path,
    doc_id: str,
    source_metadata: SourceMetadata | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    corrections: StructureCorrections | None = None,
) -> IngestionResult:
    """Read one DOCX and return artifacts/issues; never write, approve, or index it."""
    if not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", doc_id):
        raise ValueError("doc_id must be a safe assigned identifier, independent of filename")
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    metadata = dict(sorted((source_metadata or {}).items()))
    if set(metadata) - {"original_filename", "title", "source_reference"}:
        raise ValueError("Unsupported source metadata field; approval is outside ingestion")
    if any(not isinstance(value, str) or not value.strip() for value in metadata.values()):
        raise ValueError("Source metadata values must be nonblank strings")
    path = Path(document_path)
    source = path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if corrections is not None and corrections.get("source_sha256") != source_sha256:
        raise ValueError("Structure correction checksum does not match the source")
    config_id = _digest({"pipeline_version": PIPELINE_VERSION, "python_docx": version("python-docx"), "max_chars": max_chars})
    normalized_corrections = None if corrections is None else {
        "source_sha256": corrections["source_sha256"],
        "rules": sorted(corrections["rules"], key=lambda rule: rule["position"]),
    }
    build_id = _digest({
        "doc_id": doc_id, "source_sha256": source_sha256, "config_id": config_id,
        "source_metadata": metadata, "filename": metadata.get("original_filename", path.name),
        "corrections": normalized_corrections,
    })
    try:
        document = read_docx(path, doc_id, source_metadata=metadata, corrections=corrections, source_bytes=source)
    except UnsupportedDocxStructure as error:
        document, issues = None, [error.issue]
    except (BadZipFile, XMLSyntaxError) as error:
        document, issues = None, [issue("invalid_docx", "error", f"DOCX package/XML could not be parsed: {type(error).__name__}")]
    else:
        issues = document["issues"]
    chunks = []
    if document is not None:
        document.update(config_id=config_id, build_id=build_id)
        chunks = chunk_document(document, max_chars)
        for chunk in chunks:
            if chunk["oversized"]:
                issues.append(issue("oversized_chunk", "review", "An indivisible paragraph or table row exceeds the character target", *(span["block_position"] for span in chunk["source_spans"])))
    return {
        "doc_id": doc_id, "source_sha256": source_sha256, "config_id": config_id,
        "build_id": build_id, "document": document, "chunks": chunks, "issues": issues,
        "requires_review": any(problem["severity"] != "info" for problem in issues),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def rebuild_knowledge_base(
    source_root: Path, output_root: Path, max_chars: int = DEFAULT_MAX_CHARS
) -> dict[str, Any]:
    """Only ingest the allowlist, replacing fixed output files rather than appending."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    destinations = [output_root / "processed", output_root / "chunks", output_root / "metadata" / "generated"]
    for destination in destinations:
        if destination.resolve().is_relative_to(source_root.resolve()) or source_root.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Generated output must not overlap the source directory")

    # Parse all approved inputs before writing outputs, so a missing/bad source
    # does not replace previously generated files with a partial parse.
    correction_path = source_root.parent / "metadata" / "structure_corrections.json"
    catalog = json.loads(correction_path.read_text(encoding="utf-8")) if correction_path.exists() else {}
    results = []
    for doc_id, path in discover_approved_sources(source_root):
        result = ingest_document(path, doc_id, max_chars=max_chars, corrections=catalog.get(doc_id))
        if result["document"] is None:
            raise ValueError(f"Cannot rebuild {doc_id}: {result['issues']}")
        results.append(result)

    records = []
    for result in results:
        document, chunks = result["document"], result["chunks"]
        doc_id = document["doc_id"]
        _write_json(output_root / "processed" / f"{doc_id}.json", document)
        _write_json(output_root / "chunks" / f"{doc_id}.json", chunks)
        records.append({
            "doc_id": doc_id,
            "filename": document["filename"],
            "title": document["title"],
            "title_source": document["title_source"],
            "source_path": f"{doc_id}/{document['filename']}",
            "source_sha256": document["source_sha256"],
            "paragraph_count": sum(block["kind"] != "table" for block in document["blocks"]),
            "heading_count": sum(block["kind"] == "heading" for block in document["blocks"]),
            "source_heading_count": sum(block.get("source_heading_level") is not None for block in document["blocks"]),
            "inferred_label_count": sum(block["kind"] == "label" for block in document["blocks"]),
            "table_count": sum(block["kind"] == "table" for block in document["blocks"]),
            "chunk_count": len(chunks),
            "oversized_chunk_ids": [chunk["chunk_id"] for chunk in chunks if chunk["oversized"]],
            "notes": document["notes"],
            "config_id": result["config_id"],
            "build_id": result["build_id"],
            "applied_corrections": document["applied_corrections"],
            "issues": result["issues"],
            "requires_review": result["requires_review"],
        })
    manifest = {
        "schema_version": 2,
        "pipeline_version": PIPELINE_VERSION,
        "max_chars": max_chars,
        "document_count": len(records),
        "table_count": sum(record["table_count"] for record in records),
        "chunk_count": sum(record["chunk_count"] for record in records),
        "documents": records,
        "issue_counts": dict(Counter(problem["severity"] for result in results for problem in result["issues"])),
    }
    _write_json(output_root / "metadata" / "generated" / "ingestion_manifest.json", manifest)
    return manifest
