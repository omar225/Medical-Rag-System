"""Pack source sections without slicing a paragraph, cell, or table row."""

from itertools import groupby
from typing import TypedDict

from medical_rag.ingestion.loaders.docx_reader import render_table
from medical_rag.ingestion.models import Chunk, ParsedDocument, SourceBlock, SourceSpan


DEFAULT_MAX_CHARS = 2400


class Fragment(TypedDict):
    text: str
    section_path: list[str]
    source_spans: list[SourceSpan]
    context_topics: list[str]


def _render(fragments: list[Fragment]) -> str:
    """Repeat only the source headings needed to contextualize this chunk."""
    parts = []
    previous: list[str] = []
    for fragment in fragments:
        path = [topic for topic in fragment["context_topics"] if topic not in fragment["section_path"]] + fragment["section_path"]
        common = 0
        for left, right in zip(previous, path):
            if left != right:
                break
            common += 1
        source_lines = fragment["text"].splitlines()
        parts.extend(heading for heading in path[common:] if heading not in source_lines)
        parts.append(fragment["text"])
        previous = path
    return "\n\n".join(parts)


def _table_fragments(block: SourceBlock, max_chars: int) -> list[Fragment]:
    headers, rows = block["headers"], block["rows"]
    fragments = []
    pending: list[list[str]] = []
    start = len(headers) + 1

    def make_fragment(batch: list[list[str]], row_start: int) -> Fragment:
        span: SourceSpan = {"block_position": block["position"]}
        if batch:
            span.update(row_start=row_start, row_end=row_start + len(batch) - 1)
        return {
            "text": render_table(headers, batch),
            "section_path": block["section_path"],
            "source_spans": [span],
            "context_topics": block.get("context_topics", []),
        }

    for row in rows:
        candidate = make_fragment([*pending, row], start)
        if pending and len(_render([candidate])) > max_chars:
            fragments.append(make_fragment(pending, start))
            start += len(pending)
            pending = []
        pending.append(row)
    if pending or not rows:
        fragments.append(make_fragment(pending, start))
    return fragments


def _section_fragments(blocks: list[SourceBlock], max_chars: int) -> list[Fragment]:
    fragments: list[Fragment] = []
    pending: list[SourceBlock] = []

    def paragraph_fragment(batch: list[SourceBlock]) -> Fragment:
        return {
            "text": "\n".join(block["text"] for block in batch),
            "section_path": batch[0]["section_path"],
            "source_spans": [{"block_position": block["position"]} for block in batch],
            "context_topics": batch[0].get("context_topics", []),
        }

    def flush() -> None:
        if pending:
            fragments.append(paragraph_fragment(pending))
            pending.clear()

    for block in blocks:
        if not block["text"].strip():
            continue
        if block["kind"] == "table":
            flush()
            fragments.extend(_table_fragments(block, max_chars))
        else:
            if pending and (
                block["kind"] in {"heading", "label"}
                or len(_render([paragraph_fragment([*pending, block])])) > max_chars
            ):
                flush()
            pending.append(block)
    flush()
    return fragments


def chunk_document(document: ParsedDocument, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Prefer whole major sections; use paragraph/row boundaries when too long."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: list[Chunk] = []

    sections = [
        (start, list(blocks))
        for start, blocks in groupby(document["blocks"], key=lambda block: block["section_start"])
    ]
    for index, (section_start, blocks) in enumerate(sections):
        # Topic titles sometimes use the same heading level as the next class.
        # Carry a heading-only group forward without inventing a parent category.
        if all(block["kind"] == "heading" for block in blocks) and index + 1 < len(sections):
            sections[index + 1][1][:0] = blocks
            continue
        main_block = next((block for block in blocks if block["position"] == section_start), blocks[0])
        section_path = main_block["section_path"]
        pending: list[Fragment] = []

        def emit() -> None:
            if not pending:
                return
            text = _render(pending)
            chunks.append({
                "chunk_id": f"{document['doc_id']}-" + (f"{document['build_id']}-" if "build_id" in document else "") + f"{len(chunks) + 1:04d}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "section": " > ".join(section_path),
                "section_path": section_path,
                "section_start": section_start,
                "filename": document["filename"],
                "text": text,
                "source_spans": [span for fragment in pending for span in fragment["source_spans"]],
                "char_count": len(text),
                "oversized": len(text) > max_chars,
                "context_topics": main_block.get("context_topics", []),
                "scope_uncertain": any(block.get("scope_uncertain", False) for block in blocks),
                "correction_positions": sorted({p for block in blocks for p in block.get("correction_positions", [])}),
                "source_sha256": document["source_sha256"],
                **{key: document[key] for key in ("config_id", "build_id") if key in document},
            })
            pending.clear()

        for fragment in _section_fragments(blocks, max_chars):
            if pending and len(_render([*pending, fragment])) > max_chars:
                emit()
            pending.append(fragment)
        emit()
    return chunks
