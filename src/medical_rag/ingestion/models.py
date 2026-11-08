"""JSON-compatible records shared by the reader, chunker, and rebuild command."""

from typing import Literal, NotRequired, TypedDict


class SourceMetadata(TypedDict, total=False):
    original_filename: str
    title: str
    source_reference: str


class StructuralIssue(TypedDict):
    code: str
    severity: Literal["info", "review", "error"]
    message: str
    block_positions: list[int]


class StructureRule(TypedDict):
    position: int
    reason: str
    heading_level: NotRequired[int]
    parent_position: NotRequired[int | None]
    new_section: NotRequired[bool]
    topic_positions: NotRequired[list[int]]


class StructureCorrections(TypedDict):
    source_sha256: str
    rules: list[StructureRule]


class SourceBlock(TypedDict):
    position: int
    kind: Literal["paragraph", "heading", "label", "table"]
    text: str
    section_path: list[str]
    section_start: int
    heading_level: NotRequired[int]
    source_heading_level: NotRequired[int | None]
    heading_source: NotRequired[str]
    heading_candidate: NotRequired[bool]
    headers: NotRequired[list[list[str]]]
    rows: NotRequired[list[list[str]]]
    header_source: NotRequired[str]
    context_topics: NotRequired[list[str]]
    scope_uncertain: NotRequired[bool]
    correction_positions: NotRequired[list[int]]


class ParsedDocument(TypedDict):
    doc_id: str
    title: str
    title_source: str
    filename: str
    source_sha256: str
    blocks: list[SourceBlock]
    notes: list[str]
    issues: list[StructuralIssue]
    source_metadata: SourceMetadata
    applied_corrections: list[StructureRule]
    config_id: NotRequired[str]
    build_id: NotRequired[str]


class SourceSpan(TypedDict):
    block_position: int
    row_start: NotRequired[int]
    row_end: NotRequired[int]


class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    section_path: list[str]
    section_start: int
    filename: str
    text: str
    source_spans: list[SourceSpan]
    char_count: int
    oversized: bool
    context_topics: list[str]
    scope_uncertain: bool
    correction_positions: list[int]
    source_sha256: str
    config_id: NotRequired[str]
    build_id: NotRequired[str]


class IngestionResult(TypedDict):
    doc_id: str
    source_sha256: str
    config_id: str
    build_id: str
    document: ParsedDocument | None
    chunks: list[Chunk]
    issues: list[StructuralIssue]
    requires_review: bool
