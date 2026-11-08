"""Read-only approved-source smoke test; generated outputs use pytest temp space."""

import hashlib
import json
from pathlib import Path

from docx import Document

from medical_rag.ingestion.loaders.docx_reader import APPROVED_SOURCES, discover_approved_sources
from medical_rag.ingestion.processing.pipeline import rebuild_knowledge_base


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "source"
EXPECTED_TABLES = {"KB-01": 19, "KB-02": 19, "KB-03": 15, "KB-04": 9, "KB-05": 8, "KB-06": 16, "KB-07": 22, "KB-08": 30, "KB-09": 15}


def test_nine_approved_sources_preserve_all_body_text_and_table_rows(tmp_path):
    sources = discover_approved_sources(SOURCE_ROOT)
    before = {doc_id: hashlib.sha256(path.read_bytes()).hexdigest() for doc_id, path in sources}

    manifest = rebuild_knowledge_base(SOURCE_ROOT, tmp_path)

    assert manifest["document_count"] == 9
    assert manifest["table_count"] == 153
    assert {record["doc_id"]: record["table_count"] for record in manifest["documents"]} == EXPECTED_TABLES
    chunk_ids = []
    for doc_id, path in sources:
        parsed = json.loads((tmp_path / "processed" / f"{doc_id}.json").read_text(encoding="utf-8"))
        chunks = json.loads((tmp_path / "chunks" / f"{doc_id}.json").read_text(encoding="utf-8"))
        original = Document(path)
        assert parsed["doc_id"] == doc_id
        assert parsed["filename"] == APPROVED_SOURCES[doc_id]
        assert [block["text"] for block in parsed["blocks"] if block["kind"] != "table"] == [p.text for p in original.paragraphs if p.text.strip()]
        tables = [block for block in parsed["blocks"] if block["kind"] == "table"]
        assert len(tables) == len(original.tables)
        for extracted, table in zip(tables, original.tables):
            assert [*extracted["headers"], *extracted["rows"]] == [[cell.text for cell in row.cells] for row in table.rows]
        assert chunks
        for block in parsed["blocks"]:
            locations = [(chunk, span) for chunk in chunks for span in chunk["source_spans"] if span["block_position"] == block["position"]]
            assert locations, f"Lost block {doc_id}:{block['position']}"
            if block["kind"] == "table":
                covered = []
                for chunk, span in locations:
                    assert "row_start" in span
                    covered.extend(range(span["row_start"], span["row_end"] + 1))
                    for header in block["headers"]:
                        assert " | ".join(header) in chunk["text"]
                    for row_index in range(span["row_start"], span["row_end"] + 1):
                        source_row = block["rows"][row_index - len(block["headers"]) - 1]
                        assert " | ".join(source_row) in chunk["text"]
                assert covered == list(range(len(block["headers"]) + 1, len(block["headers"]) + len(block["rows"]) + 1))
            else:
                assert len(locations) == 1
                assert block["text"] in locations[0][0]["text"]
        chunk_ids.extend(chunk["chunk_id"] for chunk in chunks)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before[doc_id] == parsed["source_sha256"]
    assert len(chunk_ids) == len(set(chunk_ids)) == manifest["chunk_count"]

    first_outputs = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert rebuild_knowledge_base(SOURCE_ROOT, tmp_path) == manifest
    assert first_outputs == {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*.json")}


def test_reviewed_contexts_and_summary_scopes_are_correct(tmp_path):
    manifest = rebuild_knowledge_base(SOURCE_ROOT, tmp_path)
    assert all(record["requires_review"] == (record["doc_id"] == "KB-04") for record in manifest["documents"])
    source_gap = next(record for record in manifest["documents"] if record["doc_id"] == "KB-04")
    assert [problem["code"] for problem in source_gap["issues"] if problem["severity"] != "info"] == ["blank_table_value"]
    assert sum(len(record["applied_corrections"]) for record in manifest["documents"]) == 60
    documents = {
        doc_id: json.loads((tmp_path / "processed" / f"{doc_id}.json").read_text(encoding="utf-8"))
        for doc_id in APPROVED_SOURCES
    }
    chunks = {
        doc_id: json.loads((tmp_path / "chunks" / f"{doc_id}.json").read_text(encoding="utf-8"))
        for doc_id in APPROVED_SOURCES
    }

    def block(doc_id, position):
        return next(b for b in documents[doc_id]["blocks"] if b["position"] == position)

    def chunk(doc_id, position):
        return next(c for c in chunks[doc_id] if any(s["block_position"] == position for s in c["source_spans"]))

    expected_roots = {
        "KB-01": {7: 1, 21: 1, 119: 118, 133: 118},
        "KB-03": {16: 1, 132: 120},
        "KB-07": {2: 1, 15: 1, 119: 1},
        "KB-08": {15: 1, 130: 129, 177: 129, 214: 129},
    }
    for doc_id, cases in expected_roots.items():
        for position, root in cases.items():
            assert block(doc_id, position)["section_path"][0] == block(doc_id, root)["text"]
            assert chunk(doc_id, position)["section_path"][0] == block(doc_id, root)["text"]
            assert block(doc_id, root)["text"] in chunk(doc_id, position)["text"]

    for doc_id, positions, main_class in (
        ("KB-02", [106, 114, 115], 96),
        ("KB-02", [130, 134], 124),
        ("KB-04", [67, 75, 76], 57),
    ):
        for position in positions:
            assert block(doc_id, position)["section_path"][:-1] == block(doc_id, main_class)["section_path"]

    classification = chunk("KB-02", 121)
    assert classification["section_path"] == [block("KB-02", 1)["text"]]
    assert classification["section_start"] == 121
    assert all(span["block_position"] != 121 for span in chunk("KB-02", 96)["source_spans"])
    assert "H. pylori" not in classification["section"]

    for doc_id, summary, table, topics in (("KB-02", 193, 194, [1, 123]), ("KB-09", 223, 224, [1, 115])):
        candidate = chunk(doc_id, table)
        assert candidate["section_path"] == [block(doc_id, summary)["text"]]
        assert candidate["context_topics"] == [block(doc_id, position)["text"] for position in topics]
        for position in topics:
            assert block(doc_id, position)["text"] in candidate["text"]

    assert block("KB-08", 171)["section_path"][-1] == "Side Effects"
    assert block("KB-03", 106)["section_path"][-1] == block("KB-03", 105)["text"]
    assert all("build_id" in c and c["source_sha256"] == documents[doc_id]["source_sha256"] for doc_id in chunks for c in chunks[doc_id])
