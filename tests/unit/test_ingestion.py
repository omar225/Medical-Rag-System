"""Synthetic ingestion fixtures contain no clinical recommendations."""

import hashlib
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pytest

from medical_rag.ingestion.chunking.sections import chunk_document
from medical_rag.ingestion.loaders.docx_reader import (
    APPROVED_SOURCES,
    discover_approved_sources,
    read_docx,
)
from medical_rag.ingestion.processing.pipeline import rebuild_knowledge_base


def save_document(document, directory: Path, name="fixture.docx") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    return path


def add_table(document, rows):
    table = document.add_table(rows=0, cols=len(rows[0]))
    for values in rows:
        for cell, value in zip(table.add_row().cells, values):
            cell.text = value
    return table


def test_paragraph_table_order_exact_unicode_and_source_preservation(tmp_path):
    document = Document()
    document.add_heading("Test Class – مجموعة اختبار", level=2)
    document.add_paragraph("")
    original_text = "  نص عربي English Δ – لا يُترجم  "
    document.add_paragraph(original_text)
    rows = [
        ["Drug name / الاسم", "Active ingredient / المادة الفعالة"],
        ["Test Brand A", "Test Ingredient A"],
        ["Test Brand B", "Test Ingredient B\nملاحظة اختبار"],
    ]
    add_table(document, rows)
    document.add_paragraph("Warning / تحذير:")
    document.add_paragraph("Synthetic warning – نص اختبار فقط.")
    path = save_document(document, tmp_path)
    before = path.read_bytes()

    parsed = read_docx(path, "KB-01")

    assert [block["kind"] for block in parsed["blocks"]] == [
        "heading", "paragraph", "table", "label", "paragraph"
    ]
    assert [block["position"] for block in parsed["blocks"]] == [1, 3, 4, 5, 6]
    assert parsed["blocks"][1]["text"] == original_text
    assert parsed["blocks"][2]["headers"] == [rows[0]]
    assert parsed["blocks"][2]["rows"] == rows[1:]
    assert parsed["blocks"][2]["header_source"] == "first_row_inferred"
    assert parsed["blocks"][-1]["section_path"] == [
        "Test Class – مجموعة اختبار", "Warning / تحذير:"
    ]
    assert parsed["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before

    chunks = chunk_document(parsed)
    assert len(chunks) == 1
    for text in (original_text, "Test Brand A | Test Ingredient A", "Test Brand B | Test Ingredient B\nملاحظة اختبار", "Synthetic warning – نص اختبار فقط."):
        assert text in chunks[0]["text"]


def test_outline_levels_and_recorded_numbered_heading_correction(tmp_path):
    document = Document()
    heading = document.add_paragraph("1. Outline Class")
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), "1")
    heading._p.get_or_add_pPr().append(outline)
    document.add_paragraph("Source text.")
    document.add_heading("2. Numbered Class", level=3)
    document.add_heading("Warnings", level=3)
    document.add_paragraph("Synthetic class warning.")

    path = save_document(document, tmp_path)
    # Numbered level-3 headings are no longer promoted for every document.
    uncorrected = read_docx(path, "KB-01")
    assert uncorrected["blocks"][2]["heading_level"] == 3
    parsed = read_docx(path, "KB-01", corrections={
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rules": [{"position": 3, "heading_level": 2, "reason": "Reviewed sibling class"}],
    })

    assert parsed["blocks"][0]["heading_source"] == "paragraph_outline"
    corrected = parsed["blocks"][2]
    assert corrected["source_heading_level"] == 3
    assert corrected["heading_level"] == 2
    assert parsed["notes"] == ["Block 3: Reviewed sibling class"]
    assert parsed["blocks"][-1]["section_path"] == ["2. Numbered Class", "Warnings"]
    chunks = chunk_document(parsed)
    assert len(chunks) == 2
    assert chunks[1]["section"] == "2. Numbered Class"
    assert "Synthetic class warning." in chunks[1]["text"]


def test_word_repeated_headers_are_preserved(tmp_path):
    document = Document()
    rows = [["Names", "Ingredients"], ["Brand", "Compound"], ["Brand A", "Compound A"]]
    table = add_table(document, rows)
    for row in table.rows[:2]:
        row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))

    parsed = read_docx(save_document(document, tmp_path), "KB-01")
    block = parsed["blocks"][0]

    assert block["headers"] == rows[:2]
    assert block["rows"] == rows[2:]
    assert block["header_source"] == "word_repeated_header"


def test_long_table_splits_only_between_rows_with_repeated_headers(tmp_path):
    document = Document()
    document.add_heading("Synthetic Drug Class", level=2)
    rows = [["Drug", "Active ingredient"]] + [
        [f"Brand {i:02d}", f"Compound {i:02d} – مادة اختبار"] for i in range(20)
    ]
    add_table(document, rows)
    document.add_paragraph("Warnings:")
    document.add_paragraph("Synthetic class warning remains associated with this class.")
    parsed = read_docx(save_document(document, tmp_path), "KB-01")

    chunks = chunk_document(parsed, max_chars=220)

    table_chunks = [chunk for chunk in chunks if any("row_start" in span for span in chunk["source_spans"])]
    assert len(table_chunks) > 1
    seen_rows = []
    for chunk in table_chunks:
        assert "Synthetic Drug Class" in chunk["text"]
        assert "Drug | Active ingredient" in chunk["text"]
        for span in chunk["source_spans"]:
            if "row_start" in span:
                seen_rows.extend(range(span["row_start"], span["row_end"] + 1))
                for row_index in range(span["row_start"], span["row_end"] + 1):
                    assert " | ".join(rows[row_index - 1]) in chunk["text"]
    assert seen_rows == list(range(2, 22))
    for row in rows[1:]:
        assert sum(" | ".join(row) in chunk["text"] for chunk in chunks) == 1
    warning_chunk = next(chunk for chunk in chunks if "Synthetic class warning" in chunk["text"])
    assert warning_chunk["section"] == "Synthetic Drug Class"
    assert "Warnings:" in warning_chunk["text"]


def test_long_section_uses_paragraph_boundaries_and_keeps_context(tmp_path):
    document = Document()
    document.add_heading("Source Class", level=2)
    document.add_heading("Contraindications", level=3)
    paragraphs = [f"Synthetic warning {i}: " + "qualifier " * 8 for i in range(6)]
    for text in paragraphs:
        document.add_paragraph(text)
    parsed = read_docx(save_document(document, tmp_path), "KB-01")

    chunks = chunk_document(parsed, max_chars=260)

    assert len(chunks) > 1
    for text in paragraphs:
        assert sum(text in chunk["text"] for chunk in chunks) == 1
    for chunk in chunks:
        if "Synthetic warning" in chunk["text"]:
            assert "Source Class" in chunk["text"]
            assert "Contraindications" in chunk["text"]
            assert chunk["section"] == "Source Class"


def test_oversized_paragraph_is_not_cut(tmp_path):
    document = Document()
    document.add_heading("Class", level=2)
    text = "A synthetic indivisible paragraph. " * 30
    document.add_paragraph(text)
    parsed = read_docx(save_document(document, tmp_path), "KB-01")

    chunks = chunk_document(parsed, max_chars=100)
    chunk = next(chunk for chunk in chunks if text in chunk["text"])

    assert chunk["oversized"] is True
    assert chunk["char_count"] == len(chunk["text"])


def test_heading_only_topic_is_carried_into_following_content(tmp_path):
    document = Document()
    document.add_heading("Topic heading", level=2)
    document.add_heading("Source Class", level=2)
    document.add_paragraph("Actual source content.")

    parsed = read_docx(save_document(document, tmp_path), "KB-01")
    chunks = chunk_document(parsed)

    assert len(chunks) == 1
    assert chunks[0]["section"] == "Source Class"
    assert chunks[0]["text"] == "Topic heading\n\nSource Class\nActual source content."
    assert [span["block_position"] for span in chunks[0]["source_spans"]] == [1, 2, 3]


@pytest.mark.parametrize("with_empty_table", [False, True])
def test_empty_content_has_no_chunks(tmp_path, with_empty_table):
    document = Document()
    document.add_paragraph(" \t ")
    if with_empty_table:
        document.add_table(rows=2, cols=2)

    parsed = read_docx(save_document(document, tmp_path), "KB-01")

    assert chunk_document(parsed) == []


def test_title_uses_source_metadata_or_filename(tmp_path):
    document = Document()
    document.core_properties.title = "Actual source title"
    document.add_paragraph("Text without headings")
    path = save_document(document, tmp_path)

    parsed = read_docx(path, "KB-01")
    assert parsed["title"] == "Actual source title"
    assert parsed["title_source"] == "core_properties"
    assert chunk_document(parsed)[0]["section"] == ""
    assert chunk_document(parsed)[0]["scope_uncertain"] is True

    document.core_properties.title = ""
    parsed = read_docx(save_document(document, tmp_path), "KB-01")
    assert parsed["title"] == path.stem
    assert parsed["title_source"] == "filename_stem"


def test_nested_table_requires_review_instead_of_silent_loss(tmp_path):
    document = Document()
    table = document.add_table(rows=1, cols=1)
    table.cell(0, 0).add_table(rows=1, cols=1)

    with pytest.raises(ValueError, match="Nested DOCX tables require review"):
        read_docx(save_document(document, tmp_path), "KB-01")


@pytest.fixture
def approved_fixture_root(tmp_path):
    source = tmp_path / "source"
    for doc_id, filename in APPROVED_SOURCES.items():
        document = Document()
        document.add_heading(f"Synthetic {doc_id} Class", level=2)
        document.add_paragraph("Arabic عربي English – fixture only.")
        add_table(document, [["Drug", "Active ingredient"], ["Test A", "Compound A"]])
        save_document(document, source / doc_id, filename)
    return source


def test_discovery_uses_exact_nine_id_filename_pairs(approved_fixture_root):
    extra = Document()
    save_document(extra, approved_fixture_root / "KB-10", "unapproved.docx")
    save_document(extra, approved_fixture_root / "KB-01", "extra.docx")

    sources = discover_approved_sources(approved_fixture_root)

    assert len(sources) == 9
    assert [doc_id for doc_id, _ in sources] == [f"KB-{i:02d}" for i in range(1, 10)]
    assert [(doc_id, path.name) for doc_id, path in sources] == list(APPROVED_SOURCES.items())
    assert all(path.parent.name == doc_id for doc_id, path in sources)


def test_missing_approved_file_does_not_fall_back_or_write_partial_outputs(approved_fixture_root, tmp_path):
    (approved_fixture_root / "KB-09" / APPROVED_SOURCES["KB-09"]).rename(approved_fixture_root / "KB-09" / "renamed.docx")
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="Missing approved DOCX"):
        rebuild_knowledge_base(approved_fixture_root, output)
    assert not output.exists()


def test_rebuild_is_byte_deterministic_has_metadata_and_never_modifies_sources(approved_fixture_root, tmp_path):
    source_bytes = {path: path.read_bytes() for _, path in discover_approved_sources(approved_fixture_root)}
    output = tmp_path / "output"

    first_manifest = rebuild_knowledge_base(approved_fixture_root, output)
    first_outputs = {path.relative_to(output): path.read_bytes() for path in output.rglob("*.json")}
    second_manifest = rebuild_knowledge_base(approved_fixture_root, output)
    second_outputs = {path.relative_to(output): path.read_bytes() for path in output.rglob("*.json")}

    assert first_manifest == second_manifest
    assert first_outputs == second_outputs
    assert len(first_outputs) == 19
    assert first_manifest["document_count"] == 9
    assert first_manifest["table_count"] == 9
    ids = []
    for doc_id in APPROVED_SOURCES:
        chunks = json.loads((output / "chunks" / f"{doc_id}.json").read_text(encoding="utf-8"))
        for chunk in chunks:
            assert {"chunk_id", "doc_id", "title", "section", "text", "filename", "source_spans"} <= chunk.keys()
            assert chunk["doc_id"] == doc_id
            assert chunk["title"] and chunk["section"] and chunk["text"]
            assert chunk["filename"] == APPROVED_SOURCES[doc_id]
            assert "عربي" in chunk["text"]
            ids.append(chunk["chunk_id"])
    assert len(ids) == len(set(ids)) == first_manifest["chunk_count"]
    assert all(path.read_bytes() == original for path, original in source_bytes.items())


def test_output_cannot_overlap_source(approved_fixture_root):
    with pytest.raises(ValueError, match="must not overlap"):
        rebuild_knowledge_base(approved_fixture_root, approved_fixture_root)


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_chunk_size_fails(tmp_path, size):
    parsed = read_docx(save_document(Document(), tmp_path), "KB-01")
    with pytest.raises(ValueError, match="positive"):
        chunk_document(parsed, size)
    with pytest.raises(ValueError, match="positive"):
        rebuild_knowledge_base(tmp_path / "missing", tmp_path / "output", size)
