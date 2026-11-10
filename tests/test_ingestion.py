"""DOCX ingestion: reader, structure, chunking, single-document API and the nine approved sources."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pytest

from medical_rag.ingestion.docx_reader import (
    APPROVED_SOURCES,
    discover_approved_sources,
    read_docx,
)
from medical_rag.ingestion.docx_reader import APPROVED_SOURCES
from medical_rag.ingestion.docx_reader import APPROVED_SOURCES, discover_approved_sources
from medical_rag.ingestion.pipeline import ingest_document
from medical_rag.ingestion.pipeline import rebuild_knowledge_base
from medical_rag.ingestion.sections import chunk_document


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


@pytest.fixture
def tenth_document(tmp_path):
    document = Document()
    document.add_heading("Synthetic Topic – موضوع اختبار", level=1)
    document.add_heading("Test Drug Class", level=2)
    document.add_paragraph("نص عربي English – synthetic content only.")
    table = document.add_table(rows=3, cols=2)
    for row, values in zip(table.rows, [
        ("Drug / الاسم", "Active ingredient / المادة"),
        ("Test Brand A", "Test Compound A"),
        ("Test Brand B", "Test Compound B"),
    ]):
        for cell, value in zip(row.cells, values):
            cell.text = value
    document.add_heading("Side Effects", level=3)
    document.add_paragraph("Synthetic effect – أثر اختبار.")
    document.add_heading("Contraindications", level=3)
    document.add_paragraph("Synthetic warning – تحذير اختبار، مع قيد محفوظ.")
    path = tmp_path / "independent-tenth.docx"
    document.save(path)
    return path


def test_tenth_document_ingestion_is_independent_read_only_and_not_approval(tenth_document):
    root = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"
    outputs = [*root.glob("chunks/*.json"), *root.glob("processed/*.json"), root / "metadata/generated/ingestion_manifest.json"]
    before = {path: path.read_bytes() for path in outputs}
    original = tenth_document.read_bytes()
    catalog = dict(APPROVED_SOURCES)
    directory_files = set(tenth_document.parent.iterdir())

    result = ingest_document(tenth_document, "candidate-10_arbitrary", {"source_reference": "synthetic-test"})

    assert result["doc_id"] == "candidate-10_arbitrary"
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["requires_review"] is False  # Structural readiness is not approval.
    assert {problem["code"] for problem in result["issues"]} == {"inferred_table_header"}
    chunk = result["chunks"][0]
    for text in (
        "نص عربي English", "Drug / الاسم | Active ingredient / المادة",
        "Test Brand A | Test Compound A", "Test Brand B | Test Compound B",
        "Synthetic effect – أثر اختبار.", "Synthetic warning – تحذير اختبار، مع قيد محفوظ.",
    ):
        assert text in chunk["text"]
    assert chunk["section_path"] == ["Synthetic Topic – موضوع اختبار", "Test Drug Class"]
    assert chunk["build_id"] == result["build_id"]
    assert result["build_id"] in chunk["chunk_id"]
    assert not ({"approved", "approval_status", "active", "indexed"} & result.keys())
    assert not ({"approved", "approval_status"} & result["document"].keys())
    assert tenth_document.read_bytes() == original
    assert dict(APPROVED_SOURCES) == catalog and len(catalog) == 9
    assert "candidate-10_arbitrary" not in APPROVED_SOURCES
    assert set(tenth_document.parent.iterdir()) == directory_files
    assert all(path.read_bytes() == content for path, content in before.items())


def test_identical_inputs_and_metadata_order_have_identical_results(tenth_document):
    first = ingest_document(tenth_document, "arbitrary", {"original_filename": "Original.docx", "title": "Source title"})
    second = ingest_document(tenth_document, "arbitrary", {"title": "Source title", "original_filename": "Original.docx"})
    assert first == second


def test_original_filename_is_not_the_storage_path(tenth_document, tmp_path):
    moved = tmp_path / "stored-under-generated-name.docx"
    moved.write_bytes(tenth_document.read_bytes())
    metadata = {"original_filename": "Original title.docx"}
    first = ingest_document(tenth_document, "assigned", metadata)
    second = ingest_document(moved, "assigned", metadata)
    assert first == second
    assert first["document"]["filename"] == "Original title.docx"
    assert first["document"]["title"] == "Original title"


def test_source_config_and_document_id_changes_change_build_identity(tenth_document):
    first = ingest_document(tenth_document, "assigned")
    configured = ingest_document(tenth_document, "assigned", max_chars=200)
    other_id = ingest_document(tenth_document, "different")
    assert configured["config_id"] != first["config_id"]
    assert configured["build_id"] != first["build_id"] != other_id["build_id"]
    document = Document(tenth_document)
    document.add_paragraph("Changed source fixture.")
    document.save(tenth_document)
    changed = ingest_document(tenth_document, "assigned")
    assert changed["source_sha256"] != first["source_sha256"]
    assert changed["build_id"] != first["build_id"]
    assert not ({c["chunk_id"] for c in first["chunks"]} & {c["chunk_id"] for c in changed["chunks"]})


def test_generic_checksum_bound_correction_resolves_same_level_topics(tmp_path):
    document = Document()
    document.add_heading("1. Topic", level=2)
    document.add_heading("1. Class", level=2)
    document.add_paragraph("Original source text.")
    path = tmp_path / "unfamiliar.docx"
    document.save(path)
    original = path.read_bytes()
    uncorrected = ingest_document(path, "not-a-KB-id")
    assert uncorrected["requires_review"] is True
    assert "ambiguous_heading_hierarchy" in {i["code"] for i in uncorrected["issues"]}
    correction = {
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "rules": [{"position": 1, "heading_level": 1, "reason": "Reviewed source topic"}],
    }

    corrected = ingest_document(path, "not-a-KB-id", corrections=correction)

    assert corrected["requires_review"] is False
    assert corrected["chunks"][0]["section"] == "1. Topic > 1. Class"
    assert corrected["chunks"][0]["correction_positions"] == [1]
    assert corrected["document"]["blocks"][0]["source_heading_level"] == 2
    assert corrected["document"]["blocks"][0]["heading_level"] == 1
    assert corrected["document"]["applied_corrections"] == correction["rules"]
    assert corrected == ingest_document(path, "not-a-KB-id", corrections=correction)
    assert corrected["build_id"] != uncorrected["build_id"]
    revised = deepcopy(correction)
    revised["rules"][0]["reason"] = "A revised review explanation"
    assert ingest_document(path, "not-a-KB-id", corrections=revised)["build_id"] != corrected["build_id"]
    assert path.read_bytes() == original
    document.add_paragraph("New source version")
    document.save(path)
    with pytest.raises(ValueError, match="checksum"):
        ingest_document(path, "not-a-KB-id", corrections=correction)


@pytest.mark.parametrize("rule", [
    {"position": 999, "heading_level": 1},
    {"position": 1, "heading_level": 10},
    {"position": 1, "parent_position": 2},
    {"position": 5, "parent_position": 3},
    {"position": 5, "topic_positions": [3]},
    {"position": 1, "heading_level": 1, "invented_heading": "No"},
])
def test_invalid_correction_targets_are_rejected(tenth_document, rule):
    correction = {"source_sha256": hashlib.sha256(tenth_document.read_bytes()).hexdigest(), "rules": [{**rule, "reason": "Test"}]}
    with pytest.raises(ValueError):
        ingest_document(tenth_document, "assigned", corrections=correction)


@pytest.mark.parametrize("identifier", ["../escape", "bad/name", "", "C:\\path", "a" * 129])
def test_invalid_document_ids_are_rejected(tenth_document, identifier):
    with pytest.raises(ValueError, match="doc_id"):
        ingest_document(tenth_document, identifier)


def test_approval_is_not_accepted_as_source_metadata(tenth_document):
    with pytest.raises(ValueError, match="approval"):
        ingest_document(tenth_document, "assigned", {"approved": True})


def test_missing_headings_remain_explicitly_unresolved(tmp_path):
    document = Document()
    document.add_paragraph("Unheaded text – نص بدون عنوان.")
    path = tmp_path / "missing.docx"
    document.save(path)
    result = ingest_document(path, "unheaded")
    assert result["requires_review"] is True
    assert result["chunks"][0]["section"] == ""
    assert result["chunks"][0]["scope_uncertain"] is True
    assert "missing_section_heading" in {i["code"] for i in result["issues"]}


def test_unstyled_bold_arabic_subheading_is_flagged_not_promoted(tmp_path):
    document = Document()
    document.add_heading("Source topic", level=1)
    paragraph = document.add_paragraph()
    paragraph.add_run("عنوان فرعي غير منسق").bold = True
    document.add_paragraph("Source detail.")
    path = tmp_path / "bold.docx"
    document.save(path)
    result = ingest_document(path, "bold")
    assert result["requires_review"] is True
    assert "unformatted_subheading" in {i["code"] for i in result["issues"]}
    assert result["document"]["blocks"][1]["kind"] == "paragraph"
    assert result["chunks"][0]["scope_uncertain"] is True


@pytest.mark.parametrize("layout,expected", [
    ("blank_header", "uncertain_table_header"),
    ("merged", "merged_table_layout"),
    ("irregular", "irregular_table_layout"),
])
def test_unfamiliar_table_layouts_are_preserved_and_flagged(tmp_path, layout, expected):
    document = Document()
    document.add_heading("Source Topic", level=1)
    table = document.add_table(rows=2, cols=2)
    for row, values in zip(table.rows, [("Name", "Value"), ("Test A", "Test B")]):
        for cell, value in zip(row.cells, values):
            cell.text = value
    if layout == "blank_header":
        table.cell(0, 1).text = ""
    elif layout == "merged":
        table.cell(0, 0).merge(table.cell(0, 1))
    else:
        row = table.rows[1]
        row._tr.remove(row._tr.tc_lst[0])
        grid_before = OxmlElement("w:gridBefore")
        grid_before.set(qn("w:val"), "1")
        row._tr.get_or_add_trPr().append(grid_before)
    path = tmp_path / "layout.docx"
    document.save(path)
    original = path.read_bytes()
    result = ingest_document(path, "layout")
    assert result["requires_review"] is True
    assert expected in {i["code"] for i in result["issues"]}
    assert result["document"] is not None and result["chunks"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("unsupported,code", [("nested", "nested_table"), ("container", "unsupported_body_content")])
def test_unsupported_content_returns_an_error_instead_of_partial_chunks(tmp_path, unsupported, code):
    document = Document()
    document.add_heading("Source", level=1)
    if unsupported == "nested":
        document.add_table(rows=1, cols=1).cell(0, 0).add_table(rows=1, cols=1)
    else:
        container = OxmlElement("w:sdt")
        content = OxmlElement("w:sdtContent")
        paragraph = document.add_paragraph("Text inside unsupported container")
        content.append(paragraph._p)
        container.append(content)
        document.element.body.insert(1, container)
    path = tmp_path / "unsupported.docx"
    document.save(path)
    result = ingest_document(path, "unsupported")
    assert result["document"] is None and result["chunks"] == []
    assert result["requires_review"] is True
    assert result["issues"][0]["code"] == code
    assert result["issues"][0]["severity"] == "error"
    assert result["issues"][0]["block_positions"]


def test_unextracted_header_text_is_reported(tenth_document):
    document = Document(tenth_document)
    document.sections[0].header.paragraphs[0].text = "Additional source text outside the body"
    document.save(tenth_document)
    result = ingest_document(tenth_document, "header")
    assert result["requires_review"] is True
    assert "unextracted_document_part" in {i["code"] for i in result["issues"]}


def test_invalid_docx_package_returns_parse_issue(tmp_path):
    path = tmp_path / "invalid.docx"
    path.write_bytes(b"Not a DOCX ZIP package")
    result = ingest_document(path, "invalid")
    assert result["document"] is None and result["chunks"] == []
    assert result["issues"][0]["code"] == "invalid_docx"


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "knowledge_base" / "source"
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
