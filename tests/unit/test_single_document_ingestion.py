"""Reusable ingestion tests: candidates never join the bootstrap by being parsed."""

from copy import deepcopy
import hashlib
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import pytest

from medical_rag.ingestion.loaders.docx_reader import APPROVED_SOURCES
from medical_rag.ingestion.processing.pipeline import ingest_document


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
    root = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
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
