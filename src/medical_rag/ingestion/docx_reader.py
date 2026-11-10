"""Read source text and table structure without saving or rewriting the DOCX."""

import hashlib
from io import BytesIO
from pathlib import Path
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from medical_rag.ingestion.models import ParsedDocument, SourceBlock, SourceMetadata, StructuralIssue, StructureCorrections
from medical_rag.ingestion.structure import issue, resolve_structure


# Deliberately explicit: extra files or KB folders are never ingested implicitly.
APPROVED_SOURCES = {
    "KB-01": "asthma w copd.docx",
    "KB-02": "gerd& vomiting.docx",
    "KB-03": "pain headache ibs arthritis.docx",
    "KB-04": "pharyngitis.docx",
    "KB-05": "أدوية الالتهاب الرئوي.docx",
    "KB-06": "ادويه الحساسيه و الجيوب الانفيه.docx",
    "KB-07": "ادويه السكر.docx",
    "KB-08": "ادويه الضغط و الكوليسترول.docx",
    "KB-09": "ادويه القولون و التهابات المفاصل.docx",
}


def discover_approved_sources(source_root: Path) -> list[tuple[str, Path]]:
    """Require each approved ID/filename pair; ignore all other documents."""
    sources = [(doc_id, source_root / doc_id / name) for doc_id, name in APPROVED_SOURCES.items()]
    missing = [str(path) for _, path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing approved DOCX source(s): " + "; ".join(missing))
    return sources


def heading_level(paragraph: Paragraph) -> tuple[int, str] | None:
    """Prefer explicit outline levels, then heading styles and inherited outlines."""
    outlines = paragraph._p.xpath("./w:pPr/w:outlineLvl")
    if outlines:
        value = int(outlines[0].get(qn("w:val")))
        return (value + 1, "paragraph_outline") if 0 <= value <= 8 else None

    style = paragraph.style
    visited = set()
    while style is not None and style.style_id not in visited:
        visited.add(style.style_id)
        match = re.fullmatch(r"Heading ([1-9])", style.name, flags=re.IGNORECASE)
        if match:
            return int(match[1]), "heading_style"
        if style.name == "Title":
            return 1, "title_style"
        outlines = style.element.xpath("./w:pPr/w:outlineLvl")
        if outlines:
            value = int(outlines[0].get(qn("w:val")))
            return (value + 1, "style_outline") if 0 <= value <= 8 else None
        style = style.base_style
    return None


def render_table(headers: list[list[str]], rows: list[list[str]]) -> str:
    """Use plain separators; the JSON arrays retain exact source cell strings."""
    return "\n".join(" | ".join(row) for row in [*headers, *rows] if any(cell.strip() for cell in row))


def read_table(table: Table) -> tuple[list[list[str]], list[list[str]], str]:
    """Keep row/column order; honor repeated-header flags or infer the first row."""
    if table._tbl.xpath(".//w:tbl"):
        raise ValueError("Nested DOCX tables require review; refusing to flatten them silently")
    rows = []
    header_count = 0
    for index, row in enumerate(table.rows):
        # Pad omitted grid cells so subsequent cells do not shift under wrong headers.
        rows.append(
            [""] * row.grid_cols_before
            + [cell.text for cell in row.cells]
            + [""] * row.grid_cols_after
        )
        flags = row._tr.xpath("./w:trPr/w:tblHeader")
        is_header = bool(flags) and flags[0].get(qn("w:val"), "true").lower() not in {"0", "false", "off"}
        if index == header_count and is_header:
            header_count += 1
    if not rows:
        return [], [], "none"
    source = "word_repeated_header" if header_count else "first_row_inferred"
    header_count = header_count or 1
    return rows[:header_count], rows[header_count:], source


class UnsupportedDocxStructure(ValueError):
    """A partial extraction must not silently become a usable chunk build."""

    def __init__(self, problem: StructuralIssue):
        super().__init__(problem["message"])
        self.issue = problem


def read_docx(
    path: Path,
    doc_id: str,
    *,
    source_metadata: SourceMetadata | None = None,
    corrections: StructureCorrections | None = None,
    source_bytes: bytes | None = None,
) -> ParsedDocument:
    """Extract body evidence; interpretation is handled by the generic resolver."""
    source = path.read_bytes() if source_bytes is None else source_bytes
    document = Document(BytesIO(source))
    metadata = dict(source_metadata or {})
    filename = metadata.get("original_filename", path.name)
    title = metadata.get("title", document.core_properties.title)
    title_source = "source_metadata" if "title" in metadata else "core_properties"
    if not title.strip():
        title, title_source = Path(filename).stem, "filename_stem"
    blocks: list[SourceBlock] = []
    notes: list[str] = []
    issues: list[StructuralIssue] = []

    # Inspect extra parts so body-only extraction is never presented as complete.
    with ZipFile(BytesIO(source)) as archive:
        for name in sorted(archive.namelist()):
            if re.fullmatch(r"word/(?:header\d*|footer\d*|footnotes|endnotes|comments)\.xml", name):
                root = ET.fromstring(archive.read(name))
                if any((node.text or "").strip() for node in root.iter(qn("w:t"))):
                    issues.append(issue("unextracted_document_part", "review", f"Text in {name} is outside supported body extraction"))

    for position, element in enumerate(document.element.body, start=1):
        unsupported_tags = {qn(f"w:{name}") for name in ("txbxContent", "ins", "del", "drawing", "pict", "footnoteReference", "endnoteReference")}
        unsupported = any(node.tag in unsupported_tags for node in element.iter())
        if unsupported or element.tag not in {qn("w:p"), qn("w:tbl"), qn("w:sectPr")}:
            raise UnsupportedDocxStructure(issue("unsupported_body_content", "error", "Body content cannot be reliably extracted without losing its structure", position))
        if element.tag == qn("w:p"):
            item = Paragraph(element, document)
            text = item.text
            if not text.strip():
                continue
            detected = heading_level(item)
            kind = "heading" if detected else "label" if len(text.strip()) <= 80 and text.rstrip().endswith((":", "：")) else "paragraph"
            block: SourceBlock = {
                "position": position,
                "kind": kind,
                "text": text,
                "section_path": [],
                "section_start": 0,
                "source_heading_level": detected[0] if detected else None,
            }
            if detected:
                block.update(heading_level=detected[0], heading_source=detected[1])
            elif kind == "paragraph":
                runs = [run for run in item.runs if run.text.strip()]
                if len(text.strip()) <= 100 and runs and all(run.bold is True for run in runs):
                    block["heading_candidate"] = True
            blocks.append(block)
        elif element.tag == qn("w:tbl"):
            item = Table(element, document)
            try:
                headers, rows, header_source = read_table(item)
            except ValueError as error:
                raise UnsupportedDocxStructure(issue("nested_table", "error", str(error), position)) from error
            blocks.append({
                "position": position,
                "kind": "table",
                "text": render_table(headers, rows),
                "section_path": [],
                "section_start": 0,
                "headers": headers,
                "rows": rows,
                "header_source": header_source,
            })
            if header_source == "first_row_inferred":
                issues.append(issue("inferred_table_header", "info", "First row retained as a tentative header; Word provides no repeated-header flag", position))
            if headers and any(not cell.strip() for row in headers for cell in row):
                issues.append(issue("uncertain_table_header", "review", "Header contains blank cells; column meanings require review", position))
            if any(any(cell.strip() for cell in row) and any(not cell.strip() for cell in row) for row in rows):
                issues.append(issue("blank_table_value", "review", "A nonempty source data row contains a blank value; no missing value was inferred", position))
            widths = {len(row) for row in [*headers, *rows]}
            if len(widths) > 1 or any(row.grid_cols_before or row.grid_cols_after for row in item.rows):
                issues.append(issue("irregular_table_layout", "review", "Rows have inconsistent or omitted grid cells", position))
            if item._tbl.xpath(".//w:gridSpan | .//w:vMerge"):
                notes.append(f"Block {position}: merged cells expanded into their grid positions")
                issues.append(issue("merged_table_layout", "review", "Merged cells were expanded; header and row relationships need review", position))

    return resolve_structure({
        "doc_id": doc_id,
        "title": title,
        "title_source": title_source,
        "filename": filename,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "blocks": blocks,
        "notes": notes,
        "issues": issues,
        "source_metadata": metadata,
        "applied_corrections": [],
    }, corrections)
