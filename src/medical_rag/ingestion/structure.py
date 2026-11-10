"""Resolve source structure conservatively; reviewed rules never change text."""

from copy import deepcopy
import re

from medical_rag.ingestion.models import ParsedDocument, StructuralIssue, StructureCorrections


def issue(code: str, severity: str, message: str, *positions: int) -> StructuralIssue:
    return {"code": code, "severity": severity, "message": message, "block_positions": list(positions)}


def resolve_structure(
    document: ParsedDocument, corrections: StructureCorrections | None = None
) -> ParsedDocument:
    """Use observed levels by default; apply only checksum-bound, explicit rules."""
    result = deepcopy(document)
    blocks = result["blocks"]
    by_position = {block["position"]: block for block in blocks}
    rules = {}
    if corrections is not None:
        if set(corrections) != {"source_sha256", "rules"}:
            raise ValueError("Corrections require exactly source_sha256 and rules")
        if corrections["source_sha256"] != result["source_sha256"]:
            raise ValueError("Structure correction checksum does not match the source")
        for rule in corrections["rules"]:
            if set(rule) - {"position", "reason", "heading_level", "parent_position", "new_section", "topic_positions"}:
                raise ValueError("Unknown structure correction field")
            position = rule["position"]
            if type(position) is not int or position not in by_position or position in rules:
                raise ValueError("Correction positions must be existing, unique source blocks")
            if not isinstance(rule.get("reason"), str) or not rule["reason"].strip():
                raise ValueError("Every structure correction requires a reason")
            if "heading_level" in rule:
                if type(rule["heading_level"]) is not int or not 1 <= rule["heading_level"] <= 9 or by_position[position]["kind"] == "table":
                    raise ValueError("Corrected headings must be paragraphs with levels 1 through 9")
            parent = rule.get("parent_position")
            if parent is not None and (type(parent) is not int or parent not in by_position or parent >= position):
                raise ValueError("A corrected parent must be an earlier source heading")
            if "new_section" in rule and type(rule["new_section"]) is not bool:
                raise ValueError("new_section must be a boolean")
            topics = rule.get("topic_positions", [])
            if not isinstance(topics, list) or any(type(p) is not int or p not in by_position or p >= position for p in topics):
                raise ValueError("Context topics must refer to earlier source headings")
            rules[position] = rule

    result["applied_corrections"] = [deepcopy(rules[p]) for p in sorted(rules)]
    # A stack entry is (effective level, original text, original block position).
    stack: list[tuple[int, str, int]] = []
    snapshots: dict[int, list[tuple[int, str, int]]] = {}
    provenance: dict[int, set[int]] = {}
    section_start = 0
    label = None
    label_corrections: set[int] = set()
    scope_uncertain = False
    topic_positions: list[int] | None = None
    section_corrections: set[int] = set()
    previous_heading = None
    previous_block = None
    last_number: dict[int, tuple[int, int]] = {}
    uncertain_headings: set[int] = set()

    for block in blocks:
        position, text = block["position"], block["text"]
        rule = rules.get(position)
        raw_level = block.get("source_heading_level")
        level = rule.get("heading_level", raw_level) if rule else raw_level
        if rule and "parent_position" in rule:
            parent = rule["parent_position"]
            if parent is not None and parent not in snapshots:
                raise ValueError("Corrected parent does not refer to a resolved source heading")
            stack = list(snapshots[parent]) if parent is not None else []
            label = None
            label_corrections = set()
            scope_uncertain = by_position[parent].get("scope_uncertain", False) if parent is not None else False

        if level is not None:
            issue_start = len(result["issues"])
            if not rule:
                if not stack and level > 1:
                    result["issues"].append(issue("heading_without_parent", "review", "Heading starts below level 1; parent scope is uncertain", position))
                elif stack and level > stack[-1][0] + 1:
                    result["issues"].append(issue("heading_level_gap", "review", "Heading skips a structural level", position))
                number = re.match(r"^\s*(\d+)[.)]\s+", text)
                earlier = last_number.get(level)
                if number and earlier and int(number[1]) <= earlier[0]:
                    result["issues"].append(issue("ambiguous_heading_hierarchy", "review", "Same-level numbering restarts; a topic/class relationship may be missing", earlier[1], position))
                if previous_heading and previous_block and previous_block["kind"] == "heading" and previous_heading[0] == level:
                    result["issues"].append(issue("ambiguous_heading_hierarchy", "review", "Consecutive same-level headings do not establish parentage", previous_heading[2], position))
            while stack and stack[-1][0] >= level:
                stack.pop()
            for old_level in list(last_number):
                if old_level > level:
                    del last_number[old_level]
            number = re.match(r"^\s*(\d+)[.)]\s+", text)
            if number:
                last_number[level] = (int(number[1]), position)
            stack.append((level, text, position))
            block["kind"] = "heading"
            block["heading_level"] = level
            if raw_level is None:
                block["heading_source"] = "reviewed_correction"
            label = None
            label_corrections = set()
            if any(problem["severity"] == "review" for problem in result["issues"][issue_start:]):
                uncertain_headings.add(position)
            scope_uncertain = any(p in uncertain_headings for _, _, p in stack)
            previous_heading = stack[-1]
            if level <= 2 or section_start == 0:
                section_start = position
                topic_positions = None
                section_corrections = set()
            inherited = set().union(*(provenance[p] for _, _, p in stack[:-1])) if stack[:-1] else set()
            provenance[position] = inherited | ({position} if rule else set())
            snapshots[position] = list(stack)
        elif block["kind"] == "label":
            if stack and stack[-1][0] >= 3 and not rule:
                result["issues"].append(issue("ambiguous_label_scope", "review", "Label may apply to the whole class rather than the preceding subclass; using the major scope tentatively", position))
                stack = [entry for entry in stack if entry[0] <= 2]
                scope_uncertain = True
            label = text
            label_corrections = {position} if rule else set()
        elif block["kind"] == "paragraph" and not rule and (
            re.match(r"^\s*[A-Z][.)]\s+\S", text)
            or re.fullmatch(r"Side Effects|Contraindications|Warnings|Precautions", text, re.IGNORECASE)
            or block.get("heading_candidate", False)
        ):
            result["issues"].append(issue("unformatted_subheading", "review", "Possible subsection is ordinary paragraph text; its scope needs review", position))
            scope_uncertain = True

        if rule:
            if rule.get("new_section"):
                section_start = position
                label = None
                label_corrections = set()
                section_corrections = set()
                topic_positions = None
            if "topic_positions" in rule:
                for topic in rule["topic_positions"]:
                    if topic not in snapshots:
                        raise ValueError("Context topics must refer to resolved source headings")
                topic_positions = rule["topic_positions"]
            section_corrections.add(position)
            result["notes"].append(f"Block {position}: {rule['reason']}")
            result["issues"].append(issue("reviewed_structure_correction", "info", rule["reason"], position))

        path = [heading for _, heading, _ in stack] + ([label] if label else [])
        if not stack and text.strip():
            scope_uncertain = True
            if not any(problem["code"] == "missing_section_heading" for problem in result["issues"]):
                result["issues"].append(issue("missing_section_heading", "review", "Content has no reliable source section heading; no heading was invented", position))
        block["section_path"] = path
        block["section_start"] = section_start
        block["context_topics"] = (
            [by_position[p]["text"] for p in topic_positions]
            if topic_positions is not None else [heading for rank, heading, _ in stack if rank == 1]
        )
        block["scope_uncertain"] = scope_uncertain
        inherited = set().union(*(provenance[p] for _, _, p in stack)) if stack else set()
        block["correction_positions"] = sorted(inherited | label_corrections | section_corrections)
        previous_block = block

    if not any(block["text"].strip() for block in blocks):
        result["issues"].append(issue("empty_document", "review", "No nonempty body content was extracted"))
    return result
