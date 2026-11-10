"""Deterministic checks on a generated answer before it is published.

These checks are language-independent and target the failures that matter most:
  1. citations must quote real text from the passage they name (snippets are genuine source lines);
  2. every medicine name in the answer must appear in the cited chunks (no drug substitution or
     outside knowledge about other drugs);
  3. every number in the answer must appear in the cited chunks (no invented doses or strengths).
They do not try to prove semantic entailment; the prompt, the citations and the evaluation set
cover meaning.
"""
from dataclasses import dataclass
import re

from medical_rag.api.schemas import Citation
from medical_rag.generation.ollama import GroundedDraft
from medical_rag.generation.prompt import Passage
from medical_rag.knowledge.entities import EntityIndex
from medical_rag.retrieval.normalization import normalize_text


class VerificationError(ValueError):
    """The draft could not be grounded; the reason is logged, never shown to the user."""


@dataclass(frozen=True)
class VerifiedAnswer:
    answer: str
    citations: list[Citation]
    dropped_quotes: int


_PUNCTUATION = re.compile(r"[|•*–—\-:،,;.!?؟()\[\]\"'`’“”«»/]+")
# "[E1]", "(E1, E2)", "(Passage E4)", "passage E4", bare "E4": evidence labels never belong in an answer.
_LABEL_IN_TEXT = re.compile(r"\s*[\[(]\s*(?:passages?\s*)?E\d+(?:\s*(?:[,،]|and|و)\s*E\d+)*\s*[\])]|"
                            r"\s*\b(?:in |from |see )?(?:passages?\s*)\bE\d+\b|(?<![\w-])E[1-9]\b(?!-)", re.I)
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _canon(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", normalize_text(text)).split())


def locate_quote(quote: str, text: str) -> str | None:
    """Return the genuine source line(s) a quote comes from, or None if it is not in the text.

    Tolerates whitespace, punctuation, table separators and Arabic letter variants, but every
    word must be present in order.
    """
    quote = quote.replace("\\n", "\n").strip()  # small models often emit an escaped newline inside the string
    if len(_canon(quote)) < 4:
        return None
    lines = [line for line in text.splitlines() if line.strip()]
    found = []
    for part in (p for p in quote.splitlines() if _canon(p)):
        target = _canon(part)
        line = next((line for line in lines if target in _canon(line)), None)
        if line is None:
            # A quote may run across several source lines joined by spaces.
            spanned = [line for line in lines if _canon(line) and _canon(line) in target]
            if not spanned or target not in _canon(" ".join(spanned)):
                return None
            found.extend(spanned)
        else:
            found.append(line)
    return "\n".join(dict.fromkeys(line.strip() for line in found)) or None


_HEADER_ONLY = re.compile(r"^(?:(?:trade|brand) names?|side effects?|الاسم التجاري[^|]*)\s*\|[^|]*$|.*:\s*$", re.I)


def contiguous_runs(snippet: str, text: str) -> list[str]:
    """Split located lines into runs that are contiguous in the source; drop header-only runs.

    A citation then never juxtaposes lines that are apart in the document (e.g. a table header
    and a row ten lines below it).
    """
    runs = []
    for line in snippet.splitlines():
        if runs and (runs[-1] + "\n" + line) in text:
            runs[-1] += "\n" + line
        else:
            runs.append(line)
    kept = [run for run in runs if not ("\n" not in run and _HEADER_ONLY.match(run.strip()))]
    return kept or runs


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(normalize_text(text))}


_CITATION_SECTION = re.compile(r"(?im)^\s*\**(?:citations?|sources?|references?|quotes?|evidence)\**\s*:.*\Z", re.S)
_QUOTED_LINE = re.compile(r"(?m)^(?P<bullet>\s*(?:[-•*]\s*)?)(?P<prefix>(?:(?:passage|source|evidence|quote)\w*"
                          r"(?:\s+\w+)?\s*)?(?:\[?E\d+\]?\s*)?[-:]?\s*)[\"'“«](?P<body>[^\n]*)[\"'”»]\.?[ \t]*$\n?", re.I)
_PASSAGE_REFERENCE = re.compile(r"\b(?:in |from |see )?passages?\s+E\d+(?:\s*(?:,|،|and|&|و)\s*E\d+)*", re.I)
_ARABIC_GLOSS = re.compile(r"\s*\((?=[^)]*[ء-ي])[^)a-zA-Z]*\)")
_INLINE_QUOTE = re.compile(r"\[?(E\d)\]?\s*[:\-–]?\s*[\"'“«](.{4,600}?)[\"'”»](?=\s*$|\s*[,;\n])", re.M | re.S)


def clean_answer(text: str, language: str = "ar") -> str:
    """Remove any quotation list the model pasted into the answer text, then evidence labels.

    For English answers, parenthesised Arabic glosses copied from the bilingual source tables are
    dropped; the English term they translate is already in the sentence.
    """
    if language == "en":
        text = _ARABIC_GLOSS.sub("", text)
    text = _CITATION_SECTION.sub("", text)
    text = re.sub(r"(?im)^[ \t]*(?:[-•*][ \t]*)?(?:passage[ \t]*)?" + _INLINE_QUOTE.pattern, "", text, flags=re.S)
    # "Passage states: '...'" is a pasted citation: drop it. A bare quoted line is content: unquote it.
    text = _QUOTED_LINE.sub(lambda m: "" if m["prefix"].strip() else m["bullet"] + m["body"] + "\n", text)
    text = _PASSAGE_REFERENCE.sub("the source", text)
    text = _LABEL_IN_TEXT.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:،؛])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_NOT_COVERED = re.compile(
    r"\b(?:passages?|sources?|knowledge base|kb|documents?)\b[^.\n]{0,30}\b(?:do(?:es)?(?: not|n['’]t)|"
    r"contains? no)\b[^.\n]{0,20}\b(?:contain|include|provide|mention|cover|have|give)?\b[^.\n]{0,15}"
    r"\b(?:information|details?|data|mention)\b|\b(?:do(?:es)? not|don['’]t|doesn['’]t) (?:contain|include|provide|"
    r"give|cover)\b[^.\n]{0,15}\b(?:information|details?|data)\b|\bnot covered\b|\bnot (?:found |available )?in the "
    r"(?:knowledge base|passages|sources?)\b|\bno information\b|"
    r"(?:لا|مش|ما) ?(?:تحتوي|يحتوي|تحتويش|بتحتوي|بيحتوي|فيهاش|فيهوش|موجود|متوفر|تذكر|يذكر|بيذكر|بتذكر)|"
    r"غير (?:موجود|متوفر|مذكور)|(?:لا توجد|لا يوجد|مفيش|ما فيش|مافيش) (?:اي )?معلوم(?:ات|ة|ه)",
    re.I)


_META_SUBJECT = re.compile(r"\W*(?:the |in the )?(?:provided |supplied |given |approved )?(?:passages?|sources?|"
                           r"knowledge base|kb|documents?)\b|\W*(?:ال)?(?:مصدر|مصادر|فقرات|نصوص)", re.I)
_STRONG_BREAK = re.compile(r"[;؛]|\b(?:however|although)\b")
_CLAUSE_BREAK = re.compile(r"[,،:]|\bbut\b|(?:لكن|بس)\s")


def declares_not_covered(answer: str) -> bool:
    """True when the answer opens by saying the asked information is not in the source.

    Such an answer is a refusal followed by unrelated facts; the response status is out_of_scope.
    "Brufen contains Ibuprofen. The poem part is not covered." opens with an answer and is kept.
    """
    first = re.split(r"(?<=[.!?؟])\s|\n", answer.strip(), maxsplit=1)[0]
    found = _NOT_COVERED.search(first)
    if not found:
        return False
    before = first[:found.start()]
    if _STRONG_BREAK.search(before):
        return False  # "Cataflam contains Diclofenac; however ...": the answer comes first
    if _META_SUBJECT.match(before):
        return True  # "The passages describe X but do not contain ...": the sentence is about the source
    return not _CLAUSE_BREAK.search(before)


def _quotes(draft: GroundedDraft) -> list[tuple[str, str]]:
    """The draft's citations; if it pasted them into the answer instead, recover them from there.

    Recovered quotes go through exactly the same source check as declared ones.
    """
    if draft.citations:
        return [(item.label, item.quote) for item in draft.citations]
    return [(label, quote) for label, quote in _INLINE_QUOTE.findall(draft.answer)]


def verify(draft: GroundedDraft, passages: list[Passage], entities: EntityIndex,
           language: str = "ar") -> VerifiedAnswer:
    answer = clean_answer(draft.answer, language)
    if not answer:
        raise VerificationError("empty_answer")
    if answer.rstrip().endswith((":", "：")):
        raise VerificationError("incomplete_answer")
    by_label = {p.label: p for p in passages}
    quotes = _quotes(draft)
    if not quotes:
        raise VerificationError("missing_citations")
    citations, cited, dropped = [], {}, 0
    for label, quote in quotes:
        # Try the labelled passage first; a mislabelled but genuine quote is credited to the
        # supplied passage that really contains it.
        candidates = [by_label[label]] if label in by_label else []
        candidates += [p for p in passages if p.label != label]
        passage, snippet = next(((p, s) for p in candidates
                                 if (s := locate_quote(quote, p.chunk["text"]))), (None, None))
        if snippet is None:
            dropped += 1
            continue
        cited[passage.label] = passage
        for run in contiguous_runs(snippet, passage.chunk["text"]):
            citations.append(Citation(doc_id=passage.chunk["doc_id"], title=passage.document["title"],
                                      section=passage.chunk["section"], snippet=run))
    if not citations:
        raise VerificationError("no_valid_citation")

    def evidence_of(chosen):
        return "\n".join(f"{p.chunk['doc_id']}\n{p.chunk['section']}\n{p.chunk['text']}" for p in chosen)

    def names(entity, text):
        return re.search(r"(?<!\w)" + re.escape(entity.key) + r"(?!\w)", entities.view(text))

    for entity in entities.mentions(answer):
        if entity.kind == "class" or names(entity, evidence_of(cited.values())):
            continue
        # Named in a supplied passage the model did not quote: cite the source line(s) naming it.
        # A name that is in no supplied passage is outside knowledge and is never published.
        passage = next((p for p in passages if p.label not in cited and names(entity, p.chunk["text"])), None)
        lines = [line.strip() for line in (passage.chunk["text"].splitlines() if passage else ())
                 if line.strip() and names(entity, line)]
        if not lines:
            raise VerificationError(f"unsupported_medicine:{entity.key}")
        cited[passage.label] = passage
        citations.append(Citation(doc_id=passage.chunk["doc_id"], title=passage.document["title"],
                                  section=passage.chunk["section"], snippet=lines[0]))
    evidence = evidence_of(cited.values())

    for sentence in re.split(r"(?<=[.!?؟])\s+|\n+", answer):
        found = entities.mentions(sentence)
        ingredients = [e for e in found if e.kind == "ingredient"]
        if len(ingredients) != 1:
            continue  # pairing is only unambiguous with a single ingredient in the sentence
        for brand in (e for e in found if e.kind == "brand" and entities.brand_ingredients.get(e.key)):
            if not entities.pairs(brand.key, ingredients[0].key):
                raise VerificationError(f"wrong_pairing:{brand.key}|{ingredients[0].key}")

    unsupported = _numbers(_LIST_MARKER.sub("", answer)) - _numbers(evidence)
    if unsupported:
        raise VerificationError("unsupported_number:" + ",".join(sorted(unsupported)))

    unique = list({(c.doc_id, c.section, c.snippet): c for c in citations}.values())
    return VerifiedAnswer(answer, unique, dropped)


def feedback(error: VerificationError) -> str:
    """One corrective instruction for the model's single retry."""
    reason, _, detail = str(error).partition(":")
    return {
        "missing_citations": "You returned no citations. Put each supporting quote in the citations list "
                             "(label + exact quote), not in the answer text.",
        "no_valid_citation": "None of your quotes was found word for word in the passages. Copy each quote "
                             "exactly from the passage it comes from.",
        "unsupported_medicine": f"Your answer mentions \"{detail}\", but none of your cited passages contains it. "
                                "Add a citation from the passage that contains it, or remove it from the answer.",
        "unsupported_number": f"Your answer contains the number(s) {detail}, which are not in your cited passages. "
                              "Remove them, or cite the passage that states them.",
        "wrong_pairing": f"Your answer links a brand to an active ingredient that the passage tables do not give "
                         f"for it ({detail.replace('|', ' / ')}). Use only the Trade Name | Active Ingredient rows.",
        "incomplete_answer": "Your answer is incomplete. Write the full list of items in the answer field itself.",
        "empty_answer": "Your answer was empty. Write the answer in the answer field.",
    }.get(reason, "Your previous answer could not be verified. Answer again following all rules.") + \
        " Return the complete JSON object again."
