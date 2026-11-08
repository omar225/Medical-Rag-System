"""Conservative text views and reviewed same-name/term aliases, never answers."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import unicodedata

from medical_rag.retrieval.corpus import ApprovedCorpus, canonical_hash
from medical_rag.retrieval.embeddings.ollama import validate_text


NORMALIZATION_VERSION = "nfkc-arabic-aliases-v1"
_ORTHOGRAPHY = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ـ": "", "‐": "-", "‑": "-", "–": "-", "—": "-"})


def normalize_text(text: str) -> str:
    """Keep words, negations, drug qualifiers, and numbers; preserve ة versus ه."""
    text = unicodedata.normalize("NFKC", text).casefold().translate(_ORTHOGRAPHY)
    characters = []
    for character in text:
        if unicodedata.category(character) == "Mn" and "\u0600" <= character <= "\u08ff":
            continue
        if unicodedata.category(character) == "Cf":
            continue
        if unicodedata.category(character) == "Nd":
            character = str(unicodedata.decimal(character))
        characters.append(character)
    text = "".join(characters)
    # Separate attached Arabic/Latin scripts without changing either word's letters.
    text = re.sub(r"([\u0621-\u064a])([a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([\u0621-\u064a])", r"\1 \2", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize_text(text), flags=re.UNICODE)


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    normalized: str
    rewritten: str
    applied_aliases: tuple[str, ...]
    ambiguous_aliases: tuple[str, ...]
    unavailable_aliases: tuple[str, ...]
    normalization_id: str

    def metadata(self) -> dict:
        return asdict(self)


class AliasRegistry:
    def __init__(self, entries: list[dict]):
        if not isinstance(entries, list):
            raise ValueError("Aliases must be a reviewed list")
        self.entries = entries
        ids = set()
        variants = {}
        self.by_id = {}
        for entry in entries:
            if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) and entry[key].strip() for key in ("id", "canonical", "review_note")):
                raise ValueError("Each alias requires an ID, canonical source term, and review note")
            if entry.get("status") not in {"confirmed", "ambiguous"} or entry["id"] in ids:
                raise ValueError("Alias IDs must be unique and statuses confirmed or ambiguous")
            if not isinstance(entry.get("variants"), list) or not all(isinstance(v, str) and v.strip() for v in entry["variants"]):
                raise ValueError("Alias variants must be nonblank strings")
            ids.add(entry["id"])
            self.by_id[entry["id"]] = entry
            for variant in ([entry["canonical"]] if entry["status"] == "confirmed" else []) + entry["variants"]:
                normalized = normalize_text(variant)
                if normalized in variants and variants[normalized]["id"] != entry["id"]:
                    raise ValueError(f"Conflicting reviewed alias: {variant}")
                variants[normalized] = entry
        ordered = sorted(variants, key=lambda value: (-len(value), value))
        self.pattern = re.compile("|".join(
            f"(?P<a{index}>(?<!\\w)" + ("(?:ال)?" if re.match(r"[\u0621-\u064a]", variant) else "") + re.escape(variant) + r"(?!\w))"
            for index, variant in enumerate(ordered)
        )) if ordered else None
        self.groups = {f"a{index}": variants[variant] for index, variant in enumerate(ordered)}
        self.fingerprint = canonical_hash({"version": NORMALIZATION_VERSION, "entries": entries})

    @classmethod
    def from_file(cls, path: Path) -> "AliasRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported alias registry format")
        return cls(data["entries"])

    def available_for(self, corpus: ApprovedCorpus) -> frozenset[str]:
        """Presence validates the target vocabulary; review notes establish intended identity."""
        source = normalize_text("\n".join(chunk["text"] for chunk in corpus.chunks))
        return frozenset(entry["id"] for entry in self.entries if re.search(
            r"(?<!\w)" + re.escape(normalize_text(entry["canonical"])) + r"(?!\w)", source
        ))

    def normalize_query(self, question: str, available: frozenset[str]) -> NormalizedQuery:
        validate_text(question)
        normalized = normalize_text(question)
        if not normalized:
            raise ValueError("Question contains no text after Unicode normalization")
        applied, ambiguous, unavailable = set(), set(), set()

        def replace(match):
            entry = self.groups[match.lastgroup]
            if entry["status"] == "ambiguous":
                ambiguous.add(entry["id"])
                return match.group()
            if entry["id"] not in available:
                unavailable.add(entry["id"])
                return match.group()
            replacement = normalize_text(entry["canonical"])
            if replacement != match.group():
                applied.add(entry["id"])
            return replacement

        rewritten = self.pattern.sub(replace, normalized) if self.pattern else normalized
        return NormalizedQuery(question, normalized, rewritten, tuple(sorted(applied)), tuple(sorted(ambiguous)), tuple(sorted(unavailable)), self.fingerprint)
