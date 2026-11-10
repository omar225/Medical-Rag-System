"""Medicine names taken from the approved KB: brands, ingredients and drug classes.

Used to (1) pull the chunk(s) that define a named medicine into the LLM context and
(2) check that drug names in a generated answer come from the cited evidence.
It never decides whether a question may be answered.
"""
from dataclasses import dataclass, field
import re

from medical_rag.retrieval.corpus import ApprovedCorpus
from medical_rag.retrieval.normalization import AliasRegistry, normalize_text, tokenize


_INGREDIENT_HEADER = r"(?:active )?ingredients?|المادة الفعالة"
_BRAND_HEADER = r"(?:trade|brand)(?: names?)?|الاسم التجاري"


def ingredient_table_rows(text: str) -> list[str]:
    """Rows of every table whose header declares an active-ingredient column, verbatim.

    A non-table line or another table header ends the current table, so OTC or
    side-effect tables never contribute brand/ingredient pairs.
    """
    rows, active = [], False
    for line in text.splitlines():
        if "|" not in line:
            active = bool(re.fullmatch(_INGREDIENT_HEADER, normalize_text(line)))
            continue
        name, value = line.split("|", 1)
        if re.fullmatch(_INGREDIENT_HEADER, normalize_text(value)):
            active = True
        elif re.fullmatch(_BRAND_HEADER, normalize_text(name)):
            active = False
        elif active and name.strip() and value.strip():
            rows.append(line)
    return rows


def _ingredient_parts(cell: str) -> list[str]:
    """'Diclofenac + Methyl salicylate' -> both; 'Paracetamol – IV' -> 'Paracetamol'."""
    parts = []
    for part in re.split(r"\s\+\s|\+", cell):
        part = re.split(r"\s[–—-]\s|\(", part)[0].strip(" *")
        if len(re.findall(r"[a-zA-Z]", part)) >= 4:
            parts.append(part)
    return parts


def _heading_names(heading: str) -> list[str]:
    """Latin names in a section heading: '2. NSAIDs – Non-Steroidal ...' -> NSAIDs, Non-Steroidal ..."""
    heading = re.sub(r"^\s*\d+[.)]\s*", "", heading)
    names = re.findall(r"\(([A-Za-z][A-Za-z0-9-]*)\)", heading)
    for segment in re.split(r"\s[–—-]\s|[()/]", heading):
        segment = segment.strip(" -&,")
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 &,'-]{2,}", segment) and not re.search(r"\b(?:otc|prescription)\b", segment, re.I):
            names.append(segment)
    for name in list(names):
        if re.fullmatch(r"[A-Z0-9-]+s", name):  # NSAIDs -> NSAID, PPIs -> PPI
            names.append(name[:-1])
        elif name.endswith(" inhibitors") or name.endswith(" blockers"):
            names.append(name[:-1])
    return names


def one_edit(left: str, right: str) -> bool:
    """Exactly one insertion, deletion, substitution or adjacent transposition."""
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diff = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and left[diff[0]] == right[diff[1]] and left[diff[1]] == right[diff[0]])
    short, long = sorted((left, right), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


_ARABIC_SOUNDS = str.maketrans({
    "ب": "b", "ت": "t", "ث": "s", "ج": "g", "ح": "", "خ": "k", "د": "d", "ذ": "z", "ر": "r", "ز": "z",
    "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t", "ظ": "z", "ع": "", "غ": "g", "ف": "f", "ق": "k",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "", "و": "", "ي": "", "ى": "", "ا": "", "أ": "", "إ": "",
    "آ": "", "ء": "", "ؤ": "", "ئ": "", "ة": "", "پ": "b", "ڤ": "f", "چ": "g", "گ": "g",
})
_ARABIC_ARTICLE = re.compile(r"^(?:[وف]?(?:بال|لل|ال|ب|ل))(?=\S{4,})")


def latin_sounds(word: str) -> str:
    """Consonant skeleton of a Latin name: 'Concor' -> 'knkr', 'Ventolin' -> 'fntln'."""
    word = word.lower()
    for a, b in (("ph", "f"), ("sh", "s"), ("ch", "k"), ("th", "t"), ("kh", "k"), ("ck", "k"), ("x", "ks")):
        word = word.replace(a, b)
    word = re.sub(r"c(?=[eiy])", "s", word).translate(str.maketrans("cqvpjw", "kkfbg_"))
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^bdfgklmnrstz]", "", word))


def arabic_sounds(word: str) -> str:
    """Consonant skeleton of an Arabic-script rendering: 'الكونكور' -> 'knkr'."""
    word = _ARABIC_ARTICLE.sub("", word)
    return re.sub(r"(.)\1+", r"\1", word.translate(_ARABIC_SOUNDS))


@dataclass(frozen=True)
class Entity:
    key: str            # normalized form used for matching
    name: str           # display form as written in the source
    kind: str           # "brand" | "ingredient" | "class"


@dataclass(frozen=True)
class Detection:
    entities: tuple[Entity, ...] = ()
    corrections: tuple[tuple[str, Entity], ...] = ()   # (typed token, assumed entity)

    @property
    def all(self) -> tuple[Entity, ...]:
        return self.entities + tuple(entity for _, entity in self.corrections)


@dataclass
class EntityIndex:
    corpus: ApprovedCorpus
    aliases: AliasRegistry
    names: dict[str, Entity] = field(default_factory=dict)
    # entity key -> [(priority, chunk position)], lower priority = more specific chunk
    chunk_refs: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    # brand key -> ingredient component keys from every table row naming that brand
    brand_ingredients: dict[str, set[str]] = field(default_factory=dict)
    brand_forms: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.available = self.aliases.available_for(self.corpus)
        for position, chunk in enumerate(self.corpus.chunks):
            leaf = chunk["section_path"][-1]
            leaf_keys = {normalize_text(name) for name in _heading_names(leaf)}
            for name in _heading_names(leaf):
                self._add(name, "class", position, 0)
            for row in ingredient_table_rows(chunk["text"]):
                brand, ingredient = (cell.strip(" *") for cell in row.split("|", 1))
                self._add(brand, "brand", position, 0)
                self.brand_ingredients.setdefault(normalize_text(brand), set()).update(
                    normalize_text(part) for part in _ingredient_parts(ingredient))
                # "Paracetamol – IV": a dosage-form note, so the brand is not interchangeable with tablets
                self.brand_forms.setdefault(normalize_text(brand), set()).update(
                    normalize_text(note) for note in re.findall(r"\s[–—-]\s*([^+]+)", ingredient))
                for part in _ingredient_parts(ingredient):
                    # The section titled after an ingredient is its primary chunk.
                    self._add(part, "ingredient", position, 0 if normalize_text(part) in leaf_keys else 1)
        # Reviewed aliases need no entries here: view() rewrites them to source names first.
        self.vocabulary = {token for chunk in self.corpus.chunks for token in tokenize(chunk["text"])}
        keys = sorted(self.names, key=lambda key: (-len(key), key))
        self.pattern = re.compile("|".join(r"(?<!\w)" + re.escape(key) + r"(?!\w)" for key in keys))
        self.single_word = {key: entity for key, entity in self.names.items()
                            if " " not in key and len(key) >= 5 and entity.kind != "class"}
        # Arabic-script brand/ingredient renderings without a reviewed alias ("كونكور" -> Concor).
        # Only unambiguous skeletons of 4+ consonants are kept; it is a hint, never a gate.
        by_sound: dict[str, set] = {}
        for key, entity in self.single_word.items():
            by_sound.setdefault(latin_sounds(key), set()).add(entity)
        self.sounds = {sound: found.pop() for sound, found in by_sound.items() if len(sound) >= 4 and len(found) == 1}

    def _add(self, name: str, kind: str, position: int, priority: int):
        key = normalize_text(name)
        if len(key) < 3:
            return
        # A brand row wins over a heading of the same text; ingredients over classes.
        rank = {"brand": 0, "ingredient": 1, "class": 2}
        if key not in self.names or rank[kind] < rank[self.names[key].kind]:
            self.names[key] = Entity(key, name.strip(), kind)
        refs = self.chunk_refs.setdefault(key, [])
        if (priority, position) not in refs:
            refs.append((priority, position))

    def pairs(self, brand_key: str, ingredient_key: str) -> bool:
        """True if a KB table lists the ingredient (or a salt/qualified form of it) for the brand."""
        if brand_key == ingredient_key:
            return True
        return any(re.search(r"(?<!\w)" + re.escape(ingredient_key) + r"(?!\w)", part)
                   or re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", ingredient_key)
                   for part in self.brand_ingredients.get(brand_key, ()))

    def same_ingredient(self, entity: Entity) -> tuple[list[str], list[str]]:
        """(active ingredients, other trade names whose table rows give exactly those ingredients).

        For an ingredient it is the trade names of that ingredient alone. Only brands listed in a
        table the asked medicine also appears in count, so a gel or an IV product is not offered
        for a tablet. Combination products and descriptive rows ("Ibuprofen topical") are left out.
        """
        wanted = (self.brand_ingredients.get(entity.key, set()) if entity.kind == "brand"
                  else {entity.key} if entity.kind == "ingredient" else set())
        if not wanted:
            return [], []
        tables = {position for _, position in self.chunk_refs.get(entity.key, [])}

        def within(a, b):
            return re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b)

        def same(a, b):  # "ibuprofen" ~ "ibuprofen arginine", not "diclofenac sodium" ~ "diclofenac potassium"
            return within(a, b) or within(b, a)

        brands = []
        for key, parts in self.brand_ingredients.items():
            if key == entity.key or not parts or re.search(r"[ء-ي]", self.names[key].name) \
                    or not tables & {position for _, position in self.chunk_refs.get(key, [])} \
                    or self.brand_forms.get(key, set()) != self.brand_forms.get(entity.key, set()):
                continue
            if all(any(same(w, p) for p in parts) for w in wanted) and all(any(same(p, w) for w in wanted) for p in parts):
                if re.search(r"[A-Za-z]", self.names[key].name):
                    brands.append(self.names[key].name)
        core = [key for key in sorted(wanted) if not any(other != key and within(other, key) for other in wanted)]
        return [self.names[key].name if key in self.names else key for key in core], brands

    @staticmethod
    def _same_parts(cell: str, wanted: set[str]) -> bool:
        """True when a table cell gives the same ingredient set as wanted (any salt or form)."""
        parts = [normalize_text(part) for part in _ingredient_parts(cell)]

        def same(a, b):
            return re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b) or re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", a)

        return bool(parts) and all(any(same(w, p) for w in wanted) for p in parts) \
            and all(any(same(w, p) for p in parts) for w in wanted)

    def table_neighbours(self, entity: Entity, exclude: list[str]) -> tuple[str, list[str]]:
        """(section, "Brand (ingredient)" rows) of the first KB table listing the medicine, other
        than the medicine itself and the names in exclude, verbatim from the table."""
        skip = {normalize_text(name) for name in exclude} | {entity.key}
        wanted = (self.brand_ingredients.get(entity.key, set()) if entity.kind == "brand"
                  else {entity.key} if entity.kind == "ingredient" else set())
        for _, position in sorted(self.chunk_refs.get(entity.key, [])):
            chunk = self.corpus.chunks[position]
            rows = [tuple(cell.strip(" *") for cell in row.split("|", 1)) for row in ingredient_table_rows(chunk["text"])]
            if not any(normalize_text(brand) == entity.key for brand, _ in rows) and entity.kind == "brand":
                continue
            others = [f"{brand} ({ingredient})" for brand, ingredient in rows
                      if normalize_text(brand) not in skip and re.search(r"[A-Za-z]", brand)
                      and not self._same_parts(ingredient, wanted)]
            return re.sub(r"^\s*\d+[.)]\s*", "", chunk["section_path"][-1]), others
        return "", []

    def view(self, text: str) -> str:
        """Normalized text with reviewed aliases rewritten to source names and possessives removed."""
        text = re.sub(r"(\w)['’]s\b", r"\1", normalize_text(text))
        return self.aliases.normalize_query(text, self.available).rewritten if text.strip() else text

    def mentions(self, text: str) -> list[Entity]:
        """Every source medicine name in the text, in order of appearance, without duplicates."""
        if not text.strip():
            return []
        found = []
        for match in self.pattern.finditer(self.view(text)):
            entity = self.names[match.group()]
            if entity not in found:
                found.append(entity)
        return found

    def detect(self, question: str) -> Detection:
        entities = self.mentions(question)
        corrections = []
        matched = {token for entity in entities for token in entity.key.split()}
        for token in re.findall(r"[a-z]{5,}", self.view(question)):
            if token in matched or token in self.vocabulary:
                continue
            candidates = {entity for key, entity in self.single_word.items() if len(token) >= 6 and one_edit(token, key)}
            if len(candidates) == 1:
                corrections.append((token, candidates.pop()))
        for token in re.findall(r"[ء-ي]{4,}", self.view(question)):
            entity = self.sounds.get(arabic_sounds(token))
            if entity and token not in self.vocabulary and entity not in entities                     and entity not in (e for _, e in corrections):
                corrections.append((token, entity))
        return Detection(tuple(entities), tuple(corrections))

    def chunks_for(self, entities, limit_per_entity: int = 2) -> list[dict]:
        """Chunks that define the named medicines: their own section first, then table mentions."""
        positions = []
        for entity in entities:
            refs = sorted(self.chunk_refs.get(entity.key, []))
            for _, position in refs[:limit_per_entity]:
                if position not in positions:
                    positions.append(position)
        return [self.corpus.chunks[position] for position in positions]
