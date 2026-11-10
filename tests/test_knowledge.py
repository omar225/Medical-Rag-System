"""Medicine-name index, alternatives lookup, quote location and the not-covered check."""

from pathlib import Path

import pytest

from medical_rag.generation.verification import locate_quote
from medical_rag.knowledge.entities import EntityIndex, ingredient_table_rows, one_edit
from medical_rag.retrieval.corpus import load_bootstrap_corpus
from medical_rag.retrieval.normalization import AliasRegistry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def entities():
    corpus = load_bootstrap_corpus(ROOT / "data/knowledge_base")
    return EntityIndex(corpus, AliasRegistry.from_file(ROOT / "data/knowledge_base/metadata/retrieval_aliases.json"))


@pytest.mark.parametrize("question,name,kind", [
    ("What are Panadol's side effects?", "Panadol", "brand"),
    ("what r the side effects of brufen", "Brufen", "brand"),
    ("ايه الأعراض الجانبية لبروفين", "Brufen", "brand"),
    ("معلومات عن البنادول", "Panadol", "brand"),
    ("What is in Panadol Extra?", "Panadol Extra", "brand"),
    ("contraindications of ARB", "ARBs", "class"),
    ("side effects of PPIs", "PPIs", "class"),
    ("Is metformin contraindicated in kidney disease?", "Metformin", "ingredient"),
])
def test_names_are_found_in_natural_phrasings(entities, question, name, kind):
    found = entities.detect(question).entities
    assert found and found[0].name == name and found[0].kind == kind


def test_unknown_words_are_not_entities_and_typos_are_only_suggestions(entities):
    assert entities.detect("Which drugs treat malaria?").all == ()
    detection = entities.detect("side effects of panadoll")
    assert detection.entities == () and detection.corrections[0][1].name == "Panadol"


def test_brand_pulls_its_ingredient_section_first(entities):
    chunks = entities.chunks_for(entities.detect("Brufen").entities)
    assert "NSAIDs" in chunks[0]["section"]


def test_only_ingredient_tables_define_brand_rows():
    text = "Trade Name | Active Ingredient\nPanadol | Paracetamol\n\nProduct | OTC\nB.B.C Spray | OTC\n"
    assert ingredient_table_rows(text) == ["Panadol | Paracetamol"]


@pytest.mark.parametrize("left,right,expected", [
    ("panadoll", "panadol", True), ("brufin", "brufen", True), ("burfen", "brufen", True),
    ("panadol", "panadol", False), ("cetal", "total", False)])
def test_one_edit(left, right, expected):
    assert one_edit(left, right) is expected


SOURCE = "Trade Name | Active Ingredient\nPanadol | Paracetamol\nCommon Side Effects:\nNausea أو mild skin rash.\n" \
         "أهم خطر هو hepatotoxicity في حالة overdose."


@pytest.mark.parametrize("quote,expected", [
    ("Panadol | Paracetamol", "Panadol | Paracetamol"),
    ("Panadol – Paracetamol", "Panadol | Paracetamol"),
    ("nausea او mild skin rash", "Nausea أو mild skin rash."),
    ("Nausea أو mild skin rash. أهم خطر هو hepatotoxicity في حالة overdose.",
     "Nausea أو mild skin rash.\nأهم خطر هو hepatotoxicity في حالة overdose."),
    ("Panadol | Ibuprofen", None),
    ("severe rash", None),
    ("a", None),
])
def test_locate_quote_returns_genuine_source_lines(quote, expected):
    assert locate_quote(quote, SOURCE) == expected


@pytest.mark.parametrize("question,name", [
    ("كونكور ده دوا ايه", "Concor"),
    ("ايه البدائل التجارية للكونكور", "Concor"),
    ("سينجولير فيه ايه", "Singulair"),
])
def test_arabic_spelling_of_a_latin_brand_is_a_hint(entities, question, name):
    detection = entities.detect(question)
    assert detection.entities == () and detection.corrections[0][1].name == name


@pytest.mark.parametrize("question", ["عايز اعرف الجرعة للحامل", "المريض عنده الضغط والسكر", "الاعراض الجانبية"])
def test_ordinary_arabic_words_do_not_sound_like_brands(entities, question):
    assert entities.detect(question).corrections == ()


def test_medicine_named_in_an_uncited_supplied_passage_is_cited_from_it(entities):
    from medical_rag.generation.ollama import GroundedDraft
    from medical_rag.generation.prompt import Passage
    from medical_rag.generation.verification import VerificationError, verify

    chunks = entities.corpus.chunks
    asthma = next(c for c in chunks if "Singulair" in c["text"] and "Montelosab" not in c["text"])
    allergy = next(c for c in chunks if "Montelosab" in c["text"])
    passages = [Passage("E1", asthma, {"title": "Asthma"}), Passage("E2", allergy, {"title": "Allergy"})]
    draft = GroundedDraft(category="answer", answer="Singulair and Montelosab contain Montelukast.",
                          citations=[{"label": "E1", "quote": "Singulair | Montelukast"}])
    result = verify(draft, passages, entities, "en")
    assert any(c.snippet == "Montelosab | Montelukast" and c.doc_id == allergy["doc_id"] for c in result.citations)

    # A name that no supplied passage contains is still rejected.
    draft = GroundedDraft(category="answer", answer="Singulair and Montelosab contain Montelukast.",
                          citations=[{"label": "E1", "quote": "Singulair | Montelukast"}])
    with pytest.raises(VerificationError, match="unsupported_medicine:montelosab"):
        verify(draft, passages[:1], entities, "en")


@pytest.mark.parametrize("answer,expected", [
    ("The provided passages do not contain information about the half-life of Losartan.", True),
    ("لا توجد معلومات عن الملاريا في المصدر.", True),
    ("Brufen contains Ibuprofen. The passages do not contain information to write a poem.", False),
    ("Cataflam contains Diclofenac; however, the knowledge base does not cover malaria.", False),
    ("No. The source does not list malaria among the uses of Panadol.", False),
])
def test_declares_not_covered(answer, expected):
    from medical_rag.generation.verification import declares_not_covered
    assert declares_not_covered(answer) is expected


@pytest.mark.parametrize("answer,expected", [
    ("The provided passages describe the uses and trade names of mebeverine but do not contain information "
     "regarding its pharmacokinetics.", True),
    ("لا يوجد معلومة عن سعر زيرتك في المصدر، لكن المصدر يذكر أن Zyrtec يحتوي على Cetirizine.", True),
    ("The passage states that Curam contains Amoxicillin + Clavulanic acid; it does not contain information "
     "about the exchange rate.", False),
])
def test_declares_not_covered_reads_the_sentence_subject(answer, expected):
    from medical_rag.generation.verification import declares_not_covered
    assert declares_not_covered(answer) is expected


def test_alternatives_are_same_ingredient_brands_from_a_shared_table(entities):
    brufen = entities.detect("Brufen").entities[0]
    ingredients, brands = entities.same_ingredient(brufen)
    assert ingredients == ["Ibuprofen"]
    assert {"Nurofen", "Ibuflam", "Profinal"} <= set(brands)
    assert "Brufen" not in brands and "Cataflam" not in brands
    assert not any("Cetafen" in brand for brand in brands)  # combination product, not the same ingredient

    # A gel or an IV product is not offered for tablets.
    _, brands = entities.same_ingredient(entities.detect("Cataflam").entities[0])
    assert "Olfen" in brands and "Olfen Gel" not in brands
    _, brands = entities.same_ingredient(entities.detect("Panadol").entities[0])
    assert "Cetal" in brands and "Perfalgan" not in brands


@pytest.mark.parametrize("question,expected", [
    ("what are the alternatives of brufen", True),
    ("What can I use instead of Brufen?", True),
    ("which brands have the same active ingredient as Lasix", True),
    ("ايه البدائل التجارية لـ Brufen ؟", True),
    ("ايه بديل النيكسيوم", True),
    ("What are the side effects of Brufen?", False),
    ("البروفين بيستخدم في ايه", False),
])
def test_asks_for_alternatives(question, expected):
    from medical_rag.generation.prompt import asks_for_alternatives
    assert asks_for_alternatives(question) is expected


def test_alternatives_note_lists_only_brands_in_the_selected_passages(entities):
    from medical_rag.generation.prompt import Passage, build_messages

    chunk = next(c for c in entities.corpus.chunks if "Brufen | Ibuprofen" in c["text"] and "Flabu" not in c["text"])
    passages = [Passage("E1", chunk, {"title": "Pain"})]
    messages, _ = build_messages("ايه بدائل البروفين", "ar", passages, arabic_names=("Brufen",),
                                 alternatives=(("Brufen", ["Ibuprofen"], ["Nurofen", "Flabu"]),),
                                 context_size=8192, max_output_tokens=1024)
    user = messages[-1]["content"]
    assert "نفس المادة الفعالة: Nurofen." in user  # Flabu is not in the passage the model sees
    assert "named in Arabic script here is: Brufen" in user


def test_other_rows_of_the_medicines_table_have_a_different_ingredient(entities):
    lasix = entities.detect("Lasix").entities[0]
    _, brands = entities.same_ingredient(lasix)
    assert "Furosemide-Alex" in brands  # a brand named after its ingredient is still a brand
    section, others = entities.table_neighbours(lasix, brands)
    assert section == "Loop Diuretics" and "Burinex (Bumetanide)" in others
    assert not any("Furosemide" in row for row in others)

    augmentin = entities.detect("Augmentin").entities[0]
    _, others = entities.table_neighbours(augmentin, entities.same_ingredient(augmentin)[1])
    assert "Amoxil (Amoxicillin)" in others  # amoxicillin alone is a different ingredient set
    assert not any("Clavulanic" in row for row in others)

    panadol = entities.detect("Panadol").entities[0]
    _, others = entities.table_neighbours(panadol, entities.same_ingredient(panadol)[1])
    assert not any("Perfalgan" in row for row in others)  # same ingredient, IV form
