from copy import deepcopy
import math
from pathlib import Path

import httpx
import pytest

from medical_rag.retrieval.corpus import load_bootstrap_corpus, prepare_approved_corpus
from medical_rag.retrieval.normalization import AliasRegistry, normalize_text, tokenize
from medical_rag.retrieval.search.dense import DenseRetriever
from medical_rag.retrieval.search.hybrid import HybridRetriever, reciprocal_rank_fusion
from medical_rag.retrieval.search.lexical import BM25, LexicalRetriever
from medical_rag.retrieval.vector_store.numpy_index import StaleIndexError, WrongModelError, build_index


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reviewed_aliases():
    registry = AliasRegistry.from_file(ROOT / "data/knowledge_base/metadata/retrieval_aliases.json")
    corpus = load_bootstrap_corpus(ROOT / "data/knowledge_base")
    available = registry.available_for(corpus)
    assert len(available) == len(registry.entries)
    return registry, available


def test_unicode_arabic_case_whitespace_and_digits():
    assert normalize_text("  إِيبُوبروفينـ  ＭＥＴＦＯＲＭＩＮ  ٥٠٠\nMG ") == "ايبوبروفين metformin 500 mg"
    assert normalize_text("أ إ آ ٱ ى") == "ا ا ا ا ي"
    assert normalize_text("حساسية") != normalize_text("حساسيه")
    assert normalize_text("الـMetformin") == "ال metformin"
    assert normalize_text("مش ممنوع؛ NOT contraindicated") == "مش ممنوع؛ not contraindicated"
    assert tokenize("XR 500 mg + B12 7alat 3ala") == ["xr", "500", "mg", "b12", "7alat", "3ala"]


@pytest.mark.parametrize("text,target", [
    ("كــتافلام", "cataflam"),
    ("المَادّة الفعّالة", "active ingredient"),
    ("el mada el fa3ala", "active ingredient"),
    ("حاصرات مستقبلات الأنجيوتنسين ٢", "arbs"),
    ("إيه الأعراض الجانبية للسالبوتامول؟", "side effects"),
    ("الميتفورمين", "metformin"),
    ("a3rad ganbeya", "side effects"),
])
def test_reviewed_query_aliases_preserve_original(text, target, reviewed_aliases):
    registry, available = reviewed_aliases
    query = registry.normalize_query(text, available)
    assert query.original == text
    assert target in query.rewritten
    assert query.applied_aliases
    assert query.normalization_id == registry.fingerprint


def test_aliases_do_not_answer_ingredient_questions(reviewed_aliases):
    registry, available = reviewed_aliases
    query = registry.normalize_query("كتافلام", available)
    assert query.rewritten == "cataflam"
    assert "diclofenac" not in query.rewritten
    assert registry.normalize_query("اوجمنتين", available).rewritten == "augmentin"


def test_qualified_products_doses_and_negation_are_preserved(reviewed_aliases):
    registry, available = reviewed_aliases
    query = registry.normalize_query("مش بانادول اكسترا ٥٠٠ mg XR", available)
    assert query.rewritten == "مش panadol extra 500 mg xr"
    assert "brand-panadol-extra" in query.applied_aliases
    assert "brand-panadol" not in query.applied_aliases
    assert registry.normalize_query("Glucophage XR", available).rewritten == "glucophage xr"


@pytest.mark.parametrize("text", ["دواء الضغط", "7obob el daght", "mamno3", "مش ممنوع"])
def test_ambiguous_aliases_are_reported_not_expanded(text, reviewed_aliases):
    registry, available = reviewed_aliases
    query = registry.normalize_query(text, available)
    assert query.ambiguous_aliases
    assert query.rewritten == normalize_text(text)
    assert not query.applied_aliases


def test_absent_canonical_targets_are_not_injected(reviewed_aliases):
    registry, _ = reviewed_aliases
    query = registry.normalize_query("كتافلام", frozenset())
    assert query.rewritten == "كتافلام"
    assert query.unavailable_aliases == ("brand-cataflam",)


def test_conflicting_aliases_and_unreviewed_status_are_rejected():
    entry = {"id": "one", "canonical": "Alpha", "variants": ["اسم"], "status": "confirmed", "review_note": "Test fixture"}
    with pytest.raises(ValueError, match="Conflicting"):
        AliasRegistry([entry, {**entry, "id": "two", "canonical": "Beta"}])
    with pytest.raises(ValueError, match="statuses"):
        AliasRegistry([{**entry, "status": "automatic"}])


def test_bm25_rarity_formula_and_no_overlap(synthetic_corpus):
    scorer = BM25(synthetic_corpus, AliasRegistry([]))
    scores = scorer.scores("Alpha")
    assert scores[0] == pytest.approx(math.log(1 + (3 - 1 + .5) / (1 + .5)))
    assert scores[1] == scores[2] == 0
    assert scorer.scores("fixture")[0] < scores[0]
    assert not scorer.scores("NoSuchToken987").any()


def make_lexical(directory, corpus, provider, alias_loader=None):
    build_index(corpus, provider, directory)
    return LexicalRetriever(directory, "bge-m3", lambda: corpus, alias_loader or (lambda: AliasRegistry([])))


def test_lexical_ranking_ties_source_metadata_and_no_http(tmp_path, synthetic_corpus, synthetic_provider, monkeypatch):
    retriever = make_lexical(tmp_path, synthetic_corpus, synthetic_provider)
    monkeypatch.setattr(httpx.Client, "request", lambda *args, **kwargs: pytest.fail("Lexical mode must not contact Ollama"))
    assert [r["chunk"]["chunk_id"] for r in retriever.search("fixture", 3)] == ["a-1", "a-2", "b-1"]
    result = retriever.search("Alpha", 1)[0]
    assert result["chunk"] == synthetic_corpus.chunks[0]
    assert result["document"] == synthetic_corpus.documents["test-A"]
    assert result["retrieval"]["score_type"] == "bm25"
    assert retriever.search("NoSuchToken987", 5) == []
    assert synthetic_provider.questions == []


def test_changed_aliases_invalidate_the_derived_cache(tmp_path, synthetic_corpus, synthetic_provider):
    registry = [AliasRegistry([])]
    retriever = make_lexical(tmp_path, synthetic_corpus, synthetic_provider, lambda: registry[0])
    assert retriever.search("اسم") == []
    registry[0] = AliasRegistry([{"id": "alpha", "canonical": "Alpha", "variants": ["اسم"], "status": "confirmed", "review_note": "Synthetic same-name fixture"}])
    result = retriever.search("اسم")[0]
    assert result["chunk"]["chunk_id"] == "a-1"
    assert result["retrieval"]["query"]["applied_aliases"] == ("alpha",)


def test_rrf_formula_deterministic_ties_and_empty_lists():
    ranking = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    assert [row[0] for row in ranking] == ["a", "b"]
    assert ranking[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([[], ["z"]]) == [("z", 1 / 61)]
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a", "a"]])


def test_hybrid_one_embedding_source_preservation_and_baseline_input(tmp_path, synthetic_corpus, synthetic_provider):
    lexical = make_lexical(tmp_path, synthetic_corpus, synthetic_provider)
    dense = DenseRetriever(synthetic_provider, tmp_path, lambda: synthetic_corpus)
    hybrid = HybridRetriever(dense, lexical)
    result = hybrid.search("  ALPHA  ", 1)[0]
    assert synthetic_provider.questions == ["alpha"]
    assert result["chunk"] == synthetic_corpus.chunks[0]
    assert result["document"] == synthetic_corpus.documents["test-A"]
    assert result["retrieval"]["score_type"] == "rrf"
    assert result["retrieval"]["query"]["original"] == "  ALPHA  "
    assert result["retrieval"]["ranks"] == {"dense": 1, "lexical": 1}
    dense.search("  ALPHA  ", 1)
    assert synthetic_provider.questions == ["alpha", "ALPHA"]  # Stage 3 baseline is unchanged.


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
def test_no_stale_build_use_in_new_modes(tmp_path, synthetic_corpus, synthetic_provider, mode):
    build_index(synthetic_corpus, synthetic_provider, tmp_path)
    chunks, documents = deepcopy(synthetic_corpus.chunks), deepcopy(synthetic_corpus.documents)
    documents["test-A"]["build_id"] = "f" * 64
    for chunk in chunks:
        if chunk["doc_id"] == "test-A":
            chunk["build_id"] = "f" * 64
    changed = prepare_approved_corpus(chunks, documents)
    lexical = LexicalRetriever(tmp_path, "bge-m3", lambda: changed, lambda: AliasRegistry([]))
    retriever = lexical if mode == "lexical" else HybridRetriever(DenseRetriever(synthetic_provider, tmp_path, lambda: changed), lexical)
    with pytest.raises(StaleIndexError):
        retriever.search("Alpha")
    assert synthetic_provider.questions == []


def test_wrong_model_and_unapproved_corpus_are_not_bypassed(tmp_path, synthetic_corpus, synthetic_provider):
    build_index(synthetic_corpus, synthetic_provider, tmp_path)
    lexical = LexicalRetriever(tmp_path, "another-model", lambda: synthetic_corpus, lambda: AliasRegistry([]))
    with pytest.raises(WrongModelError):
        lexical.search("Alpha")
    chunks = deepcopy(synthetic_corpus.chunks)
    chunks[0]["doc_id"] = "unapproved"
    lexical = LexicalRetriever(tmp_path, "bge-m3", lambda: prepare_approved_corpus(chunks, synthetic_corpus.documents), lambda: AliasRegistry([]))
    from medical_rag.retrieval.corpus import CorpusError
    with pytest.raises(CorpusError, match="not in the explicitly approved"):
        lexical.search("Alpha")


def test_cross_branch_source_change_is_rejected(tmp_path, synthetic_corpus, synthetic_provider, monkeypatch):
    lexical = make_lexical(tmp_path, synthetic_corpus, synthetic_provider)
    dense = DenseRetriever(synthetic_provider, tmp_path, lambda: synthetic_corpus)
    result = dense.search("Alpha", 1)[0]
    result["chunk"]["text"] = "A different source snapshot"
    monkeypatch.setattr(dense, "search", lambda *args: [result])
    with pytest.raises(StaleIndexError, match="between hybrid branches"):
        HybridRetriever(dense, lexical).search("Alpha")


@pytest.mark.parametrize("mode", ["lexical", "hybrid"])
@pytest.mark.parametrize("question,top_k", [("", 5), (123, 5), ("query", 0), ("query", True)])
def test_invalid_query_rejected_before_corpus_read(tmp_path, synthetic_provider, mode, question, top_k):
    loader = lambda: pytest.fail("Invalid query must not load the corpus")
    lexical = LexicalRetriever(tmp_path, "bge-m3", loader, lambda: AliasRegistry([]))
    retriever = lexical if mode == "lexical" else HybridRetriever(DenseRetriever(synthetic_provider, tmp_path, loader), lexical)
    with pytest.raises(ValueError):
        retriever.search(question, top_k)
