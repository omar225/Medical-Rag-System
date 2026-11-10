"""Request flow with the real KB and a scripted model: safety, retrieval, verification, retries."""

import pytest

from medical_rag.generation.ollama import GroundedDraft
from medical_rag.services.sessions import Turn


def chunk_for(corpus, section_tail):
    return next(c for c in corpus.chunks if c["section"].endswith(section_tail))

PARACETAMOL_EFFECTS = [
    "Nausea أو mild skin rash.",
    "أهم خطر هو hepatotoxicity في حالة overdose أو الجمع بين أكثر من دواء يحتوي على Paracetamol.",
]


def draft(answer, quotes, category="answer"):
    return GroundedDraft(category=category, answer=answer,
                         citations=[{"label": label, "quote": quote} for label, quote in quotes])


def prompt_text(service):
    return "\n".join(m["content"] for m in service.generator.calls[-1])


@pytest.mark.parametrize("message,reason", [
    ("I want to die", "emergency"),
    ("عيان عنده ضيق نفس شديد وشفايفه زرقا، أديله Ventolin؟", "emergency"),
    ("How many mg of Voltaren can an adult take per day?", "dosing"),
    ("How do I get high from this medicine?", "dangerous"),
])
def test_red_flags_escalate_before_retrieval_and_generation(service, message, reason):
    run = service.run(message)
    assert run.response.status == "escalated" and run.response.citations == []
    assert run.reason == reason
    assert service.knowledge.retriever.calls == [] and service.generator.calls == []


def test_greeting_is_answered_without_the_model(service):
    run = service.run("good morning")
    assert run.response.status == "out_of_scope" and service.generator.calls == []


def test_brand_question_gets_its_ingredient_section_and_a_paraphrased_answer(service, corpus):
    service.generator.draft = draft(
        "Panadol contains Paracetamol. The source lists nausea or mild skin rash, and hepatotoxicity "
        "with overdose or when combining several Paracetamol products.",
        [("E1", "Panadol | Paracetamol"), ("E1", PARACETAMOL_EFFECTS[0]), ("E1", PARACETAMOL_EFFECTS[1])])
    run = service.run("What are Panadol's side effects?")
    assert run.response.status == "answered", run.reason
    assert run.response.language == "en"
    assert "[E1] KB-03" in prompt_text(service) and "1. Paracetamol" in prompt_text(service)
    paracetamol = chunk_for(corpus, "1. Paracetamol")
    assert all(c.snippet in paracetamol["text"] and c.doc_id == "KB-03" for c in run.response.citations)


def test_arabic_question_with_attached_prefix_finds_the_brand(service):
    service.generator.draft = draft("بروفين فيه Ibuprofen ومن أعراضه Gastric irritation, dyspepsia.",
                                    [("E1", "Brufen | Ibuprofen"), ("E1", "Gastric irritation, dyspepsia.")])
    run = service.run("ايه الأعراض الجانبية لبروفين")
    assert run.response.status == "answered", run.reason
    assert run.response.language == "ar"
    assert "NSAIDs" in prompt_text(service)


def test_quotes_are_matched_tolerantly_but_published_as_source_lines(service, corpus):
    service.generator.draft = draft("Panadol contains Paracetamol.", [("E1", "Panadol - Paracetamol")])
    run = service.run("What is the active ingredient in Panadol?")
    assert run.response.status == "answered"
    assert run.response.citations[0].snippet == "Panadol | Paracetamol"


@pytest.mark.parametrize("answer,quote,reason", [
    ("Panadol contains Paracetamol.", "Panadol | Ibuprofen", "no_valid_citation"),
    ("Panadol contains Diclofenac.", "Panadol | Paracetamol", "unsupported_medicine:diclofenac"),
    ("Take 1000 mg of Panadol.", "Panadol | Paracetamol", "unsupported_number:1000"),
])
def test_ungrounded_drafts_are_not_published(service, answer, quote, reason):
    service.generator.draft = draft(answer, [("E1", quote)])
    run = service.run("Tell me about Panadol")
    assert run.response.status == "out_of_scope" and run.response.citations == []
    assert run.reason == f"unverified:{reason}"


@pytest.mark.parametrize("category,status", [
    ("out_of_scope", "out_of_scope"), ("needs_clinician", "escalated"), ("emergency", "escalated")])
def test_model_categories_map_to_contract_statuses(service, category, status):
    service.generator.draft = draft("Short reason.", [], category=category)
    service.knowledge.retriever.sections = ["Paracetamol"]
    run = service.run("Which drugs are used to treat malaria?")
    assert run.response.status == status and run.response.citations == []
    assert "Short reason." not in run.response.answer  # fixed, reviewed wording only


def test_no_passages_is_out_of_scope_without_the_model(service):
    run = service.run("Which drugs are used to treat malaria?")
    assert run.response.status == "out_of_scope" and service.generator.calls == []


def test_answer_without_citations_is_not_published(service):
    service.generator.draft = draft("Panadol contains Paracetamol.", [])
    assert service.run("What is in Panadol?").response.status == "out_of_scope"


def test_evidence_labels_are_removed_from_the_answer(service):
    service.generator.draft = draft("Panadol contains Paracetamol [E1].", [("E1", "Panadol | Paracetamol")])
    assert service.run("What is in Panadol?").response.answer == "Panadol contains Paracetamol."


def test_follow_up_uses_the_previous_question_for_retrieval_and_context(service):
    service.generator.draft = draft("Severe liver disease.", [("E1", "Severe liver disease.")])
    history = (Turn("What are the side effects of Panadol?", "Nausea or mild skin rash."),)
    run = service.run("and what are its contraindications?", history)
    assert run.response.status == "answered", run.reason
    assert "Panadol" in service.knowledge.retriever.calls[-1]
    assert "Earlier conversation" in prompt_text(service)


def test_typo_is_passed_to_the_model_as_an_assumption(service):
    service.generator.draft = draft("Assuming you mean Panadol: it contains Paracetamol.",
                                    [("E1", "Panadol | Paracetamol")])
    run = service.run("what is in panadoll?")
    assert '"panadoll"' in prompt_text(service) and run.response.status == "answered"


def test_passages_are_whole_chunks_and_capped(service):
    service.knowledge.retriever.sections = ["Asthma", "COPD"]
    service.run("Which inhalers are used for COPD?")
    content = prompt_text(service)
    assert 1 <= content.count("\n[E") + content.startswith("[E") <= 6


def test_escaped_newlines_and_mislabelled_quotes_still_cite_the_real_source(service, corpus):
    service.generator.draft = draft("Brufen contains Ibuprofen; Nurofen also contains Ibuprofen.",
                                    [("E5", "Brufen | Ibuprofen\nNurofen | Ibuprofen")])
    run = service.run("What are the trade names of ibuprofen?")
    assert run.response.status == "answered", run.reason
    snippet = run.response.citations[0].snippet
    assert snippet == "Brufen | Ibuprofen\nNurofen | Ibuprofen"
    assert snippet in chunk_for(corpus, "NSAIDs – Non-Steroidal Anti-Inflammatory Drugs")["text"]


def test_one_corrective_retry_is_attempted(service):
    first = draft("Panadol contains Diclofenac.", [("E1", "Panadol | Paracetamol")])
    second = draft("Panadol contains Paracetamol.", [("E1", "Panadol | Paracetamol")])
    drafts = iter([first, second])
    original = service.generator.generate

    def scripted(messages, timeout):
        service.generator.draft = next(drafts)
        return original(messages, timeout)

    service.generator.generate = scripted
    run = service.run("What is in Panadol?")
    assert run.response.status == "answered" and run.reason == "verified_after_retry"
    assert "diclofenac" in service.generator.calls[-1][-1]["content"]
    assert run.usage.prompt_tokens == 2400


def test_quotes_pasted_into_the_answer_are_recovered_checked_and_removed(service):
    service.generator.draft = draft(
        "Panadol contains Paracetamol.\n\nPassage [E1]: 'Trade Name | Active Ingredient\nPanadol | Paracetamol'", [])
    run = service.run("What is the ingredient in Panadol?")
    assert run.response.status == "answered", run.reason
    assert run.response.answer == "Panadol contains Paracetamol."
    assert run.response.citations[0].snippet == "Trade Name | Active Ingredient\nPanadol | Paracetamol"


def test_english_answers_drop_copied_arabic_glosses_and_pasted_quotes(service):
    service.generator.draft = draft(
        "Panadol contains Paracetamol.\n- Nausea (غثيان)\n\nPassage states: 'Panadol | Paracetamol'.",
        [("E1", "Panadol | Paracetamol")])
    run = service.run("What is in Panadol?")
    assert run.response.answer == "Panadol contains Paracetamol.\n- Nausea"
    assert run.response.language == "en"


def test_a_failed_retry_is_a_safe_refusal_not_an_error(service):
    from medical_rag.generation.ollama import GenerationError
    service.generator.draft = draft("Panadol contains Diclofenac.", [("E1", "Panadol | Paracetamol")])
    original = service.generator.generate
    calls = []

    def flaky(messages, timeout):
        calls.append(1)
        if len(calls) == 2:
            raise GenerationError("timeout")
        return original(messages, timeout)

    service.generator.generate = flaky
    run = service.run("What is in Panadol?")
    assert run.response.status == "out_of_scope" and run.reason.startswith("unverified:")


def test_bare_quoted_line_is_kept_as_content(service):
    service.generator.draft = draft("المادة الفعالة في Panadol هي:\n'Panadol | Paracetamol'", [("E1", "Panadol | Paracetamol")])
    run = service.run("ايه المادة الفعالة في Panadol؟")
    assert run.response.status == "answered", run.reason
    assert run.response.answer.endswith("Panadol | Paracetamol")


def test_brand_linked_to_the_wrong_ingredient_is_rejected(service):
    service.knowledge.retriever.sections = ["Simple Analgesics"]
    service.generator.draft = draft("Ibuprofen is the active ingredient of Brufen and Panadol.",
                                    [("E1", "Panadol | Paracetamol"), ("E1", "Brufen | Ibuprofen")])
    run = service.run("What is the active ingredient in ibuprofen products?")
    assert run.reason.startswith("unverified:wrong_pairing:panadol")


def test_citations_are_contiguous_source_runs_without_bare_headers(service, corpus):
    service.generator.draft = draft("Cataflam contains Diclofenac potassium.",
                                    [("E1", "Trade Name | Active Ingredient\nCataflam | Diclofenac potassium")])
    run = service.run("What is in Cataflam?")
    assert [c.snippet for c in run.response.citations] == ["Cataflam | Diclofenac potassium"]


def test_a_cut_off_first_draft_is_a_safe_refusal_not_an_error(service):
    from medical_rag.generation.ollama import UnusableDraft

    def cut_off(messages, timeout):
        raise UnusableDraft("token limit")

    service.generator.generate = cut_off
    run = service.run("What is in Panadol?")
    assert run.response.status == "out_of_scope" and run.reason == "unverified:unusable_draft"


def test_an_unreachable_model_is_still_an_operational_error(service):
    from medical_rag.generation.ollama import GenerationError

    def offline(messages, timeout):
        raise GenerationError("offline")

    service.generator.generate = offline
    with pytest.raises(GenerationError):
        service.run("What is in Panadol?")


def scripted(service, *drafts):
    queue, original = list(drafts), service.generator.generate

    def generate(messages, timeout):
        item = queue.pop(0)
        if isinstance(item, Exception):
            service.generator.calls.append(messages)
            raise item
        service.generator.draft = item
        return original(messages, timeout)

    service.generator.generate = generate


def test_refusal_of_a_question_naming_a_kb_medicine_is_retried_once(service):
    scripted(service, draft("Not covered.", [], "out_of_scope"),
             draft("Panadol contains Paracetamol. The other part is not covered.", [("E1", "Panadol | Paracetamol")]))
    run = service.run("What is in Panadol, and who won the World Cup?")
    assert run.response.status == "answered" and run.reason == "verified_after_retry"
    assert "Panadol" in service.generator.calls[-1][-1]["content"]


def test_refusal_is_kept_when_the_retry_also_refuses(service):
    scripted(service, draft("Not covered.", [], "out_of_scope"), draft("Not covered.", [], "out_of_scope"))
    run = service.run("Is Panadol used for malaria?")
    assert run.response.status == "out_of_scope" and len(service.generator.calls) == 2


def test_refusal_without_a_named_medicine_is_not_retried(service):
    service.generator.draft = draft("Not covered.", [], "out_of_scope")
    run = service.run("Which drugs treat malaria?")
    assert run.response.status == "out_of_scope" and len(service.generator.calls) <= 1


def test_a_cut_off_draft_is_retried_with_a_shorter_answer_request(service):
    from medical_rag.generation.ollama import UnusableDraft
    scripted(service, UnusableDraft("token limit"), draft("Panadol contains Paracetamol.", [("E1", "Panadol | Paracetamol")]))
    run = service.run("What is in Panadol?")
    assert run.response.status == "answered" and run.reason == "verified_after_retry"
    assert "cut off" in service.generator.calls[-1][-1]["content"]


def test_answer_that_opens_by_saying_the_topic_is_not_covered_is_out_of_scope(service):
    service.generator.draft = draft("The passages do not contain information about the half-life of Panadol. "
                                    "Panadol contains Paracetamol.", [("E1", "Panadol | Paracetamol")])
    run = service.run("What is the half-life of Panadol?")
    assert run.response.status == "out_of_scope" and run.reason == "answer_not_covered"


def test_answer_with_an_uncovered_second_part_is_still_answered(service):
    service.generator.draft = draft("Panadol contains Paracetamol. The passages do not contain information "
                                    "about the World Cup.", [("E1", "Panadol | Paracetamol")])
    assert service.run("What is in Panadol, and who won the World Cup?").response.status == "answered"


def test_a_timed_out_shorter_retry_is_a_safe_refusal(service):
    from medical_rag.generation.ollama import GenerationError, UnusableDraft
    scripted(service, UnusableDraft("token limit"), GenerationError("timeout"))
    run = service.run("What is in Panadol?")
    assert run.response.status == "out_of_scope" and run.reason == "unverified:unusable_draft"
