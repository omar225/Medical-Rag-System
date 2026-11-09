"""Safety -> existing hybrid -> source sufficiency -> one LLM -> verification."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

from medical_rag.api.schemas import ChatRequest, ChatResponse
from medical_rag.generation.config import ChatSettings
from medical_rag.generation.evidence import EvidenceChecks, EvidenceRejected, ParaphraseRejected, build_messages
from medical_rag.generation.ollama import GenerationError, GenerationUsage, OllamaChat
from medical_rag.generation.formatting import format_verified_excerpts, format_verified_answer
from medical_rag.services.understanding import QuestionUnderstanding, conversational_reply, strip_greeting
from medical_rag.retrieval.normalization import normalize_text
from medical_rag.retrieval.config import PROJECT_ROOT, RetrievalSettings
from medical_rag.retrieval.corpus import ApprovedCorpus, load_bootstrap_corpus
from medical_rag.retrieval.embeddings.ollama import OllamaEmbeddings
from medical_rag.retrieval.normalization import AliasRegistry
from medical_rag.retrieval.search.dense import DenseRetriever
from medical_rag.retrieval.search.hybrid import HybridRetriever
from medical_rag.retrieval.search.lexical import LexicalRetriever
from medical_rag.safety.policy import answer_language, preferred_language, refusal_message, safety_category


@dataclass(frozen=True)
class ChatRun:
    response: ChatResponse
    reason: str
    elapsed_seconds: float
    retrieval_seconds: float
    usage: GenerationUsage | None = None
    lexical_fallback: bool = False

    def measurements(self):
        return {"reason": self.reason, "elapsed_seconds": self.elapsed_seconds,
                "retrieval_seconds": self.retrieval_seconds,
                "lexical_fallback": self.lexical_fallback,
                "usage": asdict(self.usage) if self.usage else None}


class ChatService:
    def __init__(self, retriever, generator, corpus_loader: Callable[[], ApprovedCorpus],
                 alias_loader: Callable[[], AliasRegistry], settings: ChatSettings, fallback_retriever=None):
        self.retriever, self.generator = retriever, generator
        self.corpus_loader, self.alias_loader, self.settings = corpus_loader, alias_loader, settings
        self.fallback_retriever = fallback_retriever

    def run(self, request: ChatRequest) -> ChatRun:
        started = perf_counter()
        preferred = preferred_language(request.message)
        retrieval_seconds = 0.0
        usage = None
        correction = None
        lexical_fallback = False

        def remaining():
            seconds = self.settings.request_timeout_seconds - (perf_counter() - started)
            if seconds <= 0:
                raise GenerationError("Chat response deadline exceeded")
            return seconds

        def finish(answer, citations, status, reason):
            remaining()
            if correction and status == 'answered':
                original, target = correction
                note = (f'I interpreted “{original}” as “{target}”. If that is not the name you meant, please clarify.'
                        if preferred == 'en' else f'فهمت إن المقصود بـ «{original}» هو «{target}». لو تقصد اسم تاني، وضّحه من فضلك.')
                answer = note + '\n\n' + answer
            response = ChatResponse(answer=answer, citations=citations, status=status,
                                    language=answer_language(answer))
            return ChatRun(response, reason, perf_counter() - started, retrieval_seconds, usage, lexical_fallback)

        category = safety_category(request.message)
        if category:
            return finish(refusal_message(preferred, emergency=category == "emergency", safety=True),
                          [], "escalated", category)
        friendly = conversational_reply(request.message, preferred)
        if friendly:
            return finish(friendly, [], 'out_of_scope', 'conversation')
        # Loader/index/embedding errors propagate to the HTTP 500 boundary.
        corpus = self.corpus_loader()
        checks = EvidenceChecks(corpus, self.alias_loader())
        understanding = QuestionUnderstanding(checks)
        normalized = checks.aliases.normalize_query(request.message, checks.available)
        if any(checks.intent(checks.aliases.by_id[key]['canonical']) is None
               for key in normalized.ambiguous_aliases):
            answer = ('Please give the full medicine name and your question; that description can refer to different medicines.'
                      if preferred == 'en' else 'ممكن تكتب اسم الدواء كامل مع سؤالك؟ الوصف ده ممكن يقصد أكتر من دواء.')
            return finish(answer, [], 'out_of_scope', 'clarify_medicine')
        friendly = understanding.clarify_topic(request.message, preferred)
        if friendly:
            return finish(friendly, [], 'out_of_scope', 'clarify_topic')
        stripped = strip_greeting(request.message)
        question = stripped if stripped != normalize_text(request.message) else request.message
        question, correction, suggestions = understanding.spelling(question)
        if suggestions:
            names = ', '.join(suggestions)
            answer = (f'Please confirm the medicine name. Did you mean {names}? Include the full name and your question.'
                      if preferred == 'en' else f'ممكن تأكد اسم الدواء؟ هل تقصد {names}؟ اكتب الاسم كامل مع سؤالك.')
            return finish(answer, [], 'out_of_scope', 'clarify_spelling')
        retrieval_started = perf_counter()
        results = self.retriever.search(question, top_k=5)
        retrieval_seconds = perf_counter() - retrieval_started
        remaining()
        evidence = checks.admit(question, checks.validate_results(results))
        if (not evidence and self.fallback_retriever is not None and checks.intent(question)
                and any(checks.subjects(question) == checks.words(name) for name in understanding.names)):
            lexical_fallback = True
            fallback_started = perf_counter()
            results = self.fallback_retriever.search(checks.view(question), top_k=5)
            retrieval_seconds += perf_counter() - fallback_started
            evidence = checks.admit(question, checks.validate_results(results))
            remaining()
        if not evidence:
            return finish(refusal_message(preferred), [], "out_of_scope", "no_supported_passages")
        messages, evidence = build_messages(question, preferred, evidence,
                                            self.settings.context_size, self.settings.max_output_tokens)
        choices = None
        if checks.intent(question) == "ingredient":
            subjects = checks.subjects(question)
            choices = {item.label: checks.ingredient_rows(item, subjects) for item in evidence}
            if not all(choices.values()):
                return finish(refusal_message(preferred), [], "out_of_scope", "no_explicit_ingredient_row")
        elif checks.intent(question) == 'brands':
            choices = {item.label: checks.brand_quotes(item, checks.subjects(question)) for item in evidence}
            if not all(choices.values()):
                return finish(refusal_message(preferred), [], 'out_of_scope', 'no_explicit_brand_rows')
        draft, usage = self.generator.generate(messages, timeout=remaining(), quote_choices=choices)
        remaining()
        # Re-read active source/build identities immediately before citation publication.
        current = self.corpus_loader()
        if current.fingerprint != corpus.fingerprint:
            from medical_rag.retrieval.vector_store.numpy_index import StaleIndexError
            raise StaleIndexError("Approved corpus changed during generation")
        try:
            citations = checks.verify(question, draft, evidence)
        except ParaphraseRejected as rejected:
            answer = format_verified_excerpts(checks, question, rejected.excerpts, preferred)
            return finish(answer, rejected.citations, "answered", "verified_excerpts")
        except EvidenceRejected:
            return finish(refusal_message(preferred), [], "out_of_scope", "unverified_answer")
        # Generation must not smuggle clinical advice through an information request.
        # Source qualifiers such as "high doses" or "after the last dose" are
        # general warnings, not a personalized dosing instruction.
        if safety_category(draft.answer) in {"personal", "dangerous"}:
            return finish(refusal_message(preferred, safety=True), [], "escalated", "unsafe_generation")
        return finish(format_verified_answer(draft.answer), citations, "answered", "verified")


@contextmanager
def open_chat_service(settings: ChatSettings):
    """Request-scoped clients; never builds an index, downloads a model, or stores a session."""
    retrieval = RetrievalSettings.from_env()
    loader = lambda: load_bootstrap_corpus(PROJECT_ROOT / "data/knowledge_base")
    aliases = lambda: AliasRegistry.from_file(retrieval.alias_file)
    provider = OllamaEmbeddings(retrieval.ollama_base_url, retrieval.model,
                                min(retrieval.timeout_seconds, settings.request_timeout_seconds))
    generator = OllamaChat(retrieval.ollama_base_url, settings)
    try:
        dense = DenseRetriever(provider, retrieval.index_dir, loader)
        lexical = LexicalRetriever(retrieval.index_dir, retrieval.model, loader, aliases)
        yield ChatService(HybridRetriever(dense, lexical), generator, loader, aliases, settings, lexical)
    finally:
        generator.close()
        provider.close()
