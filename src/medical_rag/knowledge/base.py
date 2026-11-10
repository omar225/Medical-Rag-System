"""The approved knowledge base loaded once per process: corpus, index, BM25, entities."""
import logging
from time import monotonic, sleep

from medical_rag.generation.prompt import Passage
from medical_rag.knowledge.entities import Detection, EntityIndex
from medical_rag.retrieval.config import PROJECT_ROOT, RetrievalSettings
from medical_rag.retrieval.corpus import ApprovedCorpus, load_bootstrap_corpus
from medical_rag.retrieval.embeddings import EmbeddingError, OllamaEmbeddings
from medical_rag.retrieval.normalization import AliasRegistry
from medical_rag.retrieval.dense import DenseRetriever
from medical_rag.retrieval.hybrid import HybridRetriever
from medical_rag.retrieval.lexical import LexicalRetriever
from medical_rag.retrieval.numpy_index import DenseIndexError, build_index, load_index


logger = logging.getLogger(__name__)

HYBRID_TOP_K = 5
MAX_PASSAGES = 6


class KnowledgeBase:
    def __init__(self, corpus: ApprovedCorpus, aliases: AliasRegistry, retriever):
        self.corpus, self.aliases, self.retriever = corpus, aliases, retriever
        self.entities = EntityIndex(corpus, aliases)
        self.documents = corpus.documents

    @classmethod
    def load(cls, settings: RetrievalSettings, provider: OllamaEmbeddings, *, build_wait_seconds: float = 0):
        """Verify the approved corpus once, make sure a matching index exists, warm BM25."""
        corpus = load_bootstrap_corpus(PROJECT_ROOT / "data/knowledge_base")
        aliases = AliasRegistry.from_file(settings.alias_file)
        ensure_index(corpus, provider, settings, build_wait_seconds)
        dense = DenseRetriever(provider, settings.index_dir, lambda: corpus)
        lexical = LexicalRetriever(settings.index_dir, settings.model, lambda: corpus, lambda: aliases)
        lexical.search("warm up", 1)  # builds the BM25 statistics now rather than on the first request
        return cls(corpus, aliases, HybridRetriever(dense, lexical))

    def retrieve(self, query: str, detection: Detection) -> list[Passage]:
        """Chunks defining the named medicines first, then hybrid results, without duplicates."""
        chunks = self.entities.chunks_for(detection.all)
        for result in self.retriever.search(query, top_k=HYBRID_TOP_K):
            chunks.append(result["chunk"])
        unique = list({chunk["chunk_id"]: chunk for chunk in chunks}.values())[:MAX_PASSAGES]
        return [Passage(f"E{rank}", chunk, self.documents[chunk["doc_id"]]) for rank, chunk in enumerate(unique, 1)]


def ensure_index(corpus: ApprovedCorpus, provider: OllamaEmbeddings, settings: RetrievalSettings,
                 wait_seconds: float = 0) -> None:
    """Use the stored index if it matches the corpus and model; otherwise build it.

    In Docker the embedding model may still be loading, so building retries for up to wait_seconds.
    """
    try:
        load_index(settings.index_dir, corpus, settings.model)
        return
    except DenseIndexError as error:
        logger.warning("Dense index unusable (%s); building it", type(error).__name__)
    deadline = monotonic() + wait_seconds
    while True:
        try:
            build_index(corpus, provider, settings.index_dir, batch_size=settings.batch_size)
            logger.info("Dense index built for %d chunks", len(corpus.chunks))
            return
        except EmbeddingError:
            if monotonic() >= deadline:
                raise
            logger.info("Embedding service not ready; retrying index build")
            sleep(5)
