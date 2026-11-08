"""Build, search, or evaluate the local dense index without changing the API."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.retrieval.config import RetrievalSettings
from medical_rag.retrieval.corpus import CorpusError, load_bootstrap_corpus
from medical_rag.retrieval.embeddings.ollama import EmbeddingError, OllamaEmbeddings
from medical_rag.retrieval.evaluation import evaluate_retrieval
from medical_rag.retrieval.search.dense import DenseRetriever
from medical_rag.retrieval.normalization import AliasRegistry
from medical_rag.retrieval.search.lexical import LexicalRetriever
from medical_rag.retrieval.search.hybrid import HybridRetriever, CANDIDATE_COUNT, RRF_K
from medical_rag.retrieval.vector_store.numpy_index import DenseIndexError, build_index, load_index


def print_results(results) -> None:
    for rank, result in enumerate(results, 1):
        chunk = result["chunk"]
        score_type = result.get("retrieval", {}).get("score_type", "cosine")
        print(f"\n{rank}. {score_type}={result['score']:.6f} | {chunk['doc_id']} | {chunk['section']}")
        print(f"chunk_id: {chunk['chunk_id']}")
        print(chunk["text"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Evidence retrieval modes; no generated medical answers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Index exactly the nine approved bootstrap documents")
    search = subparsers.add_parser("search", help="Pass a question or enter questions interactively")
    search.add_argument("--question")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--json", action="store_true", help="Return complete metadata as JSON")
    search.add_argument("--mode", choices=("dense", "lexical", "hybrid"), default="dense")
    evaluation = subparsers.add_parser("evaluate", help="Run the frozen labeled-evidence benchmark")
    evaluation.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "tests/evaluation/datasets/retrieval_queries.json")
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--mode", choices=("dense", "lexical", "hybrid"), default="dense")
    comparison = subparsers.add_parser("compare", help="Compare fixed modes on frozen and held-out fixtures")
    comparison.add_argument("--held-out", type=Path, default=PROJECT_ROOT / "tests/evaluation/datasets/retrieval_heldout.json")
    comparison.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/evaluation/stage31_comparison.json")
    args = parser.parse_args()
    try:
        settings = RetrievalSettings.from_env()
        corpus_loader = lambda: load_bootstrap_corpus(PROJECT_ROOT / "data/knowledge_base")
        corpus = corpus_loader()
        with OllamaEmbeddings(settings.ollama_base_url, settings.model, settings.timeout_seconds) as provider:
            if args.command == "build":
                metadata = build_index(corpus, provider, settings.index_dir, batch_size=settings.batch_size)
                print(json.dumps({
                    "index_dir": str(settings.index_dir),
                    **{key: metadata[key] for key in ("document_count", "chunk_count", "dimension", "embedding", "indexing_seconds", "embedding_seconds", "corpus_sha256")},
                }, indent=2))
                review_count = sum(record.get("requires_review", False) for record in corpus.documents.values())
                if review_count:
                    print(f"Preserved source review flags for {review_count} bootstrap document(s); indexing is not medical approval.")
                return
            dense = DenseRetriever(provider, settings.index_dir, corpus_loader)
            lexical = LexicalRetriever(settings.index_dir, settings.model, corpus_loader, lambda: AliasRegistry.from_file(settings.alias_file))
            modes = {"dense": dense, "lexical": lexical, "hybrid": HybridRetriever(dense, lexical)}
            if args.command == "compare":
                alias_registry = AliasRegistry.from_file(settings.alias_file)
                warmup_started = perf_counter()
                provider.embed_question("Retrieval benchmark warm-up")
                report = {
                    "schema_version": 1, "corpus_sha256": corpus.fingerprint,
                    "embedding": provider.identity(), "normalization_id": alias_registry.fingerprint,
                    "parameters": {"bm25_k1": 1.2, "bm25_b": .75, "rrf_k": RRF_K, "candidate_count": CANDIDATE_COUNT, "selection": "fixed before evaluation; no tuning on held-out results"},
                    "warmup_seconds": perf_counter() - warmup_started,
                    "frozen_inputs": {
                        "aliases_sha256": hashlib.sha256(settings.alias_file.read_bytes()).hexdigest(),
                        "held_out_sha256": hashlib.sha256(args.held_out.read_bytes()).hexdigest(),
                        "frozen_dataset_sha256": hashlib.sha256((PROJECT_ROOT / "tests/evaluation/datasets/retrieval_queries.json").read_bytes()).hexdigest(),
                        "dense_code_sha256": hashlib.sha256((PROJECT_ROOT / "src/medical_rag/retrieval/search/dense.py").read_bytes()).hexdigest(),
                    },
                }
                for name, path in (("frozen", PROJECT_ROOT / "tests/evaluation/datasets/retrieval_queries.json"), ("held_out", args.held_out)):
                    dataset = json.loads(path.read_text(encoding="utf-8"))
                    report[name] = {}
                    for mode, retriever in modes.items():
                        result = evaluate_retrieval(retriever, corpus, dataset)
                        report[name][mode] = result
                        print(f"{name} / {mode}: metrics={result['metrics']}; latency={result['latency_seconds']}", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
                print(f"Full comparison and per-query evidence: {args.output}")
                return
            retriever = modes[args.mode]
            if args.command == "search":
                if args.question is not None:
                    results = retriever.search(args.question, args.top_k)
                    if args.json:
                        print(json.dumps(results, ensure_ascii=False, indent=2))
                    else:
                        print_results(results)
                else:
                    print("Enter a question; an empty line or EOF exits. Scores are similarities, not medical confidence.")
                    while True:
                        try:
                            question = input("\nQuestion: ")
                        except EOFError:
                            break
                        if not question.strip():
                            break
                        print_results(retriever.search(question, args.top_k))
                return
            dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
            index = load_index(settings.index_dir, corpus, settings.model)
            report = evaluate_retrieval(retriever, corpus, dataset)
            report["embedding"] = index.metadata["embedding"]
            report["indexing_seconds"] = index.metadata["indexing_seconds"]
            output = args.output or PROJECT_ROOT / f"reports/evaluation/stage31_{args.mode}.json"
            report["mode"] = args.mode
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            for row in report["queries"]:
                print(f"{row['id']}: {row['latency_seconds']:.3f}s | metrics={row['metrics']}")
            print(json.dumps({key: report[key] for key in ("query_count", "answerable_count", "out_of_kb_count", "metrics", "by_language", "latency_seconds")}, indent=2))
            print(f"Full retrieved evidence and per-query latency: {output}")
    except (CorpusError, EmbeddingError, DenseIndexError, ValueError, OSError) as error:
        parser.exit(1, f"Retrieval error: {error}\n")


if __name__ == "__main__":
    main()
