"""Run the exact POST /chat contract against local models; write a separate report.

No downloads, source/index writes, ranking changes, or application fixture imports.
--cold unloads only the two configured Ollama runners before one cold request.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import threading
from time import perf_counter, sleep

import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.api.main import create_app, get_chat_service
from medical_rag.generation.config import ChatSettings
from medical_rag.retrieval.config import RetrievalSettings
from medical_rag.retrieval.corpus import load_bootstrap_corpus
from medical_rag.retrieval.embeddings.ollama import canonical_model
from medical_rag.retrieval.normalization import normalize_text
from medical_rag.safety.policy import answer_language
from medical_rag.services.chat import open_chat_service


class GPUSampler:
    """Whole-device used VRAM sampled every ~250 ms; not process-attributed memory."""
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=2,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode == 0:
                    self.samples.append((perf_counter(), int(result.stdout.splitlines()[0].strip())))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return
            self._stop.wait(.25)

    def peak_since(self, started):
        values = [value for timestamp, value in self.samples if timestamp >= started]
        return max(values) if values else None

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self.thread.join(timeout=3)


def score_case(case, data, http_status, corpus, retrieved_ids):
    """Independent mechanical checks; fact phrases are spot-checks, not entailment."""
    answered = http_status == 200 and data.get("status") == "answered"
    citations = data.get("citations", [])
    citation_valid = http_status == 200 and bool(citations) == answered
    for citation in citations:
        matches = [chunk for chunk in corpus.chunks if chunk["chunk_id"] in retrieved_ids
                   and chunk["doc_id"] == citation.get("doc_id")
                   and chunk["section"] == citation.get("section")
                   and chunk["title"] == citation.get("title")
                   and citation.get("snippet") and citation["snippet"] in chunk["text"]]
        citation_valid = citation_valid and bool(matches)
    answer = normalize_text(data.get("answer", ""))
    facts = (all(any(normalize_text(phrase) in answer for phrase in group) for group in case["fact_groups"])
             if case["status"] == "answered" else None)
    return {
        "status_match": http_status == 200 and data.get("status") == case["status"],
        "fact_spotcheck": answered and facts if facts is not None else None,
        "citation_valid": citation_valid,
        "language_valid": http_status == 200 and data.get("language") == answer_language(data.get("answer", "")),
        "language_expected": http_status == 200 and data.get("language") in case["languages"],
        "human_medical_correctness_review": "required; automatic phrase checks are insufficient",
    }


def summarize(rows):
    summary = {}
    for group in dict.fromkeys(row['set'] for row in rows):
        subset = [row for row in rows if row["set"] == group]
        supported = [row for row in subset if row["expected_status"] == "answered"]
        negatives = [row for row in subset if row["expected_status"] == "out_of_scope"]
        unsafe = [row for row in subset if row["expected_status"] == "escalated"]
        times = [row["http_seconds"] for row in subset]
        summary[group] = {
            "count": len(subset), "status_matches": sum(row["checks"]["status_match"] for row in subset),
            "supported_count": len(supported),
            "supported_fact_spotchecks_passed": sum(row["checks"]["fact_spotcheck"] is True for row in supported),
            "abstention_cases": len(negatives), "correct_abstentions": sum(row["checks"]["status_match"] for row in negatives),
            "escalation_cases": len(unsafe), "correct_escalations": sum(row["checks"]["status_match"] for row in unsafe),
            "citation_checks_passed": sum(row["checks"]["citation_valid"] for row in subset),
            "language_checks_passed": sum(row["checks"]["language_valid"] and row["checks"]["language_expected"] for row in subset),
            "http_500_count": sum(row["http_status"] == 500 for row in subset),
            "http_seconds_median": median(times) if times else None,
            "http_seconds_max": max(times) if times else None,
            "within_30_seconds": sum(value < 30 for value in times),
            "false_abstentions": sum(row['response'].get('status') == 'out_of_scope' for row in supported),
            "answered_on_negative_label": sum(row['response'].get('status') == 'answered' for row in negatives),
            "conversation_or_clarification": sum((row['measurements'] or {}).get('reason') in
                {'conversation', 'clarify_topic', 'clarify_spelling', 'clarify_medicine'} for row in subset),
            "discarded_paraphrase_fallbacks": sum((row['measurements'] or {}).get('reason') == 'verified_excerpts' for row in subset),
            "lexical_fallbacks": sum((row['measurements'] or {}).get('lexical_fallback', False) for row in subset),
            "prompt_tokens": sum(((row['measurements'] or {}).get('usage') or {}).get('prompt_tokens', 0) for row in subset),
            "generated_tokens": sum(((row['measurements'] or {}).get('usage') or {}).get('generated_tokens', 0) for row in subset),
        }
    return summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "tests/evaluation/datasets/chat_stage4a.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation/stage4a_chat.json")
    parser.add_argument("--cold", action="store_true", help="Unload configured runners before one extra cold request")
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    settings, retrieval = ChatSettings.from_env(), RetrievalSettings.from_env()
    corpus = load_bootstrap_corpus(ROOT / "data/knowledge_base")
    app = create_app()
    captures = []

    def observed_service():
        capture = {"run": None, "retrieved_ids": []}
        captures.append(capture)
        with open_chat_service(settings) as service:
            original_search = service.retriever.search

            def search(question, top_k):
                results = original_search(question, top_k)
                capture["retrieved_ids"] = [result["chunk"]["chunk_id"] for result in results]
                return results

            service.retriever.search = search
            if service.fallback_retriever is not None:
                original_fallback = service.fallback_retriever.search

                def fallback(question, top_k):
                    results = original_fallback(question, top_k)
                    capture['retrieved_ids'].extend(result['chunk']['chunk_id'] for result in results)
                    return results

                service.fallback_retriever.search = fallback

            class Observed:
                def run(self, request):
                    run = service.run(request)
                    capture["run"] = run
                    return run

            yield Observed()

    app.dependency_overrides[get_chat_service] = observed_service
    with httpx.Client(base_url=retrieval.ollama_base_url, timeout=30, trust_env=False) as ollama:
        tags = ollama.get("/api/tags").json()
        installed = {model["name"]: model for model in tags["models"]}
        version = ollama.get("/api/version").json()
        info = ollama.post("/api/show", json={"model": settings.model}).json()
        index = json.loads((retrieval.index_dir / "index.json").read_text(encoding="utf-8"))
        report = {
            "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "ollama": version,
            "model": installed.get(settings.model, installed.get(settings.model + ":latest")),
            "capabilities": info.get("capabilities"), "settings": settings.__dict__,
            "embedding": index["embedding"], "corpus_sha256": corpus.fingerprint,
            "vectors_sha256": index["vectors_sha256"],
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "source_code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for directory in ("generation", "safety", "services", "api")
                                   for path in sorted((ROOT / "src/medical_rag" / directory).rglob("*.py"))},
            "measurement_notes": ["In-process FastAPI TestClient HTTP boundary; local transport overhead included, no external network client.",
                                  "Development set, not blinded validation. Phrase-based fact checks do not establish clinical correctness.",
                                  "GPU values are sampled whole-device used VRAM, include other processes, and can miss short peaks.",
                                  "Cold means configured runners unloaded; OS/disk/driver/kernel caches are not cleared."],
        }
        with GPUSampler() as gpu, TestClient(app) as client:
            def evaluate(case):
                started = perf_counter()
                response = client.post("/chat", json={"message": case["question"]})
                elapsed = perf_counter() - started
                data = response.json()
                capture = captures[-1]
                run = capture["run"]
                row = {"id": case["id"], "set": case["set"], "question": case["question"],
                       "expected_status": case["status"], "expected_languages": case["languages"],
                       "http_status": response.status_code, "response": data,
                       "http_seconds": elapsed, "peak_gpu_used_mib": gpu.peak_since(started),
                       "measurements": run.measurements() if run else None,
                       "retrieved_chunk_ids": capture["retrieved_ids"],
                       "checks": score_case(case, data, response.status_code, corpus, capture["retrieved_ids"])}
                print(case["id"], response.status_code, data.get("status"), f"{elapsed:.3f}s", row["checks"], flush=True)
                return row

            if args.cold:
                for model in (settings.model, retrieval.model):
                    result = ollama.post("/api/generate", json={"model": model, "keep_alive": 0})
                    result.raise_for_status()
                targets = {canonical_model(settings.model), canonical_model(retrieval.model)}
                deadline = perf_counter() + 10
                while True:
                    state = ollama.get("/api/ps").json()
                    resident = {canonical_model(model["name"]) for model in state["models"]}
                    if not resident & targets:
                        break
                    if perf_counter() >= deadline:
                        raise RuntimeError("Configured runners did not unload; cannot claim a cold measurement")
                    sleep(.1)
                report["cold_runner_state_before"] = state
                report["cold"] = evaluate(dataset["cases"][0])
            rows = [evaluate(case) for case in dataset["cases"]]
            report["cases"] = rows
            report["summary"] = summarize(rows)
            report["peak_gpu_used_mib"] = max((value for _, value in gpu.samples), default=None)
            report["runner_state_after"] = ollama.get("/api/ps").json()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Report:", args.output)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
