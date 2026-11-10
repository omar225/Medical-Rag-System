"""Run a labelled question set through the chatbot and score it.

    python scripts/evaluate.py                         # dev + fresh sets, in-process
    python scripts/evaluate.py --sets chat_fresh       # one set
    python scripts/evaluate.py --url http://localhost:8000   # against a running server (e.g. Docker)

Scores per case: status (expected or listed as acceptable), fact coverage (each fact group needs one
of its alternatives in the answer), citation authenticity (every snippet exists in the cited KB
document), latency and token usage. Writes reports/evaluation/<name>.json and .md.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.corpus import load_bootstrap_corpus  # noqa: E402
from medical_rag.retrieval.normalization import normalize_text  # noqa: E402

DATASETS = ROOT / "tests/datasets"
FORBIDDEN = {"retrieval_heldout", "retrieval_queries"}  # reserved benchmark/holdout: never run here


def in_process_client():
    from medical_rag.api.main import build_service
    from medical_rag.generation.config import ChatSettings
    from medical_rag.services.sessions import Turn

    service = build_service(ChatSettings.from_env())

    def ask(question, previous):
        history, answers = [], []
        for turn in previous:
            run = service.run(turn, tuple(history))
            history.append(Turn(turn, run.response.answer))
        run = service.run(question, tuple(history))
        usage = run.usage
        return run.response.model_dump(), {"reason": run.reason, "prompt_tokens": usage.prompt_tokens if usage else 0,
                                           "generated_tokens": usage.generated_tokens if usage else 0}
    return ask


def http_client(url):
    import httpx
    client = httpx.Client(timeout=60, trust_env=False)

    def ask(question, previous):
        session = str(uuid.uuid4())
        for turn in previous:
            client.post(f"{url}/chat", json={"session_id": session, "message": turn})
        response = client.post(f"{url}/chat", json={"session_id": session, "message": question})
        if response.status_code != 200:
            return {"status": f"http_{response.status_code}", "answer": "", "citations": [], "language": ""}, {}
        return response.json(), {}
    return ask


def score(case, response, documents_text):
    acceptable = case.get("acceptable") or [case["status"]]
    answer = normalize_text(response.get("answer", ""))
    facts = case.get("fact_groups") or []
    covered = sum(any(normalize_text(alt) in answer for alt in group) for group in facts)
    citations_ok = all(normalize_text(c["snippet"]) in documents_text.get(c["doc_id"], "")
                       for c in response.get("citations", []))
    return {"status_ok": response.get("status") in acceptable,
            "facts": [covered, len(facts)] if facts and response.get("status") == "answered" else None,
            "citations_ok": citations_ok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["chat_dev", "chat_fresh"])
    parser.add_argument("--url", help="evaluate a running server instead of the in-process service")
    parser.add_argument("--name", default="results")
    args = parser.parse_args()
    if FORBIDDEN & set(args.sets):
        sys.exit("Refusing to run the reserved retrieval benchmark/holdout.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    corpus = load_bootstrap_corpus(ROOT / "data/knowledge_base")
    documents_text = {}
    for chunk in corpus.chunks:
        documents_text[chunk["doc_id"]] = documents_text.get(chunk["doc_id"], "") + "\n" + normalize_text(chunk["text"])
    ask = http_client(args.url.rstrip("/")) if args.url else in_process_client()

    results = []
    for name in args.sets:
        for case in json.loads((DATASETS / f"{name}.json").read_text(encoding="utf-8"))["cases"]:
            started = time.perf_counter()
            try:
                response, meta = ask(case["question"], case.get("turns", []))
            except Exception as error:  # an operational failure is a scored failure, not a crash
                response, meta = {"status": "error", "answer": "", "citations": []}, {"reason": type(error).__name__}
            seconds = time.perf_counter() - started
            result = {"set": name, "id": case["id"], "question": case["question"], "expected": case["status"],
                      "acceptable": case.get("acceptable"), "status": response.get("status"),
                      "language": response.get("language"), "seconds": round(seconds, 2), **meta,
                      **score(case, response, documents_text), "answer": response.get("answer", ""),
                      "citations": response.get("citations", [])}
            results.append(result)
            mark = "ok " if result["status_ok"] else "BAD"
            print(f"{mark} {name}/{case['id']:<18} {result['status']:<13} {meta.get('reason', ''):<34} "
                  f"{seconds:5.1f}s  {case['question'][:70]}", flush=True)

    summary = summarize(results)
    out = ROOT / "reports/evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"summary": summary, "results": results},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{args.name}.md").write_text(markdown(summary, results), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def summarize(results):
    summary = {}
    for name in sorted({r["set"] for r in results}) + ["all"]:
        rows = [r for r in results if name in ("all", r["set"])]
        answered_expected = [r for r in rows if r["expected"] == "answered"]
        facts = [r["facts"] for r in rows if r["facts"]]
        seconds = sorted(r["seconds"] for r in rows)
        summary[name] = {
            "cases": len(rows),
            "status_accuracy": round(sum(r["status_ok"] for r in rows) / len(rows), 3),
            "answered_when_expected": round(sum(r["status"] == "answered" for r in answered_expected)
                                            / max(1, len(answered_expected)), 3),
            "fact_coverage": round(sum(c for c, _ in facts) / max(1, sum(t for _, t in facts)), 3),
            "citations_authentic": round(sum(r["citations_ok"] for r in rows) / len(rows), 3),
            "errors": sum(r["status"] in (None, "error") or str(r["status"]).startswith("http_") for r in rows),
            "latency_p50_s": statistics.median(seconds),
            "latency_p95_s": seconds[max(0, int(len(seconds) * 0.95) - 1)],
            "mean_prompt_tokens": round(statistics.mean([r.get("prompt_tokens", 0) for r in rows]), 1),
            "mean_generated_tokens": round(statistics.mean([r.get("generated_tokens", 0) for r in rows]), 1),
        }
    return summary


def markdown(summary, results):
    lines = ["# Evaluation results", "", "| set | cases | status acc. | answered when expected | fact coverage | "
             "citations authentic | errors | p50 s | p95 s |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in summary.items():
        lines.append(f"| {name} | {s['cases']} | {s['status_accuracy']:.1%} | {s['answered_when_expected']:.1%} | "
                     f"{s['fact_coverage']:.1%} | {s['citations_authentic']:.1%} | {s['errors']} | "
                     f"{s['latency_p50_s']:.1f} | {s['latency_p95_s']:.1f} |")
    lines += ["", "## Status mismatches", "", "| set | id | expected | got | reason | question |", "| --- | --- | --- | --- | --- | --- |"]
    for r in results:
        if not r["status_ok"]:
            lines.append(f"| {r['set']} | {r['id']} | {r['expected']} | {r['status']} | {r.get('reason', '')} | "
                         f"{r['question'].replace('|', '/')[:90]} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
