"""Run from any working directory using the project's Python environment."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.ingestion.sections import DEFAULT_MAX_CHARS
from medical_rag.ingestion.pipeline import rebuild_knowledge_base


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the nine approved DOCX sources (no embeddings)")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Soft chunk size in Unicode characters")
    args = parser.parse_args()
    if args.max_chars <= 0:
        parser.error("--max-chars must be positive")
    knowledge_base = PROJECT_ROOT / "data" / "knowledge_base"
    manifest = rebuild_knowledge_base(knowledge_base / "source", knowledge_base, args.max_chars)
    for document in manifest["documents"]:
        print(f"{document['doc_id']}: {document['table_count']} tables, {document['chunk_count']} chunks")
    print(f"Total: {manifest['document_count']} documents, {manifest['table_count']} tables, {manifest['chunk_count']} chunks")
    print("Structural issues: " + ", ".join(f"{manifest['issue_counts'].get(level, 0)} {level}" for level in ("info", "review", "error")))


if __name__ == "__main__":
    main()
