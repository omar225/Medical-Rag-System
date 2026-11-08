"""Environment configuration shared by the retrieval developer commands."""

from dataclasses import dataclass
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RetrievalSettings:
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "bge-m3"
    timeout_seconds: float = 180.0
    batch_size: int = 8
    index_dir: Path = PROJECT_ROOT / "storage" / "vector_store" / "dense"
    alias_file: Path = PROJECT_ROOT / "data" / "knowledge_base" / "metadata" / "retrieval_aliases.json"

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        settings = cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            model=os.getenv("EMBEDDING_MODEL", cls.model),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
            index_dir=Path(os.getenv("RETRIEVAL_INDEX_DIR", str(cls.index_dir))),
            alias_file=Path(os.getenv("RETRIEVAL_ALIASES_FILE", str(cls.alias_file))),
        )
        if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be finite and positive")
        if settings.batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        return settings
