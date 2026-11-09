"""Small, explicit generation limits, separate from frozen retrieval settings."""

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class ChatSettings:
    model: str = "qwen3.5:4b"
    temperature: float = 0.0
    max_output_tokens: int = 768
    context_size: int = 8192
    timeout_seconds: float = 25.0
    request_timeout_seconds: float = 30.0

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("LLM_MODEL must be nonblank")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if type(self.max_output_tokens) is not int or not 128 <= self.max_output_tokens <= 4096:
            raise ValueError("LLM_MAX_OUTPUT_TOKENS must be between 128 and 4096")
        if type(self.context_size) is not int or not 4096 <= self.context_size <= 32768:
            raise ValueError("LLM_CONTEXT_SIZE must be between 4096 and 32768")
        if self.max_output_tokens >= self.context_size:
            raise ValueError("Output cap must be smaller than the context size")
        for value in (self.timeout_seconds, self.request_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Chat timeouts must be finite and positive")

    @classmethod
    def from_env(cls):
        return cls(
            model=os.getenv("LLM_MODEL", cls.model).strip(),
            temperature=float(os.getenv("LLM_TEMPERATURE", str(cls.temperature))),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", str(cls.max_output_tokens))),
            context_size=int(os.getenv("LLM_CONTEXT_SIZE", str(cls.context_size))),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", str(cls.timeout_seconds))),
            request_timeout_seconds=float(os.getenv("CHAT_TIMEOUT_SECONDS", str(cls.request_timeout_seconds))),
        )
