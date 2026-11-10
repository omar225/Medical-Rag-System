"""Ollama chat client: one non-streaming call with a JSON-schema constrained response."""

from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from medical_rag.generation.config import ChatSettings


Category = Literal["answer", "out_of_scope", "needs_clinician", "emergency"]


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: Annotated[str, StringConstraints(pattern=r"^E[1-9]$")]
    quote: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)]


class GroundedDraft(BaseModel):
    """What the model must return. Field order is generation order."""
    model_config = ConfigDict(extra="forbid", strict=True)
    category: Category
    answer: Annotated[str, StringConstraints(max_length=5000)]
    citations: list[Quote] = Field(max_length=12)


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int
    generated_tokens: int
    load_seconds: float
    prompt_seconds: float
    generation_seconds: float
    wall_seconds: float


class GenerationError(RuntimeError):
    """An operational model failure (unavailable, timeout, malformed output)."""


class UnusableDraft(GenerationError):
    """The model answered, but the output cannot be used (cut off at the token cap, or not valid JSON).

    This is a quality failure, not an outage: the service answers "could not verify" instead of a 500.
    """


def canonical_model(model: str) -> str:
    name = model.strip()
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


class OllamaChat:
    def __init__(self, base_url: str, settings: ChatSettings, *, client: httpx.Client | None = None):
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("Ollama endpoint must be an HTTP or HTTPS URL")
        self.base_url = base_url.rstrip("/")
        self.settings = settings
        self.model = canonical_model(settings.model)
        self._owns_client = client is None
        self._client = client or httpx.Client(trust_env=False)
        self._info: dict | None = None
        self._lock = Lock()

    def close(self):
        if self._owns_client:
            self._client.close()

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        try:
            response = self._client.post(self.base_url + path, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as error:
            raise GenerationError("Local generation timed out") from error
        except (httpx.HTTPError, ValueError) as error:
            raise GenerationError("Local generation is unavailable or returned invalid JSON") from error
        if not isinstance(data, dict) or "error" in data:
            raise GenerationError("Invalid local generation response")
        return data

    def _model_info(self, timeout: float) -> dict:
        """Model capabilities, read once per process (the model does not change while serving)."""
        with self._lock:
            if self._info is None:
                info = self._post("/api/show", {"model": self.model}, timeout)
                if "completion" not in info.get("capabilities", []):
                    raise GenerationError("Configured LLM does not advertise completion capability")
                self._info = info
            return self._info

    def _thinking_control(self, info: dict):
        """Disable (or minimise) reasoning tokens: they cost latency and are never shown."""
        if "thinking" not in info.get("capabilities", []):
            return None
        thinking = info.get("thinking", {})
        values = thinking.get("values", []) if isinstance(thinking, dict) else []
        if not values or False in values:
            return False
        return "low" if "low" in values else thinking.get("default")

    def generate(self, messages: list[dict], timeout: float) -> tuple[GroundedDraft, GenerationUsage]:
        started = perf_counter()
        timeout = min(timeout, self.settings.timeout_seconds)
        info = self._model_info(timeout)
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "format": GroundedDraft.model_json_schema(),
            "options": {"temperature": self.settings.temperature,
                        "num_predict": self.settings.max_output_tokens,
                        "num_ctx": self.settings.context_size, "seed": 0},
        }
        think = self._thinking_control(info)
        if think is not None:
            payload["think"] = think
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0:
            raise GenerationError("Local generation timed out")
        data = self._post("/api/chat", payload, remaining)
        if data.get("done") is True and data.get("done_reason") == "length":
            raise UnusableDraft("Local generation reached its token limit")
        if data.get("done") is not True or data.get("done_reason") != "stop":
            raise GenerationError("Local generation was incomplete")
        if canonical_model(str(data.get("model", ""))) != self.model:
            raise GenerationError("Local generation returned a different model")
        try:
            message = data["message"]
            if message.get("role") != "assistant" or message.get("tool_calls"):
                raise ValueError("Unexpected assistant message")
            draft = GroundedDraft.model_validate_json(message["content"])
            counts = [int(data.get(key, 0)) for key in ("prompt_eval_count", "eval_count")]
            durations = [int(data.get(key, 0)) / 1e9 for key in ("load_duration", "prompt_eval_duration", "eval_duration")]
        except (KeyError, TypeError, AttributeError, ValueError, ValidationError) as error:
            raise UnusableDraft("Malformed structured local generation output") from error
        if counts[0] >= self.settings.context_size - self.settings.max_output_tokens:
            # Ollama silently drops the start of an over-long prompt; never answer from a truncated prompt.
            raise GenerationError("Prompt filled the context window; evidence may have been truncated")
        return draft, GenerationUsage(*counts, *durations, perf_counter() - started)
