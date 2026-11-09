"""One non-streaming HTTPX call with schema validation and private telemetry."""

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from medical_rag.generation.config import ChatSettings
from medical_rag.retrieval.embeddings.ollama import canonical_model


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Support(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: Annotated[str, StringConstraints(pattern=r"^E[1-5]$")]
    quote: Text


class GroundedDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: Annotated[str, StringConstraints(max_length=6000)]
    evidence_sufficient: bool
    support: list[Support] = Field(max_length=6)


@dataclass(frozen=True)
class GenerationUsage:
    prompt_tokens: int
    generated_tokens: int
    load_seconds: float
    prompt_seconds: float
    generation_seconds: float
    wall_seconds: float


class GenerationError(RuntimeError):
    """An operational model failure, never a scope classification."""


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

    def generate(self, messages: list[dict], timeout: float, *,
                 quote_choices: dict[str, list[str]] | None = None) -> tuple[GroundedDraft, GenerationUsage]:
        started = perf_counter()
        timeout = min(timeout, self.settings.timeout_seconds)
        info = self._post("/api/show", {"model": self.model}, timeout)
        capabilities = info.get("capabilities", [])
        if "completion" not in capabilities:
            raise GenerationError("Configured LLM does not advertise completion capability")
        schema = GroundedDraft.model_json_schema()
        if quote_choices:
            # Ingredient table rows are already exact source strings. Constrain the
            # decoder so it cannot reconstruct a non-adjacent header + row quote.
            properties = schema["$defs"]["Support"]["properties"]
            properties["label"]["enum"] = list(quote_choices)
            properties["quote"]["enum"] = list(dict.fromkeys(quote for quotes in quote_choices.values() for quote in quotes))
        prompt_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
        if prompt_bytes + len(json.dumps(schema, ensure_ascii=False).encode("utf-8")) + 512 + self.settings.max_output_tokens > self.settings.context_size:
            raise GenerationError("Structured prompt exceeds the configured context budget")
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "format": schema,
            "options": {"temperature": self.settings.temperature,
                        "num_predict": self.settings.max_output_tokens,
                        "num_ctx": self.settings.context_size, "seed": 0},
        }
        # New runtimes advertise exact controls; older runtimes expose capabilities only.
        thinking = info.get("thinking", {})
        values = thinking.get("values", []) if isinstance(thinking, dict) else []
        if "thinking" in capabilities:
            if not values or False in values:
                payload["think"] = False
            else:
                choice = "low" if "low" in values else thinking.get("default")
                if choice not in values:
                    raise GenerationError("Unsupported advertised reasoning controls")
                payload["think"] = choice
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0:
            raise GenerationError("Local generation timed out")
        data = self._post("/api/chat", payload, remaining)
        if data.get("done") is not True or data.get("done_reason") != "stop":
            raise GenerationError("Local generation was incomplete or reached its token limit")
        if data.get("model") != self.model:
            raise GenerationError("Local generation returned a different model")
        try:
            message = data["message"]
            if message.get("role") != "assistant" or message.get("tool_calls"):
                raise ValueError("Unexpected assistant message")
            draft = GroundedDraft.model_validate_json(message["content"])
            counts = [data[key] for key in ("prompt_eval_count", "eval_count")]
            durations = [data[key] for key in ("load_duration", "prompt_eval_duration", "eval_duration")]
            if any(type(value) is not int or value < 0 for value in counts + durations):
                raise ValueError("Invalid token/duration measurements")
        except (KeyError, TypeError, AttributeError, ValueError, ValidationError) as error:
            raise GenerationError("Malformed structured local generation output") from error
        # message.thinking is intentionally neither retained nor logged.
        return draft, GenerationUsage(*counts, *(value / 1e9 for value in durations), perf_counter() - started)
