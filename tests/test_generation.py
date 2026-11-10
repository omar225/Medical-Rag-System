"""Ollama chat client: structured output, capability controls, token metrics, failures."""

import json

import httpx
import pytest

from medical_rag.generation.config import ChatSettings
from medical_rag.generation.ollama import GenerationError, OllamaChat


def response_body():
    return {"model": "qwen3.5:4b", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({
                "category": "out_of_scope", "answer": "Not covered.", "citations": []}), "thinking": "never expose this"},
            "prompt_eval_count": 40, "eval_count": 10, "load_duration": 100000000,
            "prompt_eval_duration": 200000000, "eval_duration": 300000000}


def make_client(body=None, info=None, failure=None):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if request.url.path == "/api/show":
            return httpx.Response(200, json=info or {"capabilities": ["completion", "thinking"]})
        if failure: raise failure
        return httpx.Response(200, json=response_body() if body is None else body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaChat("http://localhost:11434", ChatSettings(), client=client), calls


def test_structured_payload_reasoning_controls_and_token_metrics():
    llm, calls = make_client()
    draft, usage = llm.generate([{"role": "user", "content": "Synthetic test"}], 10)
    assert draft.category == "out_of_scope"
    payload = calls[-1]
    assert payload["stream"] is False and payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    assert payload["options"] == {"temperature": 0., "num_predict": 1024, "num_ctx": 8192, "seed": 0}
    assert usage.prompt_tokens == 40 and usage.generated_tokens == 10
    assert usage.load_seconds == .1 and usage.generation_seconds == .3
    assert "thinking" not in draft.model_dump()


def test_model_capabilities_are_read_once_per_client():
    llm, calls = make_client()
    llm.generate([], 10)
    llm.generate([], 10)
    assert sum(1 for call in calls if "messages" not in call) == 1


def test_prompt_that_filled_the_context_is_rejected():
    llm, _ = make_client(body={**response_body(), "prompt_eval_count": 8000})
    with pytest.raises(GenerationError, match="context"):
        llm.generate([], 10)


@pytest.mark.parametrize("info,expected", [
    ({"capabilities": ["completion"]}, None),
    ({"capabilities": ["completion", "thinking"], "thinking": {"values": [False, True], "default": True}}, False),
    ({"capabilities": ["completion", "thinking"], "thinking": {"values": ["low", "high"], "default": "high"}}, "low"),
])
def test_capability_based_controls_for_alternative_models(info, expected):
    llm, calls = make_client(info=info)
    llm.generate([], 10)
    if expected is None: assert "think" not in calls[-1]
    else: assert calls[-1]["think"] == expected


@pytest.mark.parametrize("change", [
    {"done": False}, {"done_reason": "length"}, {"model": "another-model"},
    {"message": {"role": "assistant", "content": "{invalid"}},
    {"message": {"role": "assistant", "content": '{"category":"maybe","answer":"x","citations":[]}'}},
    {"message": {"role": "assistant", "content": '{"category":"answer","answer":"x","citations":[],"extra":1}'}},
    {"message": {"role": "assistant", "content": '{"category":"answer","answer":"x","citations":[{"label":"X1","quote":"q"}]}'}},
    {"message": None},
])
def test_malformed_or_incomplete_outputs_are_operational_failures(change):
    llm, _ = make_client(body={**response_body(), **change})
    with pytest.raises(GenerationError): llm.generate([], 10)


@pytest.mark.parametrize("change,unusable", [
    ({"done_reason": "length"}, True), ({"message": {"role": "assistant", "content": "{invalid"}}, True),
    ({"done": False}, False), ({"model": "another-model"}, False),
])
def test_cut_off_or_malformed_drafts_are_marked_unusable(change, unusable):
    from medical_rag.generation.ollama import UnusableDraft
    llm, _ = make_client(body={**response_body(), **change})
    with pytest.raises(GenerationError) as raised: llm.generate([], 10)
    assert isinstance(raised.value, UnusableDraft) is unusable


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("timeout"), httpx.ConnectError("offline")])
def test_transport_failures_raise_generation_error(failure):
    llm, _ = make_client(failure=failure)
    with pytest.raises(GenerationError): llm.generate([], 10)


def test_missing_model_reports_operational_failure_and_never_pulls():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(404, json={"error": "model not found"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        llm = OllamaChat("http://localhost:11434", ChatSettings(), client=client)
        with pytest.raises(GenerationError): llm.generate([], 10)
    assert paths == ["/api/show"]


@pytest.mark.parametrize("settings", [
    {"model": " "}, {"temperature": float("nan")}, {"temperature": 3},
    {"max_output_tokens": 0}, {"context_size": 100}, {"timeout_seconds": 0},
    {"request_timeout_seconds": float("inf")},
])
def test_invalid_chat_configuration(settings):
    with pytest.raises(ValueError): ChatSettings(**settings)
