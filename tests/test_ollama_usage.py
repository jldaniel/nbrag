"""Tests for Ollama usage parsing (no live server)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nbrag.ollama import (
    OllamaEmbeddingProvider,
    OllamaEmbedResult,
    OllamaResponseError,
    OllamaUsage,
)


def test_usage_from_chat_response() -> None:
    raw = {
        "message": {"role": "assistant", "content": "hello"},
        "prompt_eval_count": 42,
        "eval_count": 7,
        "total_duration": 1_000_000_000,
        "load_duration": 100_000,
        "prompt_eval_duration": 200_000,
        "eval_duration": 800_000,
    }
    usage = OllamaUsage.from_response(raw)

    assert usage.prompt_tokens == 42
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 49
    assert usage.total_duration_ns == 1_000_000_000


def test_usage_from_minimal_response() -> None:
    usage = OllamaUsage.from_response({})

    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None


def test_chat_result_total_tokens() -> None:
    usage = OllamaUsage.from_response(
        {"prompt_eval_count": 10, "eval_count": 5},
    )
    assert usage.total_tokens == 15


def test_embed_many_retries_on_failure() -> None:
    provider = OllamaEmbeddingProvider(max_retries=3, retry_delay_seconds=0)
    calls = {"count": 0}

    def fake_embed(texts: object) -> OllamaEmbedResult:
        calls["count"] += 1
        if calls["count"] < 3:
            raise OllamaResponseError("EOF")
        return OllamaEmbedResult(embeddings=((1.0, 0.0),), usage=None)

    with patch.object(provider, "_embed_many_once", side_effect=fake_embed):
        result = provider.embed_many(("hello",))

    assert calls["count"] == 3
    assert result.embeddings == ((1.0, 0.0),)


def test_embed_many_raises_after_retries_exhausted() -> None:
    provider = OllamaEmbeddingProvider(max_retries=2, retry_delay_seconds=0)

    with (
        patch.object(provider, "_embed_many_once", side_effect=OllamaResponseError("EOF")),
        pytest.raises(OllamaResponseError),
    ):
        provider.embed_many(("hello",))
