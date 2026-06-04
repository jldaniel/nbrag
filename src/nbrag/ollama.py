"""Ollama clients for local embedding and chat models."""

from __future__ import annotations

import http.client
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
DEFAULT_CHAT_MODEL = "granite4.1:3b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_EMBED_RETRIES = 5
DEFAULT_EMBED_RETRY_DELAY_SECONDS = 2.0


class OllamaError(RuntimeError):
    """Base error for Ollama client failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected response."""


@dataclass(frozen=True)
class ChatMessage:
    """A chat message sent to Ollama."""

    role: str
    content: str


@dataclass(frozen=True)
class OllamaUsage:
    """Token and timing metrics from an Ollama API response."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_duration_ns: int | None
    eval_duration_ns: int | None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> OllamaUsage:
        """Parse usage fields from an Ollama chat or embed response."""

        return cls(
            prompt_tokens=_optional_int(response, "prompt_eval_count"),
            completion_tokens=_optional_int(response, "eval_count"),
            total_duration_ns=_optional_int(response, "total_duration"),
            load_duration_ns=_optional_int(response, "load_duration"),
            prompt_eval_duration_ns=_optional_int(response, "prompt_eval_duration"),
            eval_duration_ns=_optional_int(response, "eval_duration"),
        )


@dataclass(frozen=True)
class OllamaChatResult:
    """A non-streaming chat response from Ollama."""

    content: str
    usage: OllamaUsage


@dataclass(frozen=True)
class OllamaEmbedResult:
    """Embeddings and optional usage from an Ollama embed request."""

    embeddings: tuple[tuple[float, ...], ...]
    usage: OllamaUsage | None


class OllamaEmbeddingProvider:
    """Embedding provider backed by Ollama's native `/api/embed` endpoint."""

    name = "ollama_embedding"

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        normalize: bool = True,
        max_retries: int = DEFAULT_EMBED_RETRIES,
        retry_delay_seconds: float = DEFAULT_EMBED_RETRY_DELAY_SECONDS,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.host = _host(host)
        self.timeout_seconds = timeout_seconds
        self.normalize = normalize
        self.max_retries = max(1, max_retries)
        self.retry_delay_seconds = retry_delay_seconds

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed one text string."""

        return self.embed_many((text,)).embeddings[0]

    def embed_many(self, texts: Sequence[str]) -> OllamaEmbedResult:
        """Embed one or more text strings with one Ollama request (with retries)."""

        if not texts:
            return OllamaEmbedResult(embeddings=(), usage=None)

        last_error: OllamaError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._embed_many_once(texts)
            except OllamaError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay_seconds * attempt)
        assert last_error is not None
        raise last_error

    def _embed_many_once(self, texts: Sequence[str]) -> OllamaEmbedResult:
        response = _post_json(
            self.host,
            "/api/embed",
            {
                "model": self.model,
                "input": list(texts),
            },
            timeout_seconds=self.timeout_seconds,
        )
        embeddings = _embeddings(response)
        if len(embeddings) != len(texts):
            msg = f"expected {len(texts)} embeddings from Ollama, received {len(embeddings)}"
            raise OllamaResponseError(msg)
        if not self.normalize:
            normalized = embeddings
        else:
            normalized = tuple(_normalize(embedding) for embedding in embeddings)
        usage = OllamaUsage.from_response(response)
        has_usage = any(
            value is not None
            for value in (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_duration_ns,
            )
        )
        return OllamaEmbedResult(
            embeddings=normalized,
            usage=usage if has_usage else None,
        )


class OllamaChatClient:
    """Small non-streaming chat client for Ollama's native `/api/chat` endpoint."""

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.host = _host(host)
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> OllamaChatResult:
        """Return a non-streaming assistant response and usage metrics."""

        response = _post_json(
            self.host,
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": message.role, "content": message.content} for message in messages
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout_seconds=self.timeout_seconds,
        )
        message = response.get("message")
        if not isinstance(message, dict):
            msg = "Ollama chat response did not include a message object"
            raise OllamaResponseError(msg)
        content = message.get("content")
        if not isinstance(content, str):
            msg = "Ollama chat response did not include string message content"
            raise OllamaResponseError(msg)
        return OllamaChatResult(
            content=content.strip(),
            usage=OllamaUsage.from_response(response),
        )

    def chat_text(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
    ) -> str:
        """Return assistant text only (convenience wrapper)."""

        return self.chat(messages, temperature=temperature).content


def _host(host: str | None) -> str:
    return (host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)).rstrip("/")


def _post_json(
    host: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    parsed = urlsplit(host)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        msg = f"OLLAMA_HOST must be an http(s) URL, got {host!r}"
        raise ValueError(msg)

    body = json.dumps(payload).encode("utf-8")
    endpoint = f"{parsed.path.rstrip('/')}{path}" if parsed.path else path
    connection: http.client.HTTPConnection | http.client.HTTPSConnection
    connection_type = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout_seconds)

    try:
        connection.request(
            "POST",
            endpoint,
            body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        response_body = response.read().decode("utf-8")
    except OSError as exc:
        msg = (
            f"could not connect to Ollama at {host}. If Ollama is running on the host Mac "
            "from a dev container, try OLLAMA_HOST=http://host.docker.internal:11434 "
            "or OLLAMA_HOST=http://host.orb.internal:11434."
        )
        raise OllamaConnectionError(msg) from exc
    finally:
        connection.close()

    if response.status >= 400:
        msg = f"Ollama returned HTTP {response.status} for {path}: {response_body}"
        raise OllamaResponseError(msg)

    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        msg = f"Ollama returned invalid JSON for {path}: {response_body}"
        raise OllamaResponseError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Ollama returned a non-object JSON response for {path}"
        raise OllamaResponseError(msg)
    return data


def _embeddings(response: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
    raw_embeddings = response.get("embeddings")
    if raw_embeddings is None and "embedding" in response:
        raw_embeddings = [response["embedding"]]
    if not isinstance(raw_embeddings, list):
        msg = "Ollama embed response did not include an embeddings list"
        raise OllamaResponseError(msg)
    return tuple(_embedding(item) for item in raw_embeddings)


def _embedding(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        msg = "each Ollama embedding must be a non-empty list"
        raise OllamaResponseError(msg)
    values: list[float] = []
    for value in raw:
        if not isinstance(value, int | float):
            msg = "Ollama embedding values must be numeric"
            raise OllamaResponseError(msg)
        values.append(float(value))
    return tuple(values)


def _normalize(vector: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return tuple(value / norm for value in vector)


def _optional_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if isinstance(value, int):
        return value
    return None
