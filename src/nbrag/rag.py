"""Local RAG index helpers for notebook chunks."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nbrag.chunking import ChunkingStrategy
from nbrag.models import Chunk, SearchResult
from nbrag.ollama import (
    ChatMessage,
    OllamaChatClient,
    OllamaEmbeddingProvider,
    OllamaUsage,
)
from nbrag.parser import parse_notebook

INDEX_FORMAT = "nbrag.local_vector_index"
INDEX_VERSION = 2
DEFAULT_CONTEXT_CHAR_LIMIT = 8_000


@dataclass(frozen=True)
class IndexedChunk:
    """A retrieval chunk and its persisted embedding."""

    chunk: Chunk
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class LocalVectorIndex:
    """A simple inspectable vector index stored on disk as JSON."""

    source: str
    notebook_paths: tuple[str, ...]
    chunker: str
    embedding_model: str
    embedding_dimensions: int
    created_at: str
    chunks: tuple[IndexedChunk, ...]

    def search(
        self, query_embedding: tuple[float, ...], *, top_k: int = 4
    ) -> tuple[SearchResult, ...]:
        """Search saved chunk vectors with a normalized query embedding."""

        if top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)
        if len(query_embedding) != self.embedding_dimensions:
            msg = (
                f"query embedding has {len(query_embedding)} dimensions, "
                f"index expects {self.embedding_dimensions}"
            )
            raise ValueError(msg)

        ranked = sorted(
            (
                (item.chunk, _dot(query_embedding, item.embedding))
                for item in self.chunks
                if item.embedding
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return tuple(
            SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
                source=f"local_vector_index:{self.embedding_model}",
            )
            for rank, (chunk, score) in enumerate(ranked, start=1)
        )


@dataclass(frozen=True)
class AnswerMetrics:
    """Timing and token usage for one RAG answer."""

    retrieve_ms: float
    generate_ms: float
    total_ms: float
    embed_usage: OllamaUsage | None
    chat_usage: OllamaUsage | None
    context_chars: int


@dataclass(frozen=True)
class RagAnswer:
    """A generated answer and the retrieval results used to produce it."""

    question: str
    answer: str
    results: tuple[SearchResult, ...]
    metrics: AnswerMetrics | None = None


def build_index(
    notebook_paths: Sequence[Path],
    chunker: ChunkingStrategy,
    embedder: OllamaEmbeddingProvider,
    *,
    source: str = "",
    batch_size: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> LocalVectorIndex:
    """Parse notebooks, chunk them, embed chunks, and return a local vector index."""

    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)

    documents = tuple(parse_notebook(path) for path in notebook_paths)
    chunks = tuple(chunk for document in documents for chunk in chunker.chunk(document))
    embeddings: list[tuple[float, ...]] = []
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if on_progress is not None:
            on_progress(end, total)
        batch = chunks[start:end]
        result = embedder.embed_many(tuple(chunk.text for chunk in batch))
        embeddings.extend(result.embeddings)

    if len(embeddings) != len(chunks):
        msg = f"embedded {len(embeddings)} vectors for {len(chunks)} chunks"
        raise ValueError(msg)

    embedding_dimensions = _embedding_dimensions(embeddings)
    return LocalVectorIndex(
        source=source,
        notebook_paths=tuple(str(path) for path in notebook_paths),
        chunker=chunker.name,
        embedding_model=embedder.model,
        embedding_dimensions=embedding_dimensions,
        created_at=datetime.now(tz=UTC).isoformat(),
        chunks=tuple(
            IndexedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ),
    )


def write_index(index: LocalVectorIndex, path: str | Path) -> None:
    """Write a local vector index as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_index_to_dict(index), indent=2, sort_keys=True), encoding="utf-8"
    )


def read_index(path: str | Path) -> LocalVectorIndex:
    """Read a local vector index written by `write_index`."""

    input_path = Path(path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    return _index_from_dict(raw)


def answer_question(
    index: LocalVectorIndex,
    question: str,
    *,
    embedder: OllamaEmbeddingProvider,
    chat_client: OllamaChatClient,
    top_k: int = 4,
    context_char_limit: int = DEFAULT_CONTEXT_CHAR_LIMIT,
    measure_metrics: bool = True,
) -> RagAnswer:
    """Retrieve context from an index and ask the chat model to answer from it."""

    total_start = time.perf_counter()
    retrieve_start = total_start

    embed_result = embedder.embed_many((question,))
    query_embedding = embed_result.embeddings[0]
    embed_usage = embed_result.usage

    results = index.search(query_embedding, top_k=top_k)
    context = format_context(results, max_chars=context_char_limit)
    retrieve_ms = (time.perf_counter() - retrieve_start) * 1000

    messages = (
        ChatMessage(
            role="system",
            content=(
                "You answer questions about Jupyter notebooks using only the retrieved "
                "context. If the context is insufficient, say you do not know. Cite notebook "
                "ids and cell indices for factual claims."
            ),
        ),
        ChatMessage(
            role="user",
            content=(f"Retrieved notebook context:\n{context}\n\nQuestion: {question}"),
        ),
    )
    generate_start = time.perf_counter()
    chat_result = chat_client.chat(messages)
    generate_ms = (time.perf_counter() - generate_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    metrics = None
    if measure_metrics:
        metrics = AnswerMetrics(
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
            total_ms=total_ms,
            embed_usage=embed_usage,
            chat_usage=chat_result.usage,
            context_chars=len(context),
        )

    return RagAnswer(
        question=question,
        answer=chat_result.content,
        results=results,
        metrics=metrics,
    )


def format_context(results: Sequence[SearchResult], *, max_chars: int) -> str:
    """Format retrieval results for a chat prompt."""

    parts: list[str] = []
    used_chars = 0
    for result in results:
        header = (
            f"[{result.rank}] notebook={result.chunk.notebook_id} "
            f"cells={','.join(str(index) for index in result.chunk.cell_indices)} "
            f"score={result.score:.4f}"
        )
        text = result.chunk.text.strip()
        block = f"{header}\n{text}"
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = f"{block[: max(0, remaining - 15)].rstrip()}\n[truncated]"
        parts.append(block)
        used_chars += len(block)
    return "\n\n".join(parts) if parts else "No relevant notebook context was retrieved."


def _index_to_dict(index: LocalVectorIndex) -> dict[str, Any]:
    return {
        "format": INDEX_FORMAT,
        "version": INDEX_VERSION,
        "source": index.source,
        "notebook_paths": list(index.notebook_paths),
        "chunker": index.chunker,
        "embedding_model": index.embedding_model,
        "embedding_dimensions": index.embedding_dimensions,
        "created_at": index.created_at,
        "chunks": [
            {
                "chunk": _chunk_to_dict(item.chunk),
                "embedding": list(item.embedding),
            }
            for item in index.chunks
        ],
    }


def _index_from_dict(raw: Any) -> LocalVectorIndex:
    if not isinstance(raw, dict):
        msg = "index must be a JSON object"
        raise ValueError(msg)
    if raw.get("format") != INDEX_FORMAT:
        msg = f"unsupported index format {raw.get('format')!r}"
        raise ValueError(msg)
    if raw.get("version") != INDEX_VERSION:
        msg = f"unsupported index version {raw.get('version')!r}"
        raise ValueError(msg)

    indexed_chunks = tuple(_indexed_chunk(item) for item in _list_of_objects(raw, "chunks"))
    embedding_dimensions = _int(raw, "embedding_dimensions")
    for item in indexed_chunks:
        if len(item.embedding) != embedding_dimensions:
            msg = "all chunk embeddings must match embedding_dimensions"
            raise ValueError(msg)

    return LocalVectorIndex(
        source=_string(raw, "source"),
        notebook_paths=tuple(_list_of_strings(raw, "notebook_paths")),
        chunker=_string(raw, "chunker"),
        embedding_model=_string(raw, "embedding_model"),
        embedding_dimensions=embedding_dimensions,
        created_at=_string(raw, "created_at"),
        chunks=indexed_chunks,
    )


def _indexed_chunk(raw: dict[str, Any]) -> IndexedChunk:
    return IndexedChunk(
        chunk=_chunk_from_dict(_dict(raw, "chunk")),
        embedding=tuple(_list_of_floats(raw, "embedding")),
    )


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "notebook_id": chunk.notebook_id,
        "text": chunk.text,
        "strategy": chunk.strategy,
        "cell_indices": list(chunk.cell_indices),
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(raw: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=_string(raw, "chunk_id"),
        notebook_id=_string(raw, "notebook_id"),
        text=_string(raw, "text"),
        strategy=_string(raw, "strategy"),
        cell_indices=tuple(_list_of_ints(raw, "cell_indices")),
        metadata=_dict(raw, "metadata"),
    )


def _embedding_dimensions(embeddings: Sequence[tuple[float, ...]]) -> int:
    if not embeddings:
        return 0
    dimensions = len(embeddings[0])
    if any(len(embedding) != dimensions for embedding in embeddings):
        msg = "all embeddings must have the same dimensions"
        raise ValueError(msg)
    return dimensions


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def _list_of_objects(raw: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = raw.get(key)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    msg = f"{key!r} must be a list of objects"
    raise ValueError(msg)


def _list_of_strings(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    msg = f"{key!r} must be a list of strings"
    raise ValueError(msg)


def _list_of_ints(raw: dict[str, Any], key: str) -> list[int]:
    value = raw.get(key)
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return value
    msg = f"{key!r} must be a list of integers"
    raise ValueError(msg)


def _list_of_floats(raw: dict[str, Any], key: str) -> list[float]:
    value = raw.get(key)
    if isinstance(value, list) and all(isinstance(item, int | float) for item in value):
        return [float(item) for item in value]
    msg = f"{key!r} must be a list of numbers"
    raise ValueError(msg)


def _dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    msg = f"{key!r} must be an object"
    raise ValueError(msg)


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if isinstance(value, str):
        return value
    msg = f"{key!r} must be a string"
    raise ValueError(msg)


def _int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, int):
        return value
    msg = f"{key!r} must be an integer"
    raise ValueError(msg)
