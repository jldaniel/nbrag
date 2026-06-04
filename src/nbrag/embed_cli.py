"""Command-line entry point for building a local Ollama vector index."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from nbrag.chunking import CellChunker, ChunkingStrategy, NaiveTextChunker, TriadChunker
from nbrag.ollama import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_EMBED_RETRIES,
    DEFAULT_EMBED_RETRY_DELAY_SECONDS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TIMEOUT_SECONDS,
    OllamaEmbeddingProvider,
    OllamaError,
)
from nbrag.parser import find_notebooks
from nbrag.rag import build_index, write_index

DEFAULT_NOTEBOOKS = "examples/notebooks"
DEFAULT_OUTPUT = "indexes/notebooks.json"

CHUNKERS: dict[str, ChunkingStrategy] = {
    NaiveTextChunker.name: NaiveTextChunker(),
    CellChunker.name: CellChunker(),
    TriadChunker.name: TriadChunker(),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Build a local vector index for notebook RAG."""

    args = _parse_args(argv)
    notebook_paths = find_notebooks(args.notebooks)
    if not notebook_paths:
        print(f"No notebooks found under {args.notebooks}.", file=sys.stderr)
        return 1
    chunker = CHUNKERS[args.chunker]
    embedder = OllamaEmbeddingProvider(
        model=args.model,
        host=args.ollama_host,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        retry_delay_seconds=args.retry_delay,
    )

    print(
        f"Embedding {len(notebook_paths)} notebook(s) from {args.notebooks} "
        f"with {embedder.model} at {embedder.host} "
        f"(batch_size={args.batch_size}, retries={embedder.max_retries})...",
        file=sys.stderr,
    )
    try:
        index = build_index(
            notebook_paths,
            chunker,
            embedder,
            source=str(args.notebooks),
            batch_size=args.batch_size,
            on_progress=lambda done, total: print(
                f"  embedded {done}/{total} chunks",
                file=sys.stderr,
            ),
        )
        write_index(index, args.output)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(index.chunks)} chunk embedding(s), "
        f"{index.embedding_dimensions} dimensions each, to {Path(args.output)}.",
        file=sys.stderr,
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nbrag-embed",
        description="Generate an Ollama-backed local vector index for notebook RAG.",
    )
    parser.add_argument(
        "notebooks",
        nargs="?",
        default=DEFAULT_NOTEBOOKS,
        help=(
            "Notebook directory (searched recursively) or a single .ipynb file. "
            f"Default: {DEFAULT_NOTEBOOKS}."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path for the generated index JSON. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--chunker",
        choices=sorted(CHUNKERS),
        default=TriadChunker.name,
        help="Chunking strategy to index. Default: triad.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Ollama embedding model. Default: {DEFAULT_EMBED_MODEL}.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help=f"Ollama base URL. Default: OLLAMA_HOST or {DEFAULT_OLLAMA_HOST}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Ollama request timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS:g}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Chunks per Ollama embed request. Use 1 if the host runner returns "
            "EOF errors. Default: 1."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_EMBED_RETRIES,
        help=f"Retries per embed request on failure. Default: {DEFAULT_EMBED_RETRIES}.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_EMBED_RETRY_DELAY_SECONDS,
        help=(
            "Base delay in seconds between retries (multiplied by attempt number). "
            f"Default: {DEFAULT_EMBED_RETRY_DELAY_SECONDS:g}."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
