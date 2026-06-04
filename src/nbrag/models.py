"""Core data structures used by parser, chunkers, and retrievers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CellType = Literal["code", "markdown", "raw", "unknown"]


@dataclass(frozen=True)
class NotebookOutput:
    """A stored notebook output, never re-executed by nbrag."""

    output_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NotebookCell:
    """A normalized notebook cell with source, metadata, and stored outputs."""

    index: int
    cell_type: CellType
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_count: int | None = None
    outputs: tuple[NotebookOutput, ...] = ()


@dataclass(frozen=True)
class NotebookDocument:
    """A parsed notebook document."""

    notebook_id: str
    path: Path
    cells: tuple[NotebookCell, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit produced by a chunking strategy."""

    chunk_id: str
    notebook_id: str
    text: str
    strategy: str
    cell_indices: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class SearchResult:
    """A ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int
    source: str
