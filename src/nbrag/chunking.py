"""Notebook chunking strategies."""

from __future__ import annotations

from typing import Protocol

from nbrag.models import Chunk, NotebookCell, NotebookDocument


class ChunkingStrategy(Protocol):
    """Produces retrieval chunks for a parsed notebook."""

    name: str

    def chunk(self, document: NotebookDocument) -> tuple[Chunk, ...]:
        """Return chunks for a notebook document."""
        ...


class NaiveTextChunker:
    """Flatten the whole notebook and split on whitespace."""

    name = "naive_text"

    def __init__(self, *, max_words: int = 180, overlap_words: int = 30) -> None:
        if max_words <= 0:
            msg = "max_words must be positive"
            raise ValueError(msg)
        if overlap_words < 0 or overlap_words >= max_words:
            msg = "overlap_words must be non-negative and smaller than max_words"
            raise ValueError(msg)
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk(self, document: NotebookDocument) -> tuple[Chunk, ...]:
        text = "\n\n".join(_cell_to_text(cell) for cell in document.cells)
        words = text.split()
        if not words:
            return ()

        chunks: list[Chunk] = []
        step = self.max_words - self.overlap_words
        for chunk_number, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.max_words]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.notebook_id}:{self.name}:{chunk_number}",
                    notebook_id=document.notebook_id,
                    text=" ".join(window),
                    strategy=self.name,
                    cell_indices=tuple(cell.index for cell in document.cells),
                    metadata={
                        "start_word": start,
                        "end_word": start + len(window),
                    },
                )
            )
            if start + self.max_words >= len(words):
                break
        return tuple(chunk for chunk in chunks if not chunk.is_empty)


class CellChunker:
    """Emit one retrieval chunk per notebook cell."""

    name = "cell"

    def chunk(self, document: NotebookDocument) -> tuple[Chunk, ...]:
        return tuple(
            Chunk(
                chunk_id=f"{document.notebook_id}:{self.name}:{cell.index}",
                notebook_id=document.notebook_id,
                text=_cell_to_text(cell),
                strategy=self.name,
                cell_indices=(cell.index,),
                metadata={
                    "cell_type": cell.cell_type,
                    "execution_count": cell.execution_count,
                    "source_metadata": cell.metadata,
                },
            )
            for cell in document.cells
            if cell.source.strip() or cell.outputs
        )


class TriadChunker:
    """Bind code cells to preceding markdown context and stored outputs."""

    name = "triad"

    def chunk(self, document: NotebookDocument) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        last_markdown: NotebookCell | None = None

        for cell in document.cells:
            if cell.cell_type == "markdown":
                last_markdown = cell
                chunks.append(_markdown_chunk(document, cell, self.name))
                continue

            if cell.cell_type == "code":
                context_cells = (last_markdown.index,) if last_markdown is not None else ()
                text_parts = []
                if last_markdown is not None:
                    text_parts.append(f"Markdown context:\n{last_markdown.source.strip()}")
                text_parts.append(f"Code:\n{cell.source.strip()}")
                output_text = _outputs_to_text(cell)
                if output_text:
                    text_parts.append(f"Stored outputs:\n{output_text}")

                chunks.append(
                    Chunk(
                        chunk_id=f"{document.notebook_id}:{self.name}:{cell.index}",
                        notebook_id=document.notebook_id,
                        text="\n\n".join(part for part in text_parts if part.strip()),
                        strategy=self.name,
                        cell_indices=(*context_cells, cell.index),
                        metadata={
                            "cell_type": cell.cell_type,
                            "execution_count": cell.execution_count,
                            "has_markdown_context": last_markdown is not None,
                            "has_outputs": bool(cell.outputs),
                            "source_metadata": cell.metadata,
                        },
                    )
                )
                continue

            if cell.source.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.notebook_id}:{self.name}:{cell.index}",
                        notebook_id=document.notebook_id,
                        text=_cell_to_text(cell),
                        strategy=self.name,
                        cell_indices=(cell.index,),
                        metadata={"cell_type": cell.cell_type},
                    )
                )

        return tuple(chunk for chunk in chunks if not chunk.is_empty)


def _markdown_chunk(document: NotebookDocument, cell: NotebookCell, strategy: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document.notebook_id}:{strategy}:{cell.index}",
        notebook_id=document.notebook_id,
        text=f"Markdown:\n{cell.source.strip()}",
        strategy=strategy,
        cell_indices=(cell.index,),
        metadata={
            "cell_type": cell.cell_type,
            "source_metadata": cell.metadata,
        },
    )


def _cell_to_text(cell: NotebookCell) -> str:
    parts = [f"{cell.cell_type.title()} cell:", cell.source.strip()]
    output_text = _outputs_to_text(cell)
    if output_text:
        parts.extend(["Stored outputs:", output_text])
    return "\n".join(part for part in parts if part)


def _outputs_to_text(cell: NotebookCell) -> str:
    return "\n".join(output.text.strip() for output in cell.outputs if output.text.strip())
