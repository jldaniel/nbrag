"""Notebook parsing utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import nbformat

from nbrag.models import CellType, NotebookCell, NotebookDocument, NotebookOutput

_SKIP_DIRS = {".ipynb_checkpoints", "_exec", ".venv"}


def find_notebooks(root: str | Path) -> tuple[Path, ...]:
    """Discover notebooks under `root`.

    A file path is returned as-is; a directory is searched recursively for `.ipynb`
    files, skipping checkpoint, execution, and virtualenv subfolders.
    """

    root_path = Path(root)
    if root_path.is_file():
        return (root_path,)
    if not root_path.is_dir():
        msg = f"notebook source not found: {root_path}"
        raise FileNotFoundError(msg)

    return tuple(
        sorted(
            path for path in root_path.rglob("*.ipynb") if not _SKIP_DIRS.intersection(path.parts)
        )
    )


def parse_notebook(path: str | Path, *, notebook_id: str | None = None) -> NotebookDocument:
    """Parse an `.ipynb` file without executing it."""

    notebook_path = Path(path)
    nb_node = nbformat.read(notebook_path, as_version=4)
    notebook_metadata = _as_plain_dict(nb_node.get("metadata", {}))
    cells = tuple(_parse_cell(index, cell) for index, cell in enumerate(nb_node.get("cells", [])))

    return NotebookDocument(
        notebook_id=notebook_id or notebook_path.stem,
        path=notebook_path,
        cells=cells,
        metadata=notebook_metadata,
    )


def _parse_cell(index: int, cell: Mapping[str, Any]) -> NotebookCell:
    cell_type = _normalize_cell_type(cell.get("cell_type"))
    outputs = tuple(_parse_output(output) for output in cell.get("outputs", []))
    execution_count = cell.get("execution_count")

    return NotebookCell(
        index=index,
        cell_type=cell_type,
        source=_source_to_text(cell.get("source", "")),
        metadata=_as_plain_dict(cell.get("metadata", {})),
        execution_count=execution_count if isinstance(execution_count, int) else None,
        outputs=outputs,
    )


def _parse_output(output: Mapping[str, Any]) -> NotebookOutput:
    output_type = str(output.get("output_type", "unknown"))
    return NotebookOutput(
        output_type=output_type,
        text=_output_to_text(output),
        metadata=_as_plain_dict(output.get("metadata", {})),
    )


def _output_to_text(output: Mapping[str, Any]) -> str:
    if "text" in output:
        return _source_to_text(output["text"])

    if "ename" in output or "evalue" in output:
        traceback = output.get("traceback", [])
        parts = [str(output.get("ename", "")), str(output.get("evalue", ""))]
        parts.extend(str(line) for line in traceback)
        return "\n".join(part for part in parts if part)

    data = output.get("data")
    if isinstance(data, Mapping):
        for mime_type in ("text/plain", "text/markdown", "text/html"):
            if mime_type in data:
                return _source_to_text(data[mime_type])
        return " ".join(sorted(str(key) for key in data))

    return ""


def _source_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _normalize_cell_type(value: Any) -> CellType:
    if value in {"code", "markdown", "raw"}:
        return cast(CellType, value)
    return "unknown"


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return {}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _as_plain_dict(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value
