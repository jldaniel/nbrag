from pathlib import Path

from nbrag.chunking import CellChunker, NaiveTextChunker, TriadChunker
from nbrag.parser import parse_notebook

FIXTURE = Path("examples/notebooks/synthetic_forecast/synthetic_forecast.ipynb")


def test_parse_notebook_preserves_cells_and_outputs() -> None:
    document = parse_notebook(FIXTURE)

    assert document.notebook_id == "synthetic_forecast"
    assert len(document.cells) == 6
    assert document.cells[3].cell_type == "code"
    assert document.cells[3].execution_count == 2
    assert "forecast_demand" in document.cells[3].source
    assert "mae" in document.cells[3].outputs[0].text


def test_chunkers_emit_distinct_retrieval_units() -> None:
    document = parse_notebook(FIXTURE)

    naive_chunks = NaiveTextChunker(max_words=30, overlap_words=5).chunk(document)
    cell_chunks = CellChunker().chunk(document)
    triad_chunks = TriadChunker().chunk(document)

    assert len(naive_chunks) > 1
    assert len(cell_chunks) == 6
    assert len(triad_chunks) == 6

    forecast_chunk = next(chunk for chunk in triad_chunks if chunk.cell_indices[-1] == 3)
    assert "Forecast model" in forecast_chunk.text
    assert "forecast_demand" in forecast_chunk.text
    assert "mae" in forecast_chunk.text
    assert forecast_chunk.metadata["has_markdown_context"] is True
