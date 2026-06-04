from pathlib import Path

from nbrag.models import Chunk
from nbrag.rag import IndexedChunk, LocalVectorIndex, read_index, write_index


def test_local_vector_index_round_trip_and_search(tmp_path: Path) -> None:
    index = LocalVectorIndex(
        source="examples/notebooks",
        notebook_paths=("examples/notebooks/synthetic_forecast/synthetic_forecast.ipynb",),
        chunker="triad",
        embedding_model="qwen3-embedding:0.6b",
        embedding_dimensions=2,
        created_at="2026-01-01T00:00:00+00:00",
        chunks=(
            IndexedChunk(
                chunk=Chunk(
                    chunk_id="synthetic_forecast:triad:3",
                    notebook_id="synthetic_forecast",
                    text="def forecast_demand(frame, window=4): ...",
                    strategy="triad",
                    cell_indices=(2, 3),
                    metadata={"cell_type": "code"},
                ),
                embedding=(1.0, 0.0),
            ),
            IndexedChunk(
                chunk=Chunk(
                    chunk_id="synthetic_forecast:triad:5",
                    notebook_id="synthetic_forecast",
                    text="The final chart compares actual weekly units against the forecast.",
                    strategy="triad",
                    cell_indices=(4, 5),
                    metadata={"cell_type": "code"},
                ),
                embedding=(0.0, 1.0),
            ),
        ),
    )
    output_path = tmp_path / "index.json"

    write_index(index, output_path)
    loaded = read_index(output_path)
    results = loaded.search((1.0, 0.0), top_k=1)

    assert loaded == index
    assert results[0].chunk.chunk_id == "synthetic_forecast:triad:3"
