# Visualization Parse

GNN file parsing utilities tailored for the visualization pipeline.

## Exports

- `GNNParser` — Configurable parser for extracting visualization-ready data from GNN files,
  including state spaces, connection graphs, and matrix specifications
- `parse_gnn_content()` — Parse raw GNN markdown content into structured sections

## Dependencies

- Standard library only (no external dependencies)

## Usage

```python
from visualization.parse import GNNParser, parse_gnn_content
parser = GNNParser()
model = parser.parse(gnn_file_path)
```
