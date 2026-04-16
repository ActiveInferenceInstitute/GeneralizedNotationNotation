# Visualization Parse — Technical Specification

**Version**: 1.6.0

## Parsing Priority

1. Step 3 JSON output (`*_parsed.json`) — preferred
2. Raw GNN markdown format — fallback via `markdown.py`
3. Raw GNN file — fallback via `gnn_file_parser.py`

## Extracted Fields

- `states: List[str]` — Hidden state names
- `observations: List[str]` — Observation names
- `actions: List[str]` — Action names
- `connections: List[Tuple[str, str]]` — Directed edges
- `matrices: Dict[str, np.ndarray]` — Numerical parameters
