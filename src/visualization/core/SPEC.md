# Visualization Core — Technical Specification

**Version**: 1.6.0

## Model Loading Priority

1. `*_parsed.json` from Step 3 output (preferred)
2. Raw GNN file parsing via `visualization/parse/` (fallback)

## Orchestration Flow

1. Discover models in target directory
2. Load parsed JSON representations
3. Dispatch to graph visualizer
4. Dispatch to matrix visualizer
5. Generate summary HTML

## Output Structure

```
8_visualization_output/
├── <model_name>/
│   ├── network_graph.png
│   ├── bipartite_graph.png
│   ├── matrix_heatmaps/
│   └── summary.html
└── visualization_summary.json
```
