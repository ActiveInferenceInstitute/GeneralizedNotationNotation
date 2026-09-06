---
name: gnn-ml-integration
description: GNN machine learning integration and model training. Use when training ML models on GNN data, checking ML framework availability, or integrating GNN pipeline outputs with machine learning workflows.
---

# GNN Machine Learning Integration (Step 14)

## Purpose

Integrates machine learning capabilities with the GNN pipeline, enabling feature extraction from parsed models, ML model training on GNN data, and framework availability checking.

## Key Commands

```bash
# Run ML integration
python src/gnn/14_ml_integration.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/gnn/main.py --only-steps 14 --verbose
```

## API

```python
from gnn.ml_integration import process_ml_integration, check_ml_frameworks, feature_vector, predict_with_model

# Process ML integration step (used by pipeline)
result = process_ml_integration(target_dir, output_dir, verbose=True)

# Check available ML frameworks
frameworks = check_ml_frameworks()
# Returns: {'pytorch': {'available': True, 'version': '2.x'},
#           'tensorflow': {'available': False},
#           'jax': {'available': True, 'version': '0.4.x'},
#           'sklearn': {'available': True, 'version': '1.x'}}
# Build the canonical feature vector and predict from a saved .pkl artifact
vec = feature_vector(features)
label = predict_with_model(Path("output/14_ml_integration_output/gnn_decision_tree.pkl"), features, label_names=labels)
```

## Key Exports

- `process_ml_integration` — main pipeline processing function
- `check_ml_frameworks` — checks availability of PyTorch, TensorFlow, JAX, scikit-learn
- `extract_gnn_features` — structural feature extraction for one GNN file
- `feature_vector` — canonical numeric vector in `NUMERIC_FEATURE_NAMES` order
- `complexity_label` — small/medium/large via `COMPLEXITY_THRESHOLDS` (100, 1000)
- `summarize_features` — min/max/mean over `SUMMARY_STATISTIC_KEYS`
- `load_classifier`, `predict_with_model`, `predict_batch` — inference from saved `.pkl` artifacts (scikit-learn required only at call time; `label_names` comes from `ml_integration_results.json`)
- `InferenceError` — raised for missing/invalid artifacts or prediction failures
- `get_module_info` — version, feature flags, tool inventory
- Constants: `NUMERIC_FEATURE_NAMES`, `COMPLEXITY_THRESHOLDS`, `COMPLEXITY_LABELS`, `SUMMARY_STATISTIC_KEYS`

## Supported ML Frameworks

| Framework | Check Key | Purpose |
| ----------- | ----------- | --------- |
| **PyTorch** | `pytorch` | Deep learning, CUDA support detection |
| **TensorFlow** | `tensorflow` | Neural network training |
| **JAX** | `jax` | Differentiable computing |
| **scikit-learn** | `sklearn` | Classical ML algorithms |

## Dependencies

```bash
# ML training deps
uv sync --extra ml-ai

# Includes: transformers, scipy, scikit-learn
# Note: torch ships in its own `torch` extra (uv sync --extra torch;
# torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47).
```

## Output

- ML results in `output/14_ml_integration_output/`
- Framework availability reports


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `check_ml_frameworks`
- `get_ml_module_info`
- `list_ml_integration_targets`
- `process_ml_integration`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
