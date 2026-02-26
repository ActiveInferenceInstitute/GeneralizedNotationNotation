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
python src/14_ml_integration.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 14 --verbose
```

## API

```python
from ml_integration import process_ml_integration, check_ml_frameworks

# Process ML integration step (used by pipeline)
result = process_ml_integration(target_dir, output_dir, verbose=True)

# Check available ML frameworks
frameworks = check_ml_frameworks()
# Returns: {'pytorch': {'available': True, 'version': '2.x'}, 
#           'tensorflow': {'available': False},
#           'jax': {'available': True, 'version': '0.4.x'},
#           'sklearn': {'available': True, 'version': '1.x'}}
```

## Key Exports

- `process_ml_integration` — main pipeline processing function
- `check_ml_frameworks` — checks availability of PyTorch, TensorFlow, JAX, scikit-learn

## Supported ML Frameworks

| Framework | Check Key | Purpose |
| ----------- | ----------- | --------- |
| **PyTorch** | `pytorch` | Deep learning, CUDA support detection |
| **TensorFlow** | `tensorflow` | Neural network training |
| **JAX** | `jax` | Differentiable computing |
| **scikit-learn** | `sklearn` | Classical ML algorithms |

## Dependencies

```bash
# ML framework deps
uv sync --extra ml-ai

# Includes: torch, transformers, datasets
```

## Output

- ML results in `output/14_ml_integration_output/`
- Framework availability reports


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_ml_integration`
- `check_ml_frameworks`
- `list_ml_integration_targets`

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
