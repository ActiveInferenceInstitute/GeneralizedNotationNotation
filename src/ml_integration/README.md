# ML Integration Module

This module (Pipeline Step 14) extracts structural features from GNN specification files and trains scikit-learn classifiers on them — model-family classification, complexity classification, 5-fold cross-validation, and feature-importance reporting.

## Module Structure

```
src/ml_integration/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # Feature extraction + model training
├── mcp.py                         # MCP tool registrations
└── README.md                      # This documentation
```

## Core Components

### `process_ml_integration(target_dir: Path, output_dir: Path, recursive: bool = False, verbose: bool = False, **kwargs) -> bool`

Main entry point, called by `14_ml_integration.py` (Step 14). Additional kwargs are accepted and ignored (pipeline-template compatibility).

- Extracts real GNN features per file: `num_variables`, `num_states`, `num_observations`, `num_actions`, dimension statistics, `total_parameters`, directed/undirected connection counts, `connectivity_ratio`, and qualitative flags (`has_precision`, `has_learning`, ...)
- Trains two classifiers per task: `DecisionTreeClassifier(max_depth=4)` and `RandomForestClassifier(n_estimators=10, max_depth=4)`, both with `random_state=42`
- Chooses the classification task automatically: model-family classification when labels vary, complexity classification (`small`/`medium`/`large`) otherwise
- Runs k-fold cross-validation when at least two members of every represented class exist (folds chosen adaptively, up to 5); otherwise reports `validation_status: insufficient_class_support`
- Saves trained models with `pickle` and a full results JSON

**Returns:** `bool` — True if processing succeeded.

### `check_ml_frameworks() -> Dict[str, Any]`

Availability check for `pytorch`, `tensorflow`, `jax`, and `sklearn` (returns `{"available": bool, "version": str|None}` per framework; PyTorch also reports `cuda_available`).

### `extract_gnn_features(file_path: Path) -> Dict[str, Any]`

Structural feature extraction for a single GNN file.

### Exports (`from ml_integration import ...`)

- `process_ml_integration`
- `check_ml_frameworks`
- `FEATURES`, `__version__`

## Usage Examples

### Basic usage

```python
from ml_integration import process_ml_integration
from pathlib import Path

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    verbose=True,
)
```

### Framework availability

```python
from ml_integration import check_ml_frameworks

frameworks = check_ml_frameworks()
print(frameworks["sklearn"])   # {'available': True, 'version': '1.x'}
print(frameworks["pytorch"])   # {'available': ..., 'version': ..., 'cuda_available': ...}
```

## Integration with Pipeline

### Pipeline Step 14: ML Integration

`14_ml_integration.py` is a thin orchestrator: it parses the standardized `--target-dir`, `--output-dir`, `--recursive`, `--verbose` arguments and delegates to `process_ml_integration()`.

### Output Structure

```
output/14_ml_integration_output/
├── gnn_decision_tree.pkl         # Pickled DecisionTreeClassifier
├── gnn_random_forest.pkl         # Pickled RandomForestClassifier
└── ml_integration_results.json   # Features, model metrics, CV results, framework status
```

When training is not possible (insufficient label variation), a per-feature summary-statistics analysis is saved to `ml_integration_results.json` instead.

## Framework Support

| Framework | Role in this module |
|-----------|--------------------|
| scikit-learn | Training framework (DecisionTree + RandomForest) |
| PyTorch | Detection only (`check_ml_frameworks`); not used for training |
| TensorFlow | Detection only |
| JAX | Detection only |

PyTorch is intentionally not locked in any pyproject extra while GHSA-rrmf-rvhw-rf47 has no patched release; users who need it install it manually. scikit-learn and scipy come from the `ml-ai` extra (`uv sync --extra ml-ai`, which also installs transformers).

## Dependencies

- **Required (stdlib)**: json, logging, re, pickle
- **Optional (`ml-ai` extra)**: numpy, scikit-learn (deferred imports; the step degrades to feature analysis without them)
- **Detection only**: torch, tensorflow, jax (never imported for training)

## Testing

Tests live in `src/tests/ml_integration/`: `test_ml_integration_overall.py`, `test_ml_integration_public_api.py`, `test_ml_integration_coverage.py`, `test_ml_integration_mcp_tools.py`.

```bash
uv run --extra dev python -m pytest src/tests/ml_integration/ --cov=src/ml_integration
```

## References

- Project overview: ../../README.md
- Pipeline details: ../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
