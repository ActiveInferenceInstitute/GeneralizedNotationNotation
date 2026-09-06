# ML Integration Module

This module (Pipeline Step 14) extracts structural features from GNN specification files and trains scikit-learn classifiers on them — model-family classification, complexity classification, 5-fold cross-validation, and feature-importance reporting.

## Module Structure

```
src/gnn/ml_integration/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # Feature extraction + model training
├── frameworks.py                  # Detection-only framework availability probes
├── inference.py                   # Artifact loading + prediction (.pkl inference)
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

### `feature_vector(features: Mapping[str, Any]) -> list[float]`

Canonical numeric vector for a feature dict, in `NUMERIC_FEATURE_NAMES` order. Missing keys default to `0.0` (`planning_horizon` defaults to `1.0`); booleans map to `0.0`/`1.0`.

```python
from gnn.ml_integration import feature_vector

vec = feature_vector({"num_states": 3, "num_observations": 5})
```

### `complexity_label(total_parameters: float) -> str`

Maps a parameter count to `"small"`, `"medium"`, or `"large"` via `COMPLEXITY_THRESHOLDS = (100, 1000)`.

```python
from gnn.ml_integration import complexity_label

complexity_label(250)   # 'medium'
```

### `summarize_features(features: Sequence[Mapping[str, Any]]) -> dict`

Pure, deterministic min/max/mean summary over `SUMMARY_STATISTIC_KEYS` (`num_states`, `num_observations`, `num_actions`, `total_parameters`, `connectivity_ratio`).

```python
from gnn.ml_integration import summarize_features

summary = summarize_features(list_of_feature_dicts)
summary["num_states"]["mean"]
```

### Inference

Trained `.pkl` artifacts can be reused without retraining. scikit-learn is required only at inference call time (deferred import); `label_names` comes from `ml_integration_results.json` when the model was trained with label encoding.

```python
from gnn.ml_integration import predict_with_model, predict_batch, load_classifier, InferenceError

model = load_classifier(Path("output/14_ml_integration_output/gnn_decision_tree.pkl"))
label = predict_with_model(Path(".../gnn_decision_tree.pkl"), features, label_names=labels)
labels = predict_batch(Path(".../gnn_random_forest.pkl"), [features1, features2])
```

Failures raise `InferenceError` (a `RuntimeError`), including malformed artifacts and missing dependencies.

Loading supports only Step 14's `DecisionTreeClassifier` and `RandomForestClassifier` artifacts. An exact global allowlist permits those estimator classes, their tree storage, and the NumPy array/dtype reconstruction primitives used by pickle protocols 4 and 5. Other globals are rejected before import or execution; extension opcodes, trailing data, and unexpected root types are rejected as well. There is no unrestricted loading option.

These checks do not authenticate artifacts or guarantee compatibility across scikit-learn/NumPy versions. Allowed native reconstruction still consumes memory and CPU; the loader is not a resource sandbox. Use artifacts with known provenance and compatible dependencies. Other estimator families and arbitrary pickled Python objects are unsupported.

### Exports (`from ml_integration import ...`)

- `process_ml_integration`, `extract_gnn_features`
- `check_ml_frameworks`
- `feature_vector`, `complexity_label`, `summarize_features`
- `predict_with_model`, `predict_batch`, `load_classifier`, `InferenceError`
- `get_module_info`
- Constants: `NUMERIC_FEATURE_NAMES`, `COMPLEXITY_THRESHOLDS`, `COMPLEXITY_LABELS`, `SUMMARY_STATISTIC_KEYS`
- `FEATURES`, `__version__`

## Usage Examples

### Basic usage

```python
from gnn.ml_integration import process_ml_integration
from pathlib import Path

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    verbose=True,
)
```

### Framework availability

```python
from gnn.ml_integration import check_ml_frameworks

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

When training is not possible (insufficient label variation, sklearn unavailable, or fewer than 2 usable samples), a per-feature summary-statistics analysis is saved to `ml_integration_results.json` instead. In both degradation cases (sklearn missing AND <2 files) a `structural_analysis` entry is written and the JSON still contains feature_statistics and model_families.

## Framework Support

| Framework | Role in this module |
|-----------|--------------------|
| scikit-learn | Training framework (DecisionTree + RandomForest) |
| PyTorch | Detection only (`check_ml_frameworks`); not used for training |
| TensorFlow | Detection only |
| JAX | Detection only |

PyTorch ships in the dedicated `torch` extra (`uv sync --extra torch`; torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47); users without the extra see PyTorch reported unavailable. scikit-learn and scipy come from the `ml-ai` extra (`uv sync --extra ml-ai`, which also installs transformers).

## Dependencies

- **Required (stdlib)**: json, logging, re, pickle
- **Optional (`ml-ai` extra)**: numpy, scikit-learn (deferred imports; the step degrades to feature analysis without them)
- **Detection only**: torch, tensorflow, jax (never imported for training)

## Testing

Tests live in `tests/ml_integration/`: `test_ml_integration_overall.py`, `test_ml_integration_public_api.py`, `test_ml_integration_coverage.py`, `test_ml_integration_mcp_tools.py`, `test_ml_integration_features.py`, `test_ml_integration_degradation.py`, `test_ml_integration_inference.py`, `test_classifier_deserialization_security.py`.

```bash
uv run --extra dev python -m pytest tests/ml_integration/ --cov=src/gnn/ml_integration
```

## References

- Project overview: ../../README.md
- Pipeline details: ../../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
