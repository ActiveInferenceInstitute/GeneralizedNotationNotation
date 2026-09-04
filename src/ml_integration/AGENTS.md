# ML Integration Module - Agent Scaffolding

## Module Overview

**Purpose**: Extracts structural features from GNN specification files and trains scikit-learn classifiers (DecisionTree + RandomForest) for model-family and complexity classification, with cross-validation and feature-importance reporting.

**Pipeline Step**: Step 14: ML integration (14_ml_integration.py)

**Category**: Machine Learning / Model Training

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-02

---

## Core Functionality

### Primary Responsibilities
1. Extract structural features from GNN markdown files (`num_states`, `num_observations`, `num_actions`, dimension statistics, connectivity ratios, qualitative flags)
2. Train scikit-learn classifiers to predict model family or complexity class
3. Run adaptive k-fold cross-validation (up to 5 folds) when class support allows
4. Report feature importance and top-5 features per trained model
5. Serialize trained models as `.pkl` artifacts and a results JSON

### Key Capabilities
- **Real Model Training**: `DecisionTreeClassifier(max_depth=4)` and `RandomForestClassifier(n_estimators=10, max_depth=4)`, both `random_state=42`.
- **Automatic Task Selection**: model-family classification when labels vary; `small`/`medium`/`large` complexity classification otherwise.
- **Feature Extraction**: Parses `## StateSpaceBlock` dimensions and `## Connections` counts from GNN markdown.
- **Graceful Fallback**: When every sample has the same label, no classifier is trained; a per-feature summary-statistics analysis is saved instead.

---

## API Reference

### Public Functions

#### `process_ml_integration(target_dir: Path, output_dir: Path, recursive: bool = False, verbose: bool = False, **kwargs) -> bool`
**Description**: Main ML integration processing function called by orchestrator (14_ml_integration.py). Extracts features, trains models, writes results.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to process
- `output_dir` (Path): Output directory for ML integration results
- `recursive` (bool): Search target_dir recursively (default: False)
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Accepted and ignored; reserved for pipeline-template compatibility

**Returns**: `bool` - True if ML integration processing succeeded, False otherwise

**Example**:
```python
from ml_integration import process_ml_integration
from pathlib import Path

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    verbose=True,
)
```

#### `check_ml_frameworks() -> Dict[str, Any]`
Availability check returning a dict keyed by `pytorch`, `tensorflow`, `jax`, `sklearn`; each value is `{"available": bool, "version": str|None}` (PyTorch also carries `cuda_available`).

#### `extract_gnn_features(file_path: Path) -> Dict[str, Any]`
Extracts the structural feature dict for a single GNN file.

---

## ML Framework Support

| Framework | Role |
|-----------|------|
| scikit-learn | Training framework (DecisionTree + RandomForest) |
| PyTorch | Detection only (`check_ml_frameworks`); not used for training |
| TensorFlow | Detection only |
| JAX | Detection only |

**PyTorch note**: PyTorch ships in the dedicated `torch` extra (`uv sync --extra torch`; torch>=2.13.0 resolves GHSA-rrmf-rvhw-rf47, which previously kept it out of every pyproject group). Users without the extra see PyTorch reported unavailable.

---

## Dependencies

### Required Dependencies
- Standard library only for the module shell (`json`, `logging`, `re`, `pickle`)

### Optional Dependencies (via `ml-ai` extra)
- `numpy` - Array handling for training matrices
- `scikit-learn` - Classifiers, cross-validation, label encoding (deferred imports)
- `scipy`, `transformers` - other members of the `ml-ai` extra

Install with `uv sync --extra ml-ai`. Without it, the step degrades to feature-extraction-only analysis.

### Detection-Only (never imported for training)
- `torch`, `tensorflow`, `jax`

### Internal Dependencies
- `utils.pipeline_template` - Standardized pipeline processing
- `pipeline.config` - Configuration management

---

## Configuration

### Environment Variables

None dedicated to this module. Behavior is fixed by
`process_ml_integration()` in `ml_integration/processor.py`; output location is
set via the `--output-dir` CLI flag on `14_ml_integration.py` and
`input/config.yaml` pipeline settings.

### Default Settings

Classifier hyperparameters, adaptive fold selection (`_cross_validation_folds`),
and task selection are set in `ml_integration/processor.py`.

---

## Usage Examples

### Basic ML Integration
```python
from ml_integration.processor import process_ml_integration

success = process_ml_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/14_ml_integration_output"),
    verbose=True,
)
```

### Framework Detection
```python
from ml_integration import check_ml_frameworks

frameworks = check_ml_frameworks()
sklearn_status = frameworks["sklearn"]
```

---

## Output Specification

### Output Products
- `gnn_decision_tree.pkl` - Pickled DecisionTreeClassifier (when trained)
- `gnn_random_forest.pkl` - Pickled RandomForestClassifier (when trained)
- `ml_integration_results.json` - Extracted features, model metrics, CV results, framework status

### Output Directory Structure
```
output/14_ml_integration_output/
├── gnn_decision_tree.pkl
├── gnn_random_forest.pkl
└── ml_integration_results.json
```

When training is not possible (insufficient label variation), only
`ml_integration_results.json` is written, containing the feature
summary-statistics analysis.

---

## Error Handling

### Graceful Degradation
- **scikit-learn/numpy missing**: Feature extraction only; per-feature summary analysis saved
- **Single-label dataset**: No classifier trained; `classification_status: insufficient_label_variation` recorded
- **Per-model training failure**: Logged, remaining models still attempted

### Error Categories
1. **Dependency Errors**: Missing numpy/scikit-learn (deferred import)
2. **Data Errors**: Insufficient samples or label variation
3. **Training Errors**: Per-model failures (logged, non-fatal)

---

## Integration Points

### Orchestrated By
- **Script**: `14_ml_integration.py` (Step 14)
- **Function**: `process_ml_integration()`

### Imports From
- `utils.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `src/tests/ml_integration/` - ML integration tests
- `src/main.py` - Runs `14_ml_integration.py` as a pipeline step (subprocess)

### Data Flow
```
GNN Files → Feature Extraction → Task Selection → Model Training → CV Evaluation → .pkl artifacts + results JSON
```

---

## Testing

### Test Files
- `src/tests/ml_integration/test_ml_integration_overall.py` - Module-level tests
- `src/tests/ml_integration/test_ml_integration_public_api.py` - Public API contract tests
- `src/tests/ml_integration/test_ml_integration_coverage.py` - Coverage tests
- `src/tests/ml_integration/test_ml_integration_mcp_tools.py` - MCP tool tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/ml_integration/ \
    --cov=src/ml_integration --cov-report=term-missing
```

### Key Test Scenarios
1. Feature extraction from GNN markdown
2. Task selection (family vs complexity classification)
3. Cross-validation behavior with limited class support
4. Degradation without scikit-learn

---

## MCP Integration

### Tools Registered
- `process_ml_integration` - Run ML integration processing
- `check_ml_frameworks` - Check available ML frameworks and versions
- `list_ml_integration_targets` - List GNN-compatible ML integration targets
- `get_ml_module_info` - Return version, feature flags, tool inventory

Each name above is registered by `register_tools()` with a named callable,
JSON input schema, module/category metadata, and explicit success/error results.

### MCP File Location
- `src/ml_integration/mcp.py` - MCP tool registrations

---

**Last Updated**: 2026-09-02
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
