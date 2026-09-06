# ML Integration Module Specification

Feature extraction from GNN models and scikit-learn classifier training (model-family / complexity classification) with cross-validation and feature-importance reporting.

## Components

### Core
- `processor.py` - ML integration processor (`process_ml_integration`, `extract_gnn_features`, training + feature-analysis helpers)
- `mcp.py` - MCP tool registrations (4 tools)
- `frameworks.py` - framework availability probes (`check_ml_frameworks` delegates here)
- `inference.py` - artifact loading + prediction (`load_classifier`, `predict_with_model`, `predict_batch`, `InferenceError`)

## Features
- Structural feature extraction from GNN markdown
- scikit-learn DecisionTree/RandomForest training
- Adaptive k-fold cross-validation and feature importance
- Framework availability detection (PyTorch, TensorFlow, JAX, scikit-learn)
- Deterministic inference from saved `.pkl` artifacts via the canonical feature vector (`NUMERIC_FEATURE_NAMES` order; scikit-learn required only at inference call time)

## Key Exports
```python
from gnn.ml_integration import (
    process_ml_integration, check_ml_frameworks, predict_with_model, feature_vector,
)
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
