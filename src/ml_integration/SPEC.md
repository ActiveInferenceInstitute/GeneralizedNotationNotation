# ML Integration Module Specification

Feature extraction from GNN models and scikit-learn classifier training (model-family / complexity classification) with cross-validation and feature-importance reporting.

## Components

### Core
- `processor.py` - ML integration processor (`process_ml_integration`, `extract_gnn_features`, training + feature-analysis helpers)
- `mcp.py` - MCP tool registrations (4 tools)

## Features
- Structural feature extraction from GNN markdown
- scikit-learn DecisionTree/RandomForest training
- Adaptive k-fold cross-validation and feature importance
- Framework availability detection (PyTorch, TensorFlow, JAX, scikit-learn)
- Degradation to feature-analysis-only without scikit-learn

## Key Exports
```python
from ml_integration import process_ml_integration, check_ml_frameworks
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
