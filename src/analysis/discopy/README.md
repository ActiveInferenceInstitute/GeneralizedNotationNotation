# DisCoPy Analysis Module

Framework-specific analysis for DisCoPy categorical diagram results.

## Overview

This submodule provides analysis capabilities for DisCoPy (Distributional Compositional Python) simulations. DisCoPy implements categorical quantum mechanics and string diagrams for Active Inference.

## Module Structure

```
analysis/discopy/
├── __init__.py    # Module exports
├── analyzer.py    # DisCoPyAnalyzer class
├── README.md      # This file
└── AGENTS.md      # Agent scaffolding
```

## Key Components

### DisCoPyAnalyzer (`analyzer.py`)

Analyzes DisCoPy diagram outputs:

```python
from analysis.discopy.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/discopy"),
    verbose=True
)
```

**Capabilities:**
- Parse diagram representations
- Analyze circuit structure
- Extract categorical metrics
- Validate diagram composition
- Generate structure reports

## Analysis Features

- **Diagram Topology**: Analyze box and wire structure
- **Composition Analysis**: Track functor applications
- **Circuit Metrics**: Gate counts, depth analysis
- **Categorical Invariants**: Type checking, naturality
- **Visualization**: Diagram rendering and export

## Data Flow

```
DisCoPy Execution (Step 12)
    ↓
DisCoPyAnalyzer.analyze()
    ↓
├── Parse diagrams
├── Extract structure
├── Analyze composition
└── Generate metrics
    ↓
Analysis Output (Step 16)
```

## Input/Output

**Input:** `output/12_execute_output/*/discopy/`
- Diagram outputs (JSON, pickle)
- Circuit information
- Execution logs

**Output:** `output/16_analysis_output/discopy/`
- `analysis_summary.json`
- Diagram visualizations
- Structure reports

---

**Framework:** DisCoPy (Python)
**Version:** 1.1.3
**Status:** Production Ready
