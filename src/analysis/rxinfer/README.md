# RxInfer Analysis Module

Framework-specific analysis for RxInfer.jl simulation results.

## Overview

This submodule provides analysis capabilities for RxInfer.jl (Julia) message-passing simulations. RxInfer implements reactive message passing for Bayesian inference.

## Module Structure

```
analysis/rxinfer/
├── __init__.py    # Module exports
├── analyzer.py    # RxInferAnalyzer class
├── README.md      # This file
└── AGENTS.md      # Agent scaffolding
```

## Key Components

### RxInferAnalyzer (`analyzer.py`)

Analyzes RxInfer simulation outputs:

```python
from analysis.rxinfer.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/rxinfer"),
    verbose=True
)
```

**Capabilities:**
- Parse message-passing outputs
- Analyze belief propagation
- Track message convergence
- Extract factor graph metrics
- Generate inference reports

## Analysis Features

- **Message Analysis**: Track message flow and convergence
- **Belief Propagation**: Monitor belief updates
- **Factor Graph Metrics**: Node/edge statistics
- **Convergence Analysis**: Iteration counts, residuals
- **Performance Profiling**: Message scheduling efficiency

## Data Flow

```
RxInfer Execution (Step 12)
    ↓
RxInferAnalyzer.analyze()
    ↓
├── Parse outputs
├── Extract messages
├── Analyze convergence
└── Generate metrics
    ↓
Analysis Output (Step 16)
```

## Input/Output

**Input:** `output/12_execute_output/*/rxinfer/`
- `simulation_results.json`
- Message logs
- Execution outputs

**Output:** `output/16_analysis_output/rxinfer/`
- `analysis_summary.json`
- Convergence plots
- Message flow reports

---

**Framework:** RxInfer.jl (Julia)
**Version:** 1.1.3
**Status:** Production Ready
