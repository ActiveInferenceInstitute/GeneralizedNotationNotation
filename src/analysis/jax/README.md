# JAX Analysis Module

Framework-specific analysis for JAX simulation results.

## Overview

This submodule provides analysis capabilities for JAX-based Active Inference simulations. JAX enables high-performance numerical computing with automatic differentiation.

## Module Structure

```
analysis/jax/
├── __init__.py    # Module exports
├── analyzer.py    # JAXAnalyzer class
├── README.md      # This file
└── AGENTS.md      # Agent scaffolding
```

## Key Components

### JAXAnalyzer (`analyzer.py`)

Analyzes JAX simulation outputs:

```python
from analysis.jax.analyzer import generate_analysis_from_logs

results = generate_analysis_from_logs(
    execution_dir=Path("output/12_execute_output"),
    output_dir=Path("output/16_analysis_output/jax"),
    verbose=True
)
```

**Capabilities:**
- Parse JAX simulation outputs
- Extract gradient information
- Analyze numerical precision
- Track convergence metrics
- Generate performance reports

## Analysis Features

- **Gradient Analysis**: Track gradient norms and directions
- **Convergence Tracking**: Monitor optimization convergence
- **Numerical Precision**: Validate floating-point accuracy
- **Performance Metrics**: JIT compilation, execution time
- **Memory Profiling**: GPU/TPU memory usage

## Data Flow

```
JAX Execution (Step 12)
    ↓
JAXAnalyzer.analyze()
    ↓
├── Parse outputs
├── Extract metrics
├── Gradient analysis
└── Performance profiling
    ↓
Analysis Output (Step 16)
```

## Input/Output

**Input:** `output/12_execute_output/*/jax/`
- Simulation outputs
- Execution logs

**Output:** `output/16_analysis_output/jax/`
- `analysis_summary.json`
- Convergence plots
- Performance reports

---

**Framework:** JAX (Python)
**Version:** 1.1.3
**Status:** Production Ready
