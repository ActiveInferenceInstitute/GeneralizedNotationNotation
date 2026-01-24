# RxInfer Analysis - Agent Scaffolding

## Overview

Framework-specific analyzer for RxInfer.jl simulation results. Part of the Analysis module (Step 16).

## Module Structure

```
analysis/rxinfer/
├── __init__.py    # Public API
├── analyzer.py    # RxInferAnalyzer class
├── README.md      # Human documentation
└── AGENTS.md      # This file
```

## Key Functions

### analyzer.py

- `generate_analysis_from_logs(execution_dir, output_dir, verbose)` - Main entry point
- `_parse_rxinfer_outputs(filepath)` - Parse RxInfer outputs
- `_analyze_messages(data)` - Message flow analysis
- `_analyze_convergence(data)` - Convergence tracking
- `_generate_report(metrics)` - Report generation

## Integration Points

**Upstream:** Execute module (Step 12) produces RxInfer simulation results
**Downstream:** Report module (Step 23) consumes analysis outputs

## Dependencies

- pathlib, json, logging: Core Python
- numpy (optional): Numerical operations
- matplotlib (optional): Visualization

---

**Version:** 1.1.3
**Last Updated:** 2026-01-23
