# GNN Kit

## Overview

Tools and utilities for GNN model development and management.

## Purpose

The GNN Kit provides a centralized collection of scripts for:
- Model validation and type checking
- Performance benchmarking
- Resource estimation
- Framework integration

## 🛠️ Performance & CLI Tools

GNN Kit includes several high-performance utilities:
- **`gnn-validate`**: Standalone type checker for GNN files.
    - `--check-matrices`: Verifies probability stochasticity.
    - `--verbose`: Shows detailed line-by-line parsing steps.
- **`gnn-bench`**: Benchmark framework execution across PyMDP and JAX.
    - `--iterations <N>`: Run multiple inference steps to average performance.
    - `--profile`: Generate flamegraphs for bottleneck analysis.

## Future Contributions
- Improved GUI integration
- Expanded multi-agent templates
- Automated report generation extensions
