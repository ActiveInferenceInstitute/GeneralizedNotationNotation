# execution

Framework execution and simulation backend management

## Overview

This directory contains documentation and resources for the execution subsystem.

**Contents**: Framework availability, integration guides, execution strategies

## Quick Links

- **Main Documentation**: [doc/README.md](../README.md)
- **Setup Guide**: [doc/SETUP.md](../SETUP.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Directory Structure

```
execution/
├── README.md (this file)
├── AGENTS.md
└── FRAMEWORK_AVAILABILITY.md
```

## Framework Support

The execution subsystem provides support for multiple Active Inference simulation frameworks:

- **PyMDP**: Python Active Inference (primary)
- **RxInfer.jl**: Julia Bayesian inference
- **ActiveInference.jl**: Complete Julia implementation
- **DisCoPy**: Category theory and quantum computing
- **JAX**: GPU-accelerated tensor operations

## Integration

This subsystem is part of the broader GNN (Generalized Notation Notation) documentation ecosystem and pipeline.

**Related Components**:
- Setup and dependencies: [doc/SETUP.md](../SETUP.md)
- Framework guides: [pymdp/](../pymdp/), [rxinfer/](../rxinfer/)
- Pipeline orchestration: [src/](../../src/)

## Key Resources

- **Framework Availability**: [FRAMEWORK_AVAILABILITY.md](FRAMEWORK_AVAILABILITY.md)
- **PyMDP Integration**: [pymdp/gnn_pymdp.md](../pymdp/gnn_pymdp.md)
- **RxInfer Integration**: [rxinfer/gnn_rxinfer.md](../rxinfer/gnn_rxinfer.md)
- **Setup Instructions**: [doc/SETUP.md](../SETUP.md)

## Usage

See the [main documentation index](../README.md) for comprehensive guides and framework selection strategies.

---

**Status**: Production Ready  
**Version**: 1.0

