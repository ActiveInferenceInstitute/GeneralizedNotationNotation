# execution

Framework execution and simulation backend management

**Signposts:** [docs/SPEC.md](../SPEC.md) (documentation versioning) · [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) (pipeline steps; Step 12 Execute) · [docs/gnn/operations/gnn_tools.md](../gnn/operations/gnn_tools.md)

## Overview

This directory contains documentation and resources for the execution subsystem.

**Contents**: Framework availability, integration guides, execution strategies

## Quick Links

- **Main Documentation**: [docs/README.md](../README.md)
- **Setup Guide**: [docs/SETUP.md](../SETUP.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)

## Directory Structure

```
execution/
├── README.md (this file)
├── AGENTS.md
└── FRAMEWORK_AVAILABILITY.md
```

## Framework Support

The execution subsystem provides support for multiple Active Inference
simulation frameworks across both model kinds (discrete categorical and
continuous linear-Gaussian — see [FRAMEWORK_AVAILABILITY.md](FRAMEWORK_AVAILABILITY.md)):

- **PyMDP**: Python Active Inference (discrete models; primary)
- **RxInfer.jl**: Julia Bayesian inference via genuine `@model` + `infer()` (committed `Project.toml` + `Manifest.toml` under `src/gnn/execute/rxinfer/` pin RxInfer 6.1.0; run with `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`, no runtime `Pkg.add`) — discrete and continuous LGSSM lanes
- **ActiveInference.jl**: Complete Julia implementation (discrete models)
- **DisCoPy**: Category theory and quantum computing (discrete models)
- **JAX / NumPyro / PyTorch**: discrete factorized and continuous linear-Gaussian (LGSSM) programs
- **Stan**: HMM (discrete) and LGSSM (continuous) programs via the cmdstanpy driver in `src/gnn/execute/stan/`
- **Lean**: typed document checking via the pinned FEP bridge; numerical and theorem evidence remain separate
- **bnlearn**: Bayesian-network workflows (render/execute; runtime optional)
- **ngc-learn**: admitted continuous/predictive-coding models
- **cpomdp / THRML**: explicitly selected experimental contracts; see [availability](FRAMEWORK_AVAILABILITY.md)

## Integration

This subsystem is part of the broader GNN (Generalized Notation Notation) documentation ecosystem and pipeline.

**Related Components**:
- Setup and dependencies: [docs/SETUP.md](../SETUP.md)
- Framework guides: [pymdp/](../pymdp/), [rxinfer/](../rxinfer/)
- Pipeline orchestration: [src/](../../src/)

## Key Resources

- **Framework Availability**: [FRAMEWORK_AVAILABILITY.md](FRAMEWORK_AVAILABILITY.md)
- **PyMDP Integration**: [pymdp/gnn_pymdp.md](../pymdp/gnn_pymdp.md)
- **RxInfer Integration**: [rxinfer/gnn_rxinfer.md](../rxinfer/gnn_rxinfer.md)
- **Setup Instructions**: [docs/SETUP.md](../SETUP.md)

## Usage

See the [main documentation index](../README.md) for comprehensive guides and framework selection strategies.

---

**Status**: Maintained; acceptance is backend/model/runtime-specific
**Version**: 1.0

