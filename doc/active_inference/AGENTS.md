# active_inference

## Overview

This directory contains comprehensive documentation for Active Inference and the Free Energy Principle as implemented within the GNN (Generalized Notation Notation) framework.

**Status**: ✅ Documentation Module  
**Version**: 1.0  
**Last Updated**: January 2026

---

## Purpose

Central documentation hub for all theoretical, computational, and implementation aspects of Active Inference within the GNN ecosystem. This module provides signposting to all related documentation and source code.

---

## Contents

**Files**: 15 | **Subdirectories**: 0

### Navigation Files

| File | Description | Lines |
|------|-------------|-------|
| [README.md](README.md) | Directory overview and navigation | ~250 |
| [AGENTS.md](AGENTS.md) | Technical scaffolding (this file) | ~200 |

### Theory & Foundations

| File | Description |
|------|-------------|
| [fep_foundations.md](fep_foundations.md) | Free Energy Principle theoretical foundations |
| [active_inference_theory.md](active_inference_theory.md) | Core Active Inference theory and concepts |
| [variational_inference.md](variational_inference.md) | Variational Free Energy and Bayesian inference |
| [expected_free_energy.md](expected_free_energy.md) | Expected Free Energy and policy selection |
| [generative_models.md](generative_models.md) | A, B, C, D matrices and model specification |
| [pomdp_foundations.md](pomdp_foundations.md) | POMDP formalism and Active Inference |

### Implementation References

| File | Description |
|------|-------------|
| [implementation_pymdp.md](implementation_pymdp.md) | PyMDP Python implementation |
| [implementation_rxinfer.md](implementation_rxinfer.md) | RxInfer.jl message passing |
| [implementation_activeinference_jl.md](implementation_activeinference_jl.md) | ActiveInference.jl framework |
| [computational_patterns.md](computational_patterns.md) | Common algorithmic patterns |

### Integration & Applications

| File | Description |
|------|-------------|
| [gnn_integration.md](gnn_integration.md) | GNN syntax for Active Inference models |
| [applications_examples.md](applications_examples.md) | Use cases and practical examples |
| [glossary.md](glossary.md) | Terminology reference |

---

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Start here for overview
- **[glossary.md](glossary.md)**: Terminology reference

### Theory Foundations
- **[fep_foundations.md](fep_foundations.md)**: FEP theory
- **[active_inference_theory.md](active_inference_theory.md)**: Active Inference core
- **[variational_inference.md](variational_inference.md)**: VFE details
- **[expected_free_energy.md](expected_free_energy.md)**: EFE and action

### Implementation
- **[implementation_pymdp.md](implementation_pymdp.md)**: Python via PyMDP
- **[implementation_rxinfer.md](implementation_rxinfer.md)**: Julia via RxInfer
- **[implementation_activeinference_jl.md](implementation_activeinference_jl.md)**: Julia via ActiveInference.jl

---

## Source Code Signposting

### Execution Engines

| Component | Source Path | Documentation |
|-----------|-------------|---------------|
| PyMDP Runner | [`src/execute/pymdp/`](../../src/execute/pymdp/) | [implementation_pymdp.md](implementation_pymdp.md) |
| RxInfer Runner | [`src/execute/rxinfer/`](../../src/execute/rxinfer/) | [implementation_rxinfer.md](implementation_rxinfer.md) |
| ActiveInference.jl | [`src/execute/activeinference_jl/`](../../src/execute/activeinference_jl/) | [implementation_activeinference_jl.md](implementation_activeinference_jl.md) |

### Analysis Tools

| Component | Source Path |
|-----------|-------------|
| Main Analyzer | [`src/analysis/analyzer.py`](../../src/analysis/analyzer.py) |
| PyMDP Analyzer | [`src/analysis/pymdp_analyzer.py`](../../src/analysis/pymdp_analyzer.py) |
| PyMDP Visualizer | [`src/analysis/pymdp_visualizer.py`](../../src/analysis/pymdp_visualizer.py) |
| Post-Simulation | [`src/analysis/post_simulation.py`](../../src/analysis/post_simulation.py) |
| ActiveInference.jl Analyzer | [`src/analysis/activeinference_jl/`](../../src/analysis/activeinference_jl/) |

### Core GNN

| Component | Source Path |
|-----------|-------------|
| GNN Parser | [`src/gnn/`](../../src/gnn/) |
| Type Checker | [`src/type_checker/`](../../src/type_checker/) |
| Validation | [`src/validation/`](../../src/validation/) |
| Export | [`src/export/`](../../src/export/) |

---

## Related Documentation Signposting

### Implementation Guides

| Resource | Path | Description |
|----------|------|-------------|
| PyMDP | [`doc/pymdp/`](../pymdp/) | Complete PyMDP documentation |
| RxInfer | [`doc/rxinfer/`](../rxinfer/) | RxInfer.jl documentation |
| ActiveInference.jl | [`doc/activeinference_jl/`](../activeinference_jl/) | ActiveInference.jl docs |
| POMDP | [`doc/pomdp/`](../pomdp/) | POMDP theoretical foundations |

### GNN Core

| Resource | Path | Description |
|----------|------|-------------|
| GNN Overview | [`doc/gnn/gnn_overview.md`](../gnn/gnn_overview.md) | Core GNN concepts |
| GNN Syntax | [`doc/gnn/gnn_syntax.md`](../gnn/gnn_syntax.md) | Syntax specification |
| Neurosymbolic | [`doc/gnn/gnn_llm_neurosymbolic_active_inference.md`](../gnn/gnn_llm_neurosymbolic_active_inference.md) | LLM + Active Inference |
| GNN Examples | [`doc/gnn/gnn_examples_doc.md`](../gnn/gnn_examples_doc.md) | Example models |

### LLM Integration

| Resource | Path | Description |
|----------|------|-------------|
| DSPy | [`doc/dspy/`](../dspy/) | DSPy integration |
| LLM | [`doc/llm/`](../llm/) | LLM documentation |

---

## Pipeline Integration

This documentation integrates with the 24-step GNN processing pipeline:

### Relevant Pipeline Steps

| Step | Name | Relevance |
|------|------|-----------|
| 3 | GNN | Model parsing |
| 5 | Type Checker | Type validation |
| 6 | Validation | Model validation |
| 12 | Execute | Active Inference execution |
| 13 | LLM | LLM-enhanced analysis |
| 16 | Analysis | Post-simulation analysis |

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

---

## Key Concepts Reference

| Concept | Symbol | Description |
|---------|--------|-------------|
| Variational Free Energy | F, VFE | Upper bound on surprise |
| Expected Free Energy | G, EFE | Future free energy for policies |
| Likelihood Matrix | A | P(o\|s) - observations given states |
| Transition Matrix | B | P(s'\|s,a) - state transitions |
| Preference Vector | C | ln P(o) - preferred observations |
| Initial State Prior | D | P(s₀) - initial belief |
| Precision | γ, β | Inverse variance / confidence |

---

## Standards and Guidelines

All documentation in this module adheres to:

- **Technical Accuracy**: Verified mathematical formulations
- **Signposting**: Clear links to source code and related docs
- **Modularity**: Self-contained documents with cross-references
- **Consistency**: Uniform notation and terminology
- **GNN Standards**: Following project documentation standards

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new research
