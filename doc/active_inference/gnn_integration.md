# GNN Integration with Active Inference

> **📋 Document Metadata**  
> **Type**: Integration Reference | **Audience**: Developers | **Complexity**: Intermediate  
> **Cross-References**: [GNN Documentation](../gnn/README.md) | [Generative Models](generative_models.md) | [DSPy Integration](../dspy/dspy_gnn_integration_patterns.md)

## Overview

This document describes how GNN (Generalized Notation Notation) integrates with Active Inference implementations, providing signposting to all relevant documentation and source code.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## GNN Documentation Signposting

### Core GNN Documentation

| Document | Path | Description |
|----------|------|-------------|
| **GNN Overview** | [`doc/gnn/gnn_overview.md`](../gnn/gnn_overview.md) | Core concepts |
| **GNN Syntax** | [`doc/gnn/gnn_syntax.md`](../gnn/gnn_syntax.md) | Syntax specification |
| **GNN Examples** | [`doc/gnn/gnn_examples_doc.md`](../gnn/gnn_examples_doc.md) | Example models |
| **GNN Schema** | [`doc/gnn/gnn_schema.md`](../gnn/gnn_schema.md) | Schema definition |
| **Type System** | [`doc/gnn/gnn_type_system.md`](../gnn/gnn_type_system.md) | Type specification |

### Neurosymbolic Integration

| Document | Path | Description |
|----------|------|-------------|
| **LLM + Active Inference** | [`doc/gnn/gnn_llm_neurosymbolic_active_inference.md`](../gnn/gnn_llm_neurosymbolic_active_inference.md) | Comprehensive guide |
| **DSPy Integration** | [`doc/dspy/dspy_gnn_integration_patterns.md`](../dspy/dspy_gnn_integration_patterns.md) | DSPy patterns |

---

## Source Code Signposting

### GNN Core

| Component | Path | Description |
|-----------|------|-------------|
| **GNN Parser** | [`src/gnn/`](../../src/gnn/) | Model parsing |
| **Type Checker** | [`src/type_checker/`](../../src/type_checker/) | Type validation |
| **Validation** | [`src/validation/`](../../src/validation/) | Model validation |
| **Export** | [`src/export/`](../../src/export/) | Format export |

### Execution Engines

| Engine | Path | Description |
|--------|------|-------------|
| **PyMDP** | [`src/execute/pymdp/`](../../src/execute/pymdp/) | Python execution |
| **RxInfer** | [`src/execute/rxinfer/`](../../src/execute/rxinfer/) | Julia execution |
| **ActiveInference.jl** | [`src/execute/activeinference_jl/`](../../src/execute/activeinference_jl/) | Julia execution |

### Analysis

| Tool | Path |
|------|------|
| **Analyzer** | [`src/analysis/`](../../src/analysis/) |
| **Visualization** | [`src/visualization/`](../../src/visualization/) |

---

## GNN Model Structure

### Active Inference Components

```gnn
// Model definition
ModelName: "NavigationAgent"
StateSpace: discrete
TimestepCount: 10

// Generative Model
A[num_obs, num_states] = likelihood_matrix    // Observations given states
B[num_states, num_states, num_actions] = transitions  // State dynamics
C[num_obs] = preferences                       // Goal preferences
D[num_states] = initial_prior                  // Initial beliefs
E[num_policies] = habit_prior                  // Policy prior
```

### Mapping to Active Inference

| GNN Element | Active Inference Role |
|-------------|----------------------|
| `A` matrix | Likelihood P(o\|s) |
| `B` matrix | Transition P(s'\|s,a) |
| `C` vector | Log preferences ln P(o) |
| `D` vector | Initial prior P(s₀) |
| `E` vector | Policy prior P(π) |

---

## Pipeline Integration

### 24-Step Pipeline

| Step | Relevance to Active Inference |
|------|------------------------------|
| **3: GNN** | Model parsing |
| **5: Type Checker** | Matrix dimension validation |
| **6: Validation** | Probabilistic constraints |
| **7: Export** | Format conversion |
| **12: Execute** | Active Inference simulation |
| **13: LLM** | LLM-enhanced analysis |
| **16: Analysis** | Post-simulation analysis |

---

## Quick Reference

### Model Definition Example

```gnn
// Simple T-Maze Agent
ModelName: "TMazeAgent"
StateSpace: discrete
States: ["center", "left", "right", "cue_left", "cue_right"]
Observations: ["null", "reward", "cue_left", "cue_right"]
Actions: ["stay", "go_left", "go_right"]

A[4,5] = [
    [1.0, 0.0, 0.0, 0.0, 0.0],  // null
    [0.0, 0.9, 0.1, 0.0, 0.0],  // reward
    [0.0, 0.0, 0.0, 0.9, 0.1],  // cue_left
    [0.0, 0.0, 0.0, 0.1, 0.9]   // cue_right
]

B[5,5,3] = transition_matrices

C[4] = [0.0, 2.0, 0.0, 0.0]  // Prefer reward

D[5] = [1.0, 0.0, 0.0, 0.0, 0.0]  // Start center
```

---

## Related Documentation

### Active Inference Theory
- **[FEP Foundations](fep_foundations.md)**
- **[Active Inference Theory](active_inference_theory.md)**
- **[Generative Models](generative_models.md)**

### Implementation
- **[PyMDP](implementation_pymdp.md)**
- **[RxInfer](implementation_rxinfer.md)**
- **[Computational Patterns](computational_patterns.md)**

### GNN
- **[GNN README](../gnn/README.md)**
- **[GNN AGENTS.md](../gnn/AGENTS.md)**

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards
