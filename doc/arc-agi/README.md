# arc-agi

Abstraction and Reasoning Corpus (ARC) as GNN models

## Overview

This directory contains documentation, resources, and implementation guides for the arc-agi subsystem.

**Contents**:        1 files,        1 subdirectories

## Quick Links

- **Main Documentation**: [doc/README.md](../README.md)
- **GNN Overview**: [GNN Documentation](../gnn/README.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Directory Structure

```
arc-agi/
├── README.md (this file)
└── [additional resources]
```

## 🧩 Modeling ARC with GNN

To model ARC-AGI tasks using GNN, use the following template strategy:
1. **Grid as State Space**: Define `s_f0` as a flattened categorical factor representing grid cell colors.
2. **Symmetry Operators as Transition Matrices**: Map grid rotations and flips to specific `B` matrices.
3. **Object-Oriented priors**: Use the GNN `ModelAnnotation` to specify object-level constraints that guide policy inference.

## Integration

This subsystem is part of the broader GNN (Generalized Notation Notation) documentation ecosystem.

**Related Components**:
- Main documentation system: [doc/](../)
- Pipeline modules: [src/](../../src/)
- Active Inference Institute: [activeinference.org](https://activeinference.org)

## Usage

See the [main documentation index](../README.md) for comprehensive guides and tutorials.

---

**Status**: Documentation  
**Version**: 1.0
