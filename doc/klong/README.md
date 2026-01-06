# klong

Klong language integration

## Overview

This directory contains documentation, resources, and implementation guides for the klong subsystem.

**Contents**:        1 files,        1 subdirectories

## Quick Links

- **Main Documentation**: [doc/README.md](../README.md)
- **GNN Overview**: [GNN Documentation](../gnn/README.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Directory Structure

```
klong/
├── README.md (this file)
└── [additional resources]
```

## 🔢 Array-Oriented Active Inference

Klong's tacit, array-oriented nature is ideal for optimizing the tensor operations inherent in Active Inference:
- **Tacit Belief Updating**: Express categorical belief updates as simple array operations without explicit loops.
- **Direct Matrix Translation**: Map GNN `A` and `B` matrices directly to Klong arrays for ultra-fast likelihood matching.

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
