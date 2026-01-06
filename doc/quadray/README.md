# quadray

Quadray coordinate system for spatial models

## Overview

This directory contains documentation, resources, and implementation guides for the quadray subsystem.

**Contents**:        4 files,        1 subdirectories

## Quick Links

- **Main Documentation**: [doc/README.md](../README.md)
- **GNN Overview**: [GNN Documentation](../gnn/README.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Directory Structure

```
quadray/
├── README.md (this file)
└── [additional resources]
```

## 📐 4D Spatial Priors in GNN

Quadray coordinates provide a unique 4D perspective on 3D space:
- **Symmetric Spatial Priors**: Encode spatial layout using Quadray symmetry, which can simplify the Dirichlet priors `d_f0` in navigation models.
- **Direct Geometry Mapping**: Translate 3D movements into GNN policies via Quadray transformation matrices.

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
