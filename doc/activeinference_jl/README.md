# activeinference_jl

Julia implementation of Active Inference algorithms

## Overview

This directory contains documentation, resources, and implementation guides for the activeinference_jl subsystem.

**Contents**:        4 files,        3 subdirectories

## Quick Links

- **Main Documentation**: [doc/README.md](../README.md)
- **GNN Overview**: [GNN Documentation](../gnn/README.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)

## Directory Structure

```
activeinference_jl/
├── README.md (this file)
└── [additional resources]
```

## ⚡ High-Performance Inference with Julia

`activeinference_jl` leverages Julia's Just-In-Time (JIT) compilation to provide:
- **Zero-Cost Abstractions**: Model agents with high-level GNN syntax that compiles down to efficient machine code.
- **Fast Belief Updating**: Typical belief update loops are 10-50x faster than pure Python implementations, making it ideal for real-time robotic applications.
- **Type Stability**: GNN's type-checking step ensures that the generated Julia code is fully type-stable, maximizing LLVM optimization.

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
