# Dependencies Management Documentation Agents

## Overview

This document provides agent scaffolding for the dependencies subsystem documentation. The documentation covers package management, optional dependencies, and framework selection strategies.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Purpose

Dependency management, package versioning, and optional installation documentation for the GNN pipeline.

## Documentation Organization

The dependencies documentation system is part of the broader GNN (Generalized Notation Notation) pipeline and ecosystem.

### Related Systems
- **Main Documentation**: [doc/README.md](../README.md)
- **Setup Guide**: [doc/SETUP.md](../SETUP.md)
- **Pipeline Reference**: [src/AGENTS.md](../../src/AGENTS.md)
- **Active Inference**: [activeinference.org](https://activeinference.org)

## Subsystems

- **OPTIONAL_DEPENDENCIES.md**: Optional packages and installation options
- **Core Dependencies**: NumPy, SciPy, Matplotlib, Pandas, PyTest
- **Active Inference**: PyMDP, ActiveInference.jl, RxInfer.jl
- **Visualization**: DisCoPy, Altair, Graphviz
- **GPU Acceleration**: JAX, CuPy, TensorFlow

## Quick Navigation

- **Get Started**: See [README.md](README.md)
- **Optional Packages**: [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md)
- **Setup Guide**: [doc/SETUP.md](../SETUP.md)
- **Full Documentation Index**: [doc/README.md](../README.md)

## Documentation Standards

All documentation in this subsystem follows the professional standards established in the GNN documentation system:

- **Clarity**: Concrete examples, technical accuracy
- **Functionality**: Shows what code actually does
- **Evidence-Based**: Specific metrics and real examples
- **Professional**: Clear structure, no promotional language

## Integration

This documentation integrates with the 25-step GNN processing pipeline:

1. **Setup** (Step 1): Dependency installation and virtual environment creation
2. **Testing** (Step 2): Validation of installed dependencies
3. **All Steps**: Each pipeline step uses required dependencies
4. **Execution** (Step 12): Framework-specific dependencies for simulation
5. **Optional Features**: Advanced features requiring additional packages

See [src/AGENTS.md](../../src/AGENTS.md) for complete pipeline documentation.

---

**Status**: ✅ Production Ready  
**Compliance**: Professional documentation standards  
**Maintenance**: Regular updates with new package versions

