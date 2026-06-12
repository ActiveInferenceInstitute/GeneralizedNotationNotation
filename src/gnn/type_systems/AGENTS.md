# GNN Type System Definitions

## Overview

Contains type system implementations and mappings for GNN models. Defines categorical, functional, and algebraic type annotations used during type checking (Step 5).

## Architecture

```
type_systems/
├── __init__.py           # Package marker
├── categorical.scala     # Categorical type system (Scala DSL)
├── haskell.hs            # Haskell algebraic type definitions
├── scala.scala           # Scala type annotations
├── mapping.md            # Type mapping documentation (cross-system)
└── examples/             # Example type annotations
```

## Purpose

- **Type annotations** — Define the type algebra used by `type_checker/processor.py`.
- **Cross-language mapping** — `mapping.md` documents how types translate across Haskell, Scala, and Python representations.
- **Categorical semantics** — `categorical.scala` defines functorial and natural transformation types for DisCoPy integration.

## Parent Module

See [gnn/AGENTS.md](../AGENTS.md) for the overall GNN processing architecture.

**Version**: 1.6.0
