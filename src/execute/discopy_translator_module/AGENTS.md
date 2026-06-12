# DisCoPy Translator Execution Sub-module

## Overview

Translates between DisCoPy categorical representations and other execution frameworks (particularly JAX). Provides visualization of JAX outputs within the categorical framework.

## Architecture

```
discopy_translator_module/
├── __init__.py                  # Package exports (26 lines)
├── translator.py                # DisCoPy ↔ framework translation (234 lines)
└── visualize_jax_output.py      # JAX output visualization (270 lines)
```

## Key Functions

- **`translate_to_jax(diagram)`** — Converts DisCoPy diagrams to JAX-compatible tensor networks.
- **`visualize_jax_output(results, output_dir)`** — Renders JAX computation results as categorical visualizations.
- **Cross-framework bridging** — Enables consistent analysis across categorical and numerical representations.

## Dependencies

- `discopy`, `jax` (optional, graceful degradation)

## Parent Module

See [execute/AGENTS.md](../AGENTS.md) for the overall execution architecture.

**Version**: 1.6.0
