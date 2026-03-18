# Stan Framework Implementation

> **GNN Integration Layer**: Stan probabilistic programming
> **Framework Base**: Stan (statistical modeling language)
> **Documentation Version**: 2.0.0

## Overview

The Stan renderer generates **Stan model code** from parsed GNN variables and connections. It uses simple structural heuristics to classify variables into `data` and `parameters` blocks and emits directed connection comments. This backend produces valid Stan syntax from GNN structure; it does not encode full Active Inference semantics.

## Architecture

| Stage | Module | Description |
|-------|--------|-------------|
| Rendering (Step 11) | `src/render/stan/stan_renderer.py` | GNN variables/connections → Stan program string |

## Usage

The render module exposes Stan via the standard pipeline. The API returns Stan code as a string (`.stan` file content). The parent render step writes output to per-model/per-framework directories when Stan is selected.

## Implementation Notes

- Variables are classified into `data {}`, `parameters {}`, and `model {}` blocks.
- Directed edges are reflected as comments; a default likelihood may be emitted.
- For full backend details, see **[src/render/stan/AGENTS.md](../../../src/render/stan/AGENTS.md)** and **[src/render/stan/README.md](../../../src/render/stan/README.md)**.

## Navigation

- [← GNN Implementations Index](README.md)
- [← GNN Main Index](../README.md)
