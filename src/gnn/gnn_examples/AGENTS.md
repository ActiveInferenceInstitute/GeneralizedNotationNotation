# GNN Examples Module

## Purpose

This module contains **reference GNN model files** that serve as both documentation and test fixtures. The primary example is a complete Active Inference POMDP agent specification.

## Example Files

| File | Description |
|------|-------------|
| `actinf_pomdp_agent.md` | Complete GNN specification for a discrete POMDP Active Inference agent with 4 hidden states, 5 observations, 3 control states, and full parameterization |

## Key Sections in the POMDP Example

The reference file demonstrates all required GNN sections:

- **StateSpaceBlock**: Defines hidden states (s), observations (o), and control states (u) with dimensionality annotations
- **Connections**: Maps state-observation relationships using GNN edge notation (`->`, `<->`)
- **InitialParameterization**: Provides matrices A, B, C, D with full numeric values
- **Equations**: Active Inference update equations (variational free energy, posterior, expected free energy)
- **Time**: Dynamic/discrete temporal configuration

## For AI Agents

1. **Use as reference** when parsing or generating GNN files — this is the canonical example
2. **Use for testing** — this file is the reference for round-trip validation across all 23 formats
3. **Model structure**: The POMDP agent demonstrates proper Active Inference variable naming (A=likelihood, B=transition, C=preference, D=prior, E=policy prior)
