# ActiveInference.jl Renderer — Technical Specification

**Version**: 1.6.0

## Purpose

Generates Julia code using the ActiveInference.jl package API.

## Code Generation

- Parses GNN model structure
- Maps to ActiveInference.jl generative model constructor
- Generates `init_model()`, `run_inference()`, `save_results()` functions

## Output

- Julia script files (`.jl`) only — no TOML sidecars

## Continuous models

Discrete POMDPs only (`supports_continuous: false` in `framework_registry.py`):
continuous linear-Gaussian specs report status `unsupported`.

## Template System

- Uses string template engine for Julia code generation
- Parameterized on model dimensions and structure

## Dependencies

Target: `julia >= 1.8`, `ActiveInference.jl >= 0.1.0`
