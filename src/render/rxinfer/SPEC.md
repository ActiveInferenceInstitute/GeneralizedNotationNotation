# RxInfer Renderer — Technical Specification

**Version**: 1.6.0

## Purpose

Generates Julia code using the RxInfer.jl reactive message-passing framework.

## Code Generation

- Maps GNN model structure to RxInfer factor graph specification
- Generates probabilistic model definition and inference queries
- Produces TOML configuration for model parameters

## Output

- Julia script files (`.jl`) using RxInfer API
- TOML parameter files

## Architecture

```
rxinfer/
├── rxinfer_renderer.py     # Core renderer (625 lines)
├── toml_generator.py       # TOML config generation (995 lines)
└── ...
```

## Dependencies

Target: `julia >= 1.8`, `RxInfer.jl >= 3.0`
