# bnlearn Renderer — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Step 11 codegen-only render backend for the bnlearn framework: GNN models are
emitted as standalone bnlearn scripts for Bayesian-network structure/parameter
learning plus exact (junction-tree) inference; render is codegen-only — the
emitted script imports `bnlearn`, never the renderer.

## Architecture

```
bnlearn/
├── __init__.py            # Package export (generate_bnlearn_code)
├── bnlearn_renderer.py    # Entry point: standalone-script code generation
├── AGENTS.md              # Backend scaffolding (contract, integration, testing)
├── README.md              # Usage, output, dependency notes
└── SPEC.md                # This specification
```

## Contract

`generate_bnlearn_code(model_data, output_path=None) -> str`

- Emits a standalone Python program importing `bnlearn as bn` (`bn.make_DAG`,
  `bn.parameter_learning.fit`, `bn.inference.fit`) that builds a DAG,
  simulates categorical traces, fits CPTs with MLE, and runs exact inference
- Validation failures return `""`; generation errors are logged and re-raised
- Registry entry (`render.framework_registry`): language Python,
  output_format python, pomdp_compatible True, requires_matrices [],
  available True, supports_execution True, supports_continuous False
- Executed by Step 12 via `src/gnn/execute/bnlearn/` (`BNLEARN_OUTPUT_DIR`);
  scripts skip with an install hint until `uv sync --extra bnlearn`
- Render-time dependencies: none beyond the core pipeline (codegen-only —
  the renderer never imports `bnlearn`)
