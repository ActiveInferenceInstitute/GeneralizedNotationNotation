---
name: gnn-code-generation
description: GNN code generation for simulation frameworks. Use when generating PyMDP, RxInfer.jl, ActiveInference.jl, JAX, or DisCoPy simulation code from GNN model specifications.
---

# GNN Code Generation / Render (Step 11)

## Purpose

Generates executable simulation code from parsed GNN models targeting multiple Active Inference frameworks. Performs pre-render validation of POMDP structures and matrix normalization.

## Key Commands

```bash
# Render code for all frameworks
python src/11_render.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 11 --verbose
```

## Supported Frameworks

| Framework | Language | Subdirectory | Output |
| ----------- | ---------- | ------------- | -------- |
| **PyMDP** | Python | `render/pymdp/` | `.py` scripts |
| **RxInfer.jl** | Julia | `render/rxinfer/` | `.jl` scripts |
| **ActiveInference.jl** | Julia | `render/activeinference_jl/` | `.jl` scripts |
| **JAX** | Python | `render/jax/` | `.py` scripts |
| **DisCoPy** | Python | `render/discopy/` | `.py` scripts |

## API

```python
from render import (
    process_render, render_gnn_spec, get_supported_frameworks,
    generate_pymdp_code, generate_rxinfer_code,
    generate_activeinference_jl_code, generate_discopy_code,
    validate_render, PyMDPRenderer, JAXRenderer
)

# Render for all frameworks (used by pipeline)
process_render(target_dir, output_dir, verbose=True)

# Render a single GNN spec
result = render_gnn_spec(parsed_spec, framework="pymdp")

# Framework-specific code generation
pymdp_code = generate_pymdp_code(parsed_model)
rxinfer_code = generate_rxinfer_code(parsed_model)

# Query supported frameworks
frameworks = get_supported_frameworks()  # ['pymdp', 'rxinfer', 'activeinference_jl', 'jax', 'discopy']

# Validate render output
validate_render(result, framework="pymdp")
```

## Key Exports

- `process_render` / `render_gnn_spec` — core rendering functions
- `generate_pymdp_code`, `generate_rxinfer_code`, `generate_discopy_code`, `generate_activeinference_jl_code`
- `PyMDPRenderer` / `JAXRenderer` — renderer classes
- `get_supported_frameworks` / `validate_render` — utilities

## POMDP Processing Pipeline

The render processor follows a structured pipeline for each GNN file:

1. **POMDP Extraction** — extracts state spaces (A, B, C, D matrices) from GNN specs
2. **Validation** — validates dimensional consistency and completeness
3. **Matrix Normalization** — ensures A-matrix columns and B-matrix transition rows sum to 1.0
4. **Framework Code Generation** — generates framework-specific executable code
5. **Documentation** — creates structured overview of rendering results

## Dependencies

```bash
# Core rendering (no extra deps needed)
uv sync

# For Julia frameworks (RxInfer.jl, ActiveInference.jl)
# Requires Julia installed and Julia packages: RxInfer, ActiveInference
julia -e 'using Pkg; Pkg.add(["RxInfer", "ActiveInference"])'

# For DisCoPy
uv sync --extra graphs
```

## Output

- Generated scripts in `output/11_render_output/`
- Per-framework subdirectories: `pymdp/`, `rxinfer/`, `jax/`, `discopy/`, `activeinference_jl/`
- One script per model per framework
- Render overview documentation

## Troubleshooting

| Issue | Solution |
| ----- | ------- |
| Empty render output | Check GNN file has valid `StateSpaceBlock` section |
| Matrix normalization warnings | Verify matrix dimensions match connections |
| Julia framework errors | Ensure Julia is installed and packages available |


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_render_module_info`
- `list_render_frameworks`
- `process_render`
- `render_gnn_to_format`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
