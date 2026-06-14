---
name: gnn-environment-setup
description: GNN environment setup and dependency management. Use when configuring the development environment, installing dependencies, managing virtual environments, or troubleshooting dependency issues for the GNN pipeline.
---

# GNN Environment Setup (Step 1)

## Purpose

Manages environment configuration, virtual environment creation, and dependency installation for the GNN pipeline. Supports both core and optional dependency groups via `uv` or `pip`.

## Key Commands

```bash
# Run setup step
python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 1 --verbose

# Using uv (recommended)
uv sync                    # Core dependencies (includes Step 12 Python backends: jax, numpyro, discopy)
uv sync --extra dev        # Development tools
uv sync --all-extras       # Everything
uv pip install torch       # Optional PyTorch backend while no patched torch release exists
```

## Optional Dependency Groups

These groups match `[project.optional-dependencies]` in `pyproject.toml`:

| Group | Key Packages | Purpose |
| ----- | ------------ | ------- |
| Core `uv sync` | pymdp, jax, numpyro, discopy, LLM clients, visualization | Standard pipeline runtime |
| `audio` | librosa, soundfile, pedalboard, pydub | Audio generation |
| `gui` | gradio, streamlit | GUI interface |
| `ml-ai` | transformers, scipy, scikit-learn | Machine learning extensions |
| Manual install | torch, bnlearn | Optional backends excluded from the lock while Torch has no patched advisory release |
| `graphs` | graphviz | Graphviz bindings |
| `dev` | pytest-*, mypy, ruff | Development tools |
| `research` | jupyterlab, sympy, numba, cython | Research tools |
| `scaling` | dask, distributed, ray | Parallel processing |
| `all` | Named optional groups above | Full optional installation |

## Environment Requirements

- **Python**: 3.11+ (recommended 3.12.x or 3.13.x)
- **Package manager**: uv (recommended) or pip
- **Core deps**: numpy, networkx, pyyaml, jsonschema

## Troubleshooting

| Issue | Solution |
| ----- | ------- |
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| PyMDP import fails | `uv sync` |
| matplotlib missing | `uv sync` |
| CUDA not detected | Install PyTorch manually, then check `torch.cuda.is_available()` |


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `check_uv_project_status`
- `ensure_directory_exists`
- `find_project_gnn_files`
- `get_standard_output_paths`
- `get_uv_environment_info`
- `install_uv_dependency`
- `setup_uv_project_structure`
- `sync_uv_dependencies`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification
- [../../SETUP_GUIDE.md](../../SETUP_GUIDE.md) — Full setup guide
- [../../pyproject.toml](../../pyproject.toml) — Dependency definitions


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
