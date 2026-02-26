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
uv sync                    # Core dependencies
uv sync --extra dev        # Development tools
uv sync --all-extras       # Everything
```

## Optional Dependency Groups

These groups match `[project.optional-dependencies]` in `pyproject.toml`:

| Group | Key Packages | Purpose |
| ----- | ------------ | ------- |
| `active-inference` | pymdp, jax, jaxlib, flax, optax | Active Inference simulation |
| `visualization` | plotly, altair, seaborn, bokeh, holoviews | Visualization |
| `llm` | openai, anthropic, cohere, ollama | LLM integration |
| `audio` | librosa, soundfile, pedalboard, pydub | Audio generation |
| `gui` | gradio, streamlit | GUI interface |
| `ml-ai` | torch, transformers, datasets | Machine learning |
| `graphs` | igraph, graphviz, discopy | Graph libraries |
| `dev` | pytest-*, mypy, ruff | Development tools |
| `database` | sqlalchemy, alembic | Database integration |
| `research` | jupyterlab, sympy, numba, cython | Research tools |
| `scaling` | dask, distributed, joblib | Parallel processing |
| `all` | Everything above | Full installation |

## Environment Requirements

- **Python**: 3.11+ (recommended 3.12.x or 3.13.x)
- **Package manager**: uv (recommended) or pip
- **Core deps**: numpy, networkx, pyyaml, jsonschema

## Troubleshooting

| Issue | Solution |
| ----- | ------- |
| `uv: command not found` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| PyMDP import fails | `uv sync --extra active-inference` |
| matplotlib missing | `uv sync --extra visualization` |
| CUDA not detected | Check `torch.cuda.is_available()` after `uv sync --extra ml-ai` |


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `ensure_directory_exists`
- `find_project_gnn_files`
- `get_standard_output_paths`

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
