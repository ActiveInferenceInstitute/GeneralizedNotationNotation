# Step 1: Setup — Environment and Dependency Management

## Overview

Handles project initialization, UV virtual environment creation, and dependency installation. Supports core-only and optional dependency groups.

## Usage

```bash
# Core dependencies only
python src/1_setup.py --target-dir input/gnn_files --output-dir output --verbose

# Optional groups (LLM PyPI packages are already in core dependencies)
python src/1_setup.py --install-optional --optional-groups=llm --verbose

# Install all optional dependencies
python src/1_setup.py --install-optional --verbose

# Direct UV usage
uv sync                       # Core includes openai, ollama, dotenv, aiohttp
uv sync --extra llm           # Same LLM pins (compatibility extra)
uv sync --extra visualization  # Visualization packages
uv sync --extra all            # All optional packages
```

## Architecture

| Component | Path |
|-----------|------|
| Orchestrator | `src/1_setup.py` (148 lines) |
| Module | `src/setup/` |
| Module functions | `setup_uv_environment()`, `setup_complete_environment()`, `install_optional_package_group()` |

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--recreate-venv` | `bool` | Recreate virtual environment from scratch |
| `--dev` | `bool` | Install development dependencies (default: `True`) |
| `--install-optional` | `bool` | Install optional dependency groups |
| `--optional-groups` | `str` | Comma-separated groups: `jax,pymdp,visualization,audio,llm,ml` |

## Notes

This step uses a `setup_orchestrator()` wrapper (not direct delegation) to parse optional groups and route between basic vs. full setup paths.

## Source

- **Script**: [src/1_setup.py](../../../src/1_setup.py)
