# GNN Pipeline - Complete Setup Guide

## Quick Start (Cold Start Installation)

### Prerequisites

- **Python 3.11+** installed
- **UV** package manager installed

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Basic Setup (Core Dependencies)

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Run basic setup
python3 src/1_setup.py --verbose
```

A normal `uv sync` / core install includes:

- Scientific stack: numpy, matplotlib, networkx, PyYAML, psutil, httpx
- Active Inference: `inferactively-pymdp`
- **LLM (Step 13+)**: `openai`, `ollama` (client), `python-dotenv`, `aiohttp` (core dependencies — no extra required)
- Dev tooling when using `--extra dev`: pytest, ruff, black, etc.

### Complete Setup (With Optional Packages)

For full functionality including Active Inference, machine learning, and visualization:

```bash
# Install all optional packages
uv sync --all-extras
```

Or install specific groups (canonical list in `pyproject.toml` → `[project.optional-dependencies]`):

```bash
# Install machine-learning extras (transformers, scipy, scikit-learn)
uv sync --extra ml-ai

# Install audio processing extras
uv sync --extra audio

# Install GUI interfaces
uv sync --extra gui
```

## Optional Package Groups

The GNN pipeline declares the following optional package groups (all installable
with `uv sync --extra <group>` or together with `uv sync --all-extras`):

### 1. **dev** - Development Tooling

- **Packages**: pytest, pytest-cov, pytest-xdist, ruff, mypy, black, isort,
  flake8, pylint, bandit, sphinx, jupyterlab, py-spy, and related dev tools
- **Use case**: Testing, linting, type checking, documentation builds
- **Installation**: `uv sync --extra dev`

### 2. **api** - REST API Server

- **Packages**: `fastapi`, `uvicorn[standard]`
- **Use case**: Running the GNN REST API server (`src/api/`)
- **Installation**: `uv sync --extra api`

### 3. **ml-ai** - Machine Learning

- **Packages**: `transformers`, `scipy`, `scikit-learn`
- **Use case**: Deep learning and scientific computing beyond the core Step 12
  backends (PyTorch is intentionally not locked as a dependency while
  GHSA-rrmf-rvhw-rf47 has no patched release; install `torch` manually if you
  need the PyTorch backend)
- **Installation**: `uv sync --extra ml-ai`

### 4. **audio** - Audio Processing & Sonification

- **Packages**: `librosa`, `soundfile`, `pedalboard`
- **Use case**: Audio analysis, sonification of model dynamics (Step 15)
- **Installation**: `uv sync --extra audio`

### 5. **gui** - Interactive GUI

- **Packages**: `gradio`, `streamlit`
- **Use case**: GUI interfaces for model construction (Step 22)
- **Installation**: `uv sync --extra gui`

### 6. **graphs** - Graphviz Bindings

- **Packages**: `graphviz`
- **Use case**: System Graphviz graph layouts (Steps 8-9)
- **Installation**: `uv sync --extra graphs`

### 7. **research** - Research & Notebooks

- **Packages**: `sympy`, `numba`, `cython`, `jupyterlab`, `jupyter-server`, `bleach`
- **Use case**: Notebook-based research workflows for non-developer users
- **Installation**: `uv sync --extra research`

### 8. **scaling** - Distributed Execution

- **Packages**: `dask`, `distributed`, `ray`
- **Use case**: Distributed execution of parameter sweeps (`execute/distributed.py`)
- **Installation**: `uv sync --extra scaling`

### 9. **all** - Everything

- **Packages**: union of every group above
- **Installation**: `uv sync --all-extras`

> **Core vs optional**: `inferactively-pymdp`, `jax[cpu]`, `jaxlib`, `flax`,
> `optax`, `numpyro`, and `discopy` are **core** dependencies installed by a
> plain `uv sync` — no extra is required for the Step 12 Python backends.
> Julia backends (RxInfer.jl, ActiveInference.jl) additionally require a local
> Julia installation.

## Installation Methods

**Note**: The recommended installation method uses UV's built-in extras system via `pyproject.toml`.

### Method 1: Using UV Extras (Recommended)

```bash
# List available groups (shown in pyproject.toml [project.optional-dependencies])
# Groups: dev, api, ml-ai, audio, gui, graphs, research, scaling, all

# Install all optional packages
uv sync --all-extras

# Install specific groups
uv sync --extra ml-ai --extra audio --extra gui
```

### Method 2: Using Setup Module

```bash
# Install via the setup step
python3 src/1_setup.py --verbose
```

### Method 3: Using UV Directly

```bash
# Install packages using UV
uv pip install inferactively-pymdp
uv pip install "jax[cpu]" optax flax
uv pip install plotly seaborn bokeh
```

## Platform-Specific Notes

### Linux (Tested on Parrot OS / Debian)

- All packages install successfully using UV
- No additional system dependencies required for core functionality
- PyMDP and JAX work perfectly with CPU backend

### macOS

- All packages install successfully using UV
- Same behavior as Linux
- Native M1/M2 support with JAX

### Windows

- Core packages install successfully
- Some optional packages may require Windows-specific builds
- Recommended: Use WSL2 for best compatibility

## Verification

After installation, verify that packages are working:

```python
# Test PyMDP
from pymdp.agent import Agent

print("✅ PyMDP working!")

# Test JAX
import jax.numpy as jnp

print("✅ JAX working!")

# Test visualization
import plotly.express as px

print("✅ Plotly working!")
```

## Running the Pipeline

### Full Pipeline

```bash
python3 src/main.py --target-dir input/gnn_files --verbose
```

### Specific Steps

```bash
# GNN parsing, rendering, and execution
python3 src/main.py --only-steps "3,11,12" --verbose
```

### Individual Steps

```bash
# Just GNN parsing
python3 src/3_gnn.py --target-dir input/gnn_files --verbose
```

## Performance Metrics

### Installation Times (on modern hardware)

- **Core dependencies**: 30-60 seconds
- **JAX**: 30-45 seconds
- **PyMDP**: 10-15 seconds
- **Visualization**: 20-30 seconds
- **Audio**: 40-60 seconds
- **LLM**: 10-20 seconds
- **ML (PyTorch)**: 2-4 minutes

### Pipeline Execution

- **Full 25-step pipeline**: ~5 minutes (with LLM step)
- **GNN parsing**: ~130ms
- **Code rendering**: ~150ms
- **Execution (with PyMDP/JAX)**: ~16 seconds

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'psutil'"

**Solution**: Run setup to install core dependencies

```bash
python3 src/1_setup.py --verbose
```

### Issue: "externally-managed-environment" error with pip

**Solution**: Always use UV pip with --python flag

```bash
uv pip install package_name --python .venv/bin/python
```

### Issue: JAX not using GPU

**Solution**: Install CUDA-enabled JAX (optional)

```bash
uv pip install jax[cuda] --python .venv/bin/python
```

### Issue: PyMDP import errors

**Solution**: Ensure correct package name

```bash
uv pip install inferactively-pymdp --python .venv/bin/python
```

## Package Versions (Current)

Versions are pinned in `uv.lock`; `pyproject.toml` declares the floor
constraints. Notable floors as of 2026-08-02:

### Core Dependencies

- Python: 3.11+ (`>=3.11,<3.14`)
- `inferactively-pymdp>=1.0.0` (JAX-first rewrite)
- `jax[cpu]>=0.7.0,<0.10`, `jaxlib`, `flax`, `optax`, `numpyro>=0.14`
- numpy, matplotlib, networkx, pyyaml, pandas, plotly, seaborn, h5py

### Optional Dependencies

- See `[project.optional-dependencies]` in `pyproject.toml` and `uv.lock`
  for the resolved versions of `dev`, `api`, `ml-ai`, `audio`, `gui`,
  `graphs`, `research`, and `scaling` groups.

> Run `uv lock --check` and `uv sync --frozen` for the authoritative resolved
> set.

## References

- **UV Documentation**: <https://docs.astral.sh/uv/>
- **PyMDP Repository**: <https://github.com/infer-actively/pymdp>
- **JAX Documentation**: <https://jax.readthedocs.io/>
- **GNN Pipeline Documentation**: See `README.md` and `ARCHITECTURE.md`

---

**Last Updated**: 2026-08-02
**Pipeline Version**: 3.0.0
**Status**: ✅ Production Ready (Linux & macOS)
**Validation authority**: use the command of record and latest dated receipt in
[`README.md`](README.md). Julia RxInfer execution uses the committed
`Project.toml` under `src/execute/rxinfer/`; local Ollama tests remain an
explicit opt-in surface.
