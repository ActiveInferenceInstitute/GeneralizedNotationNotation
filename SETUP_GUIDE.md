# GNN Pipeline - Complete Setup Guide

## Quick Start (Cold Start Installation)

### Prerequisites

- **Python 3.11+** installed
- **UV** package manager installed

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Basic Setup (Core Dependencies Only)

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Run basic setup
python3 src/1_setup.py --verbose
```

This installs **162 core packages** including:

- Scientific computing: numpy, scipy, pandas, matplotlib
- Testing: pytest, pytest-cov, pytest-asyncio
- Development: black, isort, flake8, mypy
- Pipeline essentials: networkx, pyyaml, psutil, jupyter

### Complete Setup (With Optional Packages)

For full functionality including Active Inference, machine learning, and visualization:

```bash
# Install all optional packages
uv sync --all-extras
```

Or install specific groups:

```bash
# Install just Active Inference (JAX + PyMDP)
uv sync --extra active-inference

# Install visualization libraries
uv sync --extra visualization

# Install LLM integration
uv sync --extra llm
```

## Optional Package Groups

The GNN pipeline supports the following optional package groups:

### 1. **active-inference** - High-Performance Computing

- **Packages**: `jax[cpu]`, `jaxlib`, `optax`, `flax`
- **Use case**: Fast numerical computing, automatic differentiation, JIT compilation
- **Size**: ~500MB
- **Installation**:

  ```bash
  uv sync --extra active-inference
  ```

### 2. **pymdp** - Active Inference Framework

- **Packages**: `inferactively-pymdp`
- **Use case**: Active Inference agents, POMDP modeling, free energy principle
- **Size**: ~50MB
- **Note**: Already included in core dependencies (`pyproject.toml`)

- **PyMDP Example**:

  ```python
  import pymdp
  from pymdp import utils
  from pymdp.agent import Agent

  num_obs = [3, 5]
  num_states = [3, 2, 2]
  num_controls = [3, 1, 1]
  
  A_matrix = utils.random_A_matrix(num_obs, num_states)
  B_matrix = utils.random_B_matrix(num_states, num_controls)
  C_vector = utils.obj_array_uniform(num_obs)
  
  my_agent = Agent(A=A_matrix, B=B_matrix, C=C_vector)
  observation = [1, 4]
  qs = my_agent.infer_states(observation)
  ```

### 3. **visualization** - Data Visualization

- **Packages**: `plotly`, `seaborn`, `bokeh`, `h5py`
- **Use case**: Interactive plots, statistical graphics, dashboards
- **Size**: ~100MB
- **Installation**:

  ```bash
  uv sync --extra visualization
  ```

### 4. **audio** - Audio Processing & Sonification

- **Packages**: `librosa`, `soundfile`, `pedalboard`, `pydub`, `pyaudio`
- **Use case**: Audio analysis, sonification of model dynamics
- **Size**: ~150MB
- **Installation**:

  ```bash
  uv sync --extra audio
  ```

### 5. **llm** - LLM Integration

- **Packages**: `openai`, `ollama`, `python-dotenv`, `aiohttp`
- **Use case**: AI-enhanced analysis, model interpretation
- **Size**: ~50MB
- **Installation**:

  ```bash
  uv sync --extra llm
  ```

### 6. **ml-ai** - Machine Learning

- **Packages**: `torch`, `torchvision`, `torchaudio`, `transformers`, `scipy`, `scikit-learn`
- **Use case**: Deep learning, neural networks, model training
- **Size**: ~2GB
- **Installation**:

  ```bash
  uv sync --extra ml-ai
  ```

## Installation Methods

**Note**: The recommended installation method uses UV's built-in extras system via `pyproject.toml`.

### Method 1: Using UV Extras (Recommended)

```bash
# List available groups (shown in pyproject.toml [project.optional-dependencies])
# Groups: dev, api, active-inference, probabilistic-programming, ml-ai, llm,
#         visualization, audio, gui, graphs, research, scaling, database, all

# Install all optional packages
uv sync --all-extras

# Install specific groups
uv sync --extra active-inference --extra visualization --extra llm
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

### Core Dependencies

- Python: 3.11+
- numpy: 2.4.2
- scipy: 1.16.2
- matplotlib: 3.10.3
- pandas: 2.3.0
- networkx: 3.5
- pytest: 8.4.2

### Optional Dependencies

- JAX: 0.7.2
- Optax: 0.2.6
- Flax: 0.12.0
- PyMDP: 0.0.7.1 (inferactively-pymdp)
- Plotly: 6.3.1
- Altair: 5.5.0
- Seaborn: 0.13.2

## References

- **UV Documentation**: <https://docs.astral.sh/uv/>
- **PyMDP Repository**: <https://github.com/infer-actively/pymdp>
- **JAX Documentation**: <https://jax.readthedocs.io/>
- **GNN Pipeline Documentation**: See `README.md` and `ARCHITECTURE.md`

---

**Last Updated**: 2026-03-15
**Pipeline Version**: 1.3.0
**Status**: ✅ Production Ready (Linux & macOS)
**Latest Validation**: 100% Success (25/25 steps)
