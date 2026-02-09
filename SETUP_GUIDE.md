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
python3 src/pipeline/install_optional_packages.py --all --verbose
```

Or install specific groups:

```bash
# Install just JAX and PyMDP for Active Inference
python3 src/pipeline/install_optional_packages.py --groups jax,pymdp --verbose

# Install visualization libraries
python3 src/pipeline/install_optional_packages.py --groups visualization --verbose
```

## Optional Package Groups

The GNN pipeline supports the following optional package groups:

### 1. **jax** - High-Performance Computing

- **Packages**: `jax[cpu]`, `jaxlib`, `optax`, `flax`
- **Use case**: Fast numerical computing, automatic differentiation, JIT compilation
- **Size**: ~500MB
- **Installation**:

  ```bash
  python3 src/pipeline/install_optional_packages.py --groups jax
  ```

### 2. **pymdp** - Active Inference Framework

- **Packages**: `inferactively-pymdp`
- **Use case**: Active Inference agents, POMDP modeling, free energy principle
- **Size**: ~50MB
- **Installation**:

  ```bash
  python3 src/pipeline/install_optional_packages.py --groups pymdp
  ```

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

- **Packages**: `plotly`, `altair`, `seaborn`
- **Use case**: Interactive plots, statistical graphics, dashboards
- **Size**: ~100MB
- **Installation**:

  ```bash
  python3 src/pipeline/install_optional_packages.py --groups visualization
  ```

### 4. **audio** - Audio Processing & Sonification

- **Packages**: `librosa`, `soundfile`, `pedalboard`
- **Use case**: Audio analysis, sonification of model dynamics
- **Size**: ~150MB
- **Installation**:

  ```bash
  python3 src/pipeline/install_optional_packages.py --groups audio
  ```

### 5. **llm** - LLM Integration

- **Packages**: `openai`, `anthropic`
- **Use case**: AI-enhanced analysis, model interpretation
- **Size**: ~50MB
- **Installation**:

  ```bash
  python3 src/pipeline/install_optional_packages.py --groups llm
  ```

### 6. **ml** - Machine Learning

- **Packages**: `torch`, `torchvision`, `transformers`
- **Use case**: Deep learning, neural networks, model training
- **Size**: ~2GB
- **Installation**:

  ```bash
  python3 src/pipeline/install_optional_packages.py --groups ml
  ```

## Installation Methods

**Note**: The optional package installation script is located in `src/pipeline/` for better organization with other pipeline utilities.

### Method 1: Using Standalone Script (Recommended)

```bash
# List available groups
python3 src/pipeline/install_optional_packages.py --list

# Install all optional packages
python3 src/pipeline/install_optional_packages.py --all

# Install specific groups
python3 src/pipeline/install_optional_packages.py --groups jax,pymdp,visualization
```

### Method 2: Using Setup Module Directly

```python
import sys
sys.path.insert(0, 'src')

from setup.setup import install_optional_package_group

# Install a specific group
install_optional_package_group('pymdp', verbose=True)
install_optional_package_group('jax', verbose=True)
```

### Method 3: Using UV Directly

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Install packages using UV
uv pip install inferactively-pymdp
uv pip install jax[cpu] optax flax
uv pip install plotly altair seaborn
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

- **Full 25-step pipeline**: ~40 seconds
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

- Python: 3.11.2
- numpy: 1.26.4
- scipy: 1.16.2
- matplotlib: 3.10.7
- pandas: 2.3.3
- networkx: 3.5
- pytest: 7.4.4

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

**Last Updated**: February 9, 2026
**Pipeline Version**: 1.1.0
**Status**: ✅ Production Ready (Linux & macOS)
**Latest Validation**: 100% Success (25/25 steps)
