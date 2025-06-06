# GNN Project Setup Guide

This document provides comprehensive instructions for setting up the GNN Processing Pipeline environment, including installation steps, environment variables, and detailed information about dependencies.

> **Note**: For comprehensive documentation on GNN itself, please refer to the [GNN Documentation](gnn/about_gnn.md) in the `doc/gnn/` directory.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Run the setup script
cd src
python3 main.py --only-steps 2_setup --dev
```

## System Requirements

- **Python**: 3.9 or newer
- **Operating System**: Linux (primary support)
- **Disk Space**: At least 2GB free for dependencies
- **System Packages**: 
  - `build-essential`
  - `python3-dev`
  - `graphviz` (for visualization)

You can install the required system packages on Ubuntu/Debian with:

```bash
sudo apt update
sudo apt install build-essential python3-dev graphviz
```

## Detailed Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation
```

### 2. Set Up the Environment

The setup can be done in two ways:

#### Option A: Using main.py (Recommended)

This method runs only the setup step through the main pipeline:

```bash
cd src
python3 main.py --only-steps 2_setup
```

#### Option B: Running the Setup Step Directly

```bash
cd src
python3 2_setup.py
```

### 3. Advanced Setup Options

#### Development Dependencies

To also install development dependencies (testing, code quality, documentation tools):

```bash
python3 main.py --only-steps 2_setup --dev
```

#### Recreating the Virtual Environment

If you need to recreate the virtual environment from scratch:

```bash
python3 main.py --only-steps 2_setup --recreate-venv
```

#### Verbose Setup

For detailed logging during setup:

```bash
python3 main.py --only-steps 2_setup --verbose
```

## Environment Variables

The GNN project uses the following environment variables:

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `OPENAI_API_KEY` | API key for OpenAI services (for LLM step) | For `11_llm.py` only | `sk-abcd1234...` |
| `GNN_CACHE_DIR` | Directory for caching intermediate results | No | `/path/to/cache` |
| `PYTHONPATH` | Ensures Python can find project modules | Auto-set by scripts | `src:src/.venv/lib/python3.9/site-packages` |

You can set these variables in a `.env` file in the project root, or set them directly in your shell:

```bash
export OPENAI_API_KEY="your-key-here"
export GNN_CACHE_DIR="/path/to/cache"
```

## Dependencies Explained

### Core Dependencies

- **numpy, scipy**: Scientific computing and numerical operations
- **matplotlib, networkx, graphviz**: Visualization tools
- **pandas**: Data manipulation and analysis
- **pytest**: Testing framework

### Active Inference Ecosystem

- **inferactively-pymdp**: Python implementation of active inference for Markov Decision Processes

### High-Performance Computing

- **jax, jaxlib**: High-performance numerical computing with hardware acceleration support
  - JAX accelerates tensor operations and enables automatic differentiation
  - Required for the DisCoPy matrix backend

### Visualization and Diagramming

- **discopy[matrix]**: Category theory toolkit with diagram creation
  - The `[matrix]` extra installs JAX support for tensor operations
- **altair**: Declarative statistical visualization library

### Optional Dependencies

The following dependencies are included in `requirements-dev.txt`:

- **Testing**: pytest-cov, pytest-mock
- **Code Quality**: flake8, mypy, black, isort
- **Documentation**: sphinx, sphinx-rtd-theme
- **Debugging**: ipython, ipdb
- **Performance Profiling**: py-spy

## Common Issues and Troubleshooting

### JAX Installation Issues

JAX can sometimes have compatibility issues. If you encounter problems:

1. Ensure you have the latest pip: `python -m pip install --upgrade pip`
2. Install JAX and JAXlib explicitly first: `python -m pip install --upgrade jax jaxlib`
3. Then proceed with the rest of the setup

### DisCoPy Matrix Backend Problems

If you see errors related to DisCoPy's matrix functionality:

1. Uninstall DisCoPy: `python -m pip uninstall -y discopy`
2. Reinstall with matrix support: `python -m pip install "discopy[matrix]>=1.0.0"`

### PyMDP Import Errors

If you encounter issues importing PyMDP:

1. Check that `inferactively-pymdp` is installed: `python -m pip list | grep pymdp`
2. Try reinstalling: `python -m pip install --force-reinstall inferactively-pymdp`

## Dependency Version Compatibility

The GNN project has been tested with the following key dependency versions:

| Dependency | Tested Versions | Notes |
|------------|----------------|-------|
| Python | 3.9, 3.10, 3.11 | 3.10 recommended |
| NumPy | 1.24.x - 1.26.x | Required by JAX |
| JAX | 0.4.20+ | Required for DisCoPy matrix backend |
| DisCoPy | 1.0.0+ | With matrix extras |
| inferactively-pymdp | 0.2.0+ | Required for PyMDP steps |

## Verifying Your Installation

After setup, verify your installation with:

```bash
cd src
python3 3_tests.py
```

This will run the test suite to ensure everything is working correctly.

## Updating Dependencies

To update dependencies in an existing installation:

```bash
cd src
python3 main.py --only-steps 2_setup --recreate-venv
```

## Using a Custom Python Executable

If you need to use a specific Python executable:

```bash
/path/to/your/python main.py --only-steps 2_setup
```

## Docker Setup (Experimental)

For containerized setup, a Dockerfile is available:

```bash
docker build -t gnn-project .
docker run -it gnn-project
```

## Need Help?

If you encounter issues during setup:

1. Check the logs in `output/logs/` directory
2. Refer to this guide's troubleshooting section
3. Open an issue on GitHub with the error details 