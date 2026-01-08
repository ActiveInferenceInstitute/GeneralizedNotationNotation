# GNN Project Setup Guide

> **📋 Document Metadata**
> **Type**: Setup Guide | **Audience**: All Users | **Complexity**: Beginner
> **Cross-References**: [Quickstart Guide](quickstart.md) | [Main Documentation](../README.md) | [Environment Template](../.env) | [Troubleshooting](troubleshooting/README.md)

This document provides comprehensive instructions for setting up the GNN (Generalized Notation Notation) Processing Pipeline environment, including installation steps, environment variables, and detailed information about dependencies.

> **🎯 Quick Start**: For immediate setup, run:
> ```bash
> cd src && python3 main.py --only-steps 1 --dev
> ```

> **📖 Complete Guide**: For comprehensive documentation on GNN itself, please refer to the [GNN Documentation](gnn/about_gnn.md) in the `doc/gnn/` directory.

## 🎯 Setup Overview

The GNN project provides multiple installation paths depending on your needs:

- **🚀 Quick Setup** (5 minutes): Basic functionality for trying GNN
- **🔧 Standard Setup** (15 minutes): Full pipeline with core dependencies
- **⚡ Complete Setup** (30+ minutes): All features including optional heavy packages
- **🛠️ Development Setup** (45+ minutes): Full development environment with all tools

Choose based on your use case:
- **Researchers**: Quick or Standard setup
- **Developers**: Standard or Complete setup
- **Production**: Complete setup with optimization

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation

# Run the setup script
cd src
python3 main.py --only-steps 1 --dev
```

## System Requirements

- **Python**: 3.11 or newer (up to 3.13)
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
python3 main.py --only-steps 1
```

#### Option B: Running the Setup Step Directly

```bash
cd src
python3 1_setup.py
```

### 3. Advanced Setup Options

#### Development Dependencies

To also install development dependencies (testing, code quality, documentation tools):

```bash
python3 main.py --only-steps 1 --dev
```

#### Recreating the Virtual Environment

If you need to recreate the virtual environment from scratch:

```bash
python3 main.py --only-steps 1 --recreate-venv
```

#### Verbose Setup

For detailed logging during setup:

```bash
python3 main.py --only-steps 1 --verbose
```

## Environment Variables

The GNN project uses the following environment variables:

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `OPENAI_API_KEY` | API key for OpenAI services (for LLM step) | For `11_llm.py` only | `sk-abcd1234...` |
| `GNN_CACHE_DIR` | Directory for caching intermediate results | No | `/path/to/cache` |
| `PYTHONPATH` | Ensures Python can find project modules | Auto-set by scripts | `src:src/.venv/lib/python3.11/site-packages` |

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

## Framework Dependencies

The GNN pipeline supports multiple Active Inference simulation frameworks. Each framework has different installation requirements and provides unique capabilities.

### Framework Overview

| Framework | Status | Install Method | Primary Use Case | GPU Support |
|-----------|--------|----------------|------------------|-------------|
| **DisCoPy** | ✅ Built-in | Included | Categorical diagrams | No |
| **ActiveInference.jl** | ✅ Auto | Julia | Complete Active Inference | No |
| **PyMDP** | ⚠️ Optional | pip | Python Active Inference | No |
| **JAX** | ⚠️ Optional | pip | GPU-accelerated inference | Yes |
| **RxInfer.jl** | ⚠️ Optional | Julia | Bayesian message passing | No |

### Quick Install (Recommended)

For most users, install the "lite" framework preset:

```bash
# Install lite preset (PyMDP + JAX)
python src/1_setup.py --install_optional --optional_groups "pymdp,jax"
```

For complete framework support:

```bash
# Install all optional frameworks
python src/1_setup.py --install_optional --optional_groups "all"
```

### Individual Framework Installation

#### DisCoPy (✅ Included)

DisCoPy is included by default and requires no additional installation.

**Capabilities**:
- Categorical diagram generation
- String diagram composition
- Functor visualization

**Verification**:
```bash
python3 -c "import discopy; print('DisCoPy OK')"
```

#### ActiveInference.jl (✅ Auto-Install)

ActiveInference.jl is automatically installed when first needed via Julia's package manager.

**Requirements**:
- Julia 1.6+ installed on system
- Internet connection for first run

**Capabilities**:
- Full Active Inference agent implementation
- Hierarchical temporal models
- Comprehensive belief updating

**Installation**:
```bash
# Julia installs automatically on first execution
# Or install manually:
julia -e 'using Pkg; Pkg.add("ActiveInference")'
```

**Verification**:
```bash
julia -e 'using ActiveInference; println("ActiveInference.jl OK")'
```

#### PyMDP (⚠️ Optional - Recommended)

PyMDP provides Python-based Active Inference for POMDPs.

**Installation**:
```bash
# Correct package name is inferactively-pymdp
uv pip install inferactively-pymdp

# Or using the setup module
python src/1_setup.py --install_optional --optional_groups pymdp

# Or from source for latest features
uv pip install git+https://github.com/infer-actively/pymdp.git
```

**Important**: The correct package name is `inferactively-pymdp`, not `pymdp`. The `pymdp` package on PyPI contains MDP/MDPSolver but not the Active Inference Agent class.

**Capabilities**:
- POMDP agent implementation
- Variational message passing
- Policy inference and learning

**Common Issues**:
- **Wrong package installed**: If you have `pymdp` installed, uninstall it and install `inferactively-pymdp`
- **Import errors**: Verify correct package: `python -c "from pymdp import Agent; print('PyMDP OK')"`
- **Package detection**: The execute module automatically detects wrong package variants

**Verification**:
```bash
python3 -c "from pymdp import Agent; print('PyMDP OK')"
```

#### JAX (⚠️ Optional - Recommended)

JAX enables high-performance numerical computing with GPU acceleration.

**Installation**:
```bash
# CPU-only version (most users)
uv pip install jax[cpu] flax optax

# GPU version (CUDA 12.x)
uv pip install jax[cuda12_pip] flax optax

# GPU version (CUDA 11.x)
uv pip install jax[cuda11_pip] flax optax
```

**Capabilities**:
- GPU-accelerated tensor operations
- Just-in-time (JIT) compilation
- Automatic differentiation
- Vectorized computations

**System Requirements**:
- CPU version: Any modern CPU
- GPU version: NVIDIA GPU with CUDA support

**Verification**:
```bash
python3 -c "import jax; import flax.linen; print('JAX + Flax OK')"
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

#### RxInfer.jl (⚠️ Optional)

RxInfer.jl provides reactive Bayesian inference via message passing.

**Requirements**:
- Julia 1.6+ installed
- RxInfer.jl Julia package

**Installation**:
```bash
# Install via Julia package manager
julia -e 'using Pkg; Pkg.add("RxInfer")'
julia -e 'using Pkg; Pkg.add("ReactiveMP")'
julia -e 'using Pkg; Pkg.add("GraphPPL")'
```

**Capabilities**:
- Reactive probabilistic programming
- Efficient message-passing inference
- Factor graph models
- Streaming inference

**Verification**:
```bash
julia -e 'using RxInfer; println("RxInfer.jl OK")'
```

### Framework Selection Strategies

#### Lite Preset (Recommended for Most Users)
```bash
# Install PyMDP + JAX only
python src/1_setup.py --install_optional --optional_groups "pymdp,jax"
```

**Included**:
- DisCoPy (built-in)
- ActiveInference.jl (auto-install)
- PyMDP (manual install)
- JAX (manual install)

**Best for**: Python developers, GPU users, fast prototyping

#### Full Preset (Complete Functionality)
```bash
# Install all frameworks
python src/1_setup.py --install_optional --optional_groups "all"
```

**Included**: All 5 frameworks

**Best for**: Research, comprehensive benchmarking, production use

#### Minimal Preset (Quick Start)
```bash
# Use built-in frameworks only (no optional install)
python src/1_setup.py
```

**Included**:
- DisCoPy (built-in)
- ActiveInference.jl (auto-install on first use)

**Best for**: Quick testing, minimal dependencies

### Framework Execution

#### Running Specific Frameworks
```bash
# Execute specific frameworks only
python src/12_execute.py --frameworks "pymdp,jax"

# Execute lite preset
python src/12_execute.py --frameworks "lite"

# Execute all available frameworks
python src/12_execute.py --frameworks "all"
```

#### Framework Availability Check
```bash
# Check which frameworks are available
python src/12_execute.py --frameworks "all" --dry-run
```

### Troubleshooting Framework Issues

#### PyMDP Issues

**Symptom**: `ModuleNotFoundError: No module named 'pymdp.agent'`

**Solution**:
```bash
uv pip install inferactively-pymdp  # Install correct package
python3 -c "from pymdp import Agent; print('✅ PyMDP OK')"  # Verify using modern API
```

#### JAX Issues

**Symptom**: `No module named 'flax'` or JAX import errors

**Solution**:
```bash
# Reinstall with all components
pip uninstall jax jaxlib flax -y
pip install jax[cpu] flax optax

# Verify
python3 -c "import jax; print(jax.devices())"
```

#### RxInfer.jl Issues

**Symptom**: `Half-edge has been found` errors

**Solution**: This is a known issue with older generated code templates. Regenerate code with:
```bash
# Regenerate RxInfer code with latest templates
python src/11_render.py --target-dir input/gnn_files --force-regenerate
```

#### Julia Framework Issues

**Symptom**: Julia packages not found

**Solution**:
```bash
# Update Julia packages
julia -e 'using Pkg; Pkg.update()'
julia -e 'using Pkg; Pkg.gc()'  # Clean package cache

# Reinstall if needed
julia -e 'using Pkg; Pkg.add("ActiveInference"); Pkg.add("RxInfer")'
```

### Framework Performance Comparison

| Framework | Execution Speed | Memory Usage | Setup Complexity | GPU Support |
|-----------|----------------|--------------|------------------|-------------|
| DisCoPy | Fast | Low | Easy | No |
| ActiveInference.jl | Medium | Medium | Medium | No |
| PyMDP | Medium | Low | Easy | No |
| JAX | Very Fast | Medium | Easy | Yes |
| RxInfer.jl | Fast | Low | Medium | No |

### Best Practices

1. **Start with Lite Preset**: Install PyMDP + JAX for most use cases
2. **Test Incrementally**: Install one framework at a time if issues occur
3. **Use Virtual Environments**: Isolate framework dependencies
4. **Check Versions**: Ensure compatible versions of Python/Julia
5. **Monitor Resources**: Some frameworks require significant memory

### Framework Documentation

- **DisCoPy**: [https://discopy.org/](https://discopy.org/)
- **ActiveInference.jl**: [https://github.com/biaslab/ActiveInference.jl](https://github.com/biaslab/ActiveInference.jl)
- **PyMDP**: [https://github.com/infer-actively/pymdp](https://github.com/infer-actively/pymdp)
- **JAX**: [https://jax.readthedocs.io/](https://jax.readthedocs.io/)
- **RxInfer.jl**: [https://rxinfer.ml/](https://rxinfer.ml/)

## 🛠️ Troubleshooting Dependency Conflicts

While the `main.py --only-steps 1` approach typically handles dependencies, certain environments (especially on Apple Silicon or specialized Linux distros) can experience conflicts.

### **JAX vs. PyMDP Version Mismatch**
Some versions of PyMDP depend on specific NumPy ranges that can conflict with the latest JAX requirements.
- **Symptom**: `ImportError: numpy.core.multiarray failed to import`
- **Solution**:
    ```bash
    uv pip install --upgrade "numpy>=1.24,<1.27" "jax[cpu]" "inferactively-pymdp"
    ```

### **Julia Environment Issues**
- **Symptom**: `julia: command not found` but Julia is definitely installed.
- **Solution**: Ensure your PATH includes the Julia bin directory. On macOS:
    ```bash
    export PATH="$PATH:/Applications/Julia-1.10.app/Contents/Resources/julia/bin"
    ```

### **Category Theory Backend Conflicts**
- **Symptom**: `TypeError: 'module' object is not callable` when using DisCoPy with JAX.
- **Solution**: Ensure you install DisCoPy with the matrix extra:
    ```bash
    uv pip install "discopy[matrix]"
    ```


## Common Issues and Troubleshooting

### JAX Installation Issues

JAX can sometimes have compatibility issues. If you encounter problems:

1. Ensure you have the latest uv: `uv self update`
2. Install JAX and JAXlib explicitly first: `uv pip install --upgrade jax jaxlib`
3. Then proceed with the rest of the setup

### DisCoPy Matrix Backend Problems

If you see errors related to DisCoPy's matrix functionality:

1. Uninstall DisCoPy: `uv pip uninstall -y discopy`
2. Reinstall with matrix support: `uv pip install "discopy[matrix]>=1.0.0"`

### PyMDP Import Errors

If you encounter issues importing PyMDP:

1. Check that PyMDP is installed: `uv pip list | grep pymdp`
2. Try reinstalling: `uv pip install --force-reinstall inferactively-pymdp`

## Dependency Version Compatibility

The GNN project has been tested with the following key dependency versions:

| Dependency | Tested Versions | Notes |
|------------|----------------|-------|
| Python | 3.11, 3.12, 3.13 | 3.11 recommended |
| NumPy | 1.24.x - 1.26.x | Required by JAX |
| JAX | 0.4.20+ | Required for DisCoPy matrix backend |
| DisCoPy | 1.0.0+ | With matrix extras |
| inferactively-pymdp | 0.2.0+ | Required for PyMDP steps |

## Verifying Your Installation

After setup, verify your installation with:

```bash
cd src
python3 2_tests.py
```

This will run the test suite to ensure everything is working correctly.

## Updating Dependencies

To update dependencies in an existing installation:

```bash
cd src
python3 main.py --only-steps 1 --recreate-venv
```

## Using a Custom Python Executable

If you need to use a specific Python executable:

```bash
/path/to/your/python main.py --only-steps 1
```

## Docker Setup (Experimental)

For containerized setup, a Dockerfile is available:

```bash
docker build -t gnn-project .
docker run -it gnn-project
```

## Security Considerations

### **Environment Security**
- Store API keys securely (use environment variables or secret management)
- Validate GNN file inputs in production environments
- Review [Security Guide](security/README.md) for comprehensive security practices

### **LLM Integration Security**
- **API Key Protection**: Never commit API keys to version control
- **Prompt Injection Prevention**: GNN includes built-in prompt sanitization
- **Output Validation**: All LLM outputs are validated before execution

## Performance Optimization

### **System Requirements for Optimal Performance**
```yaml
recommended_specs:
  memory: ">= 8GB RAM"
  storage: ">= 5GB SSD space"
  cpu: "Multi-core processor (4+ cores recommended)"
  python: "3.11, 3.12, or 3.13 (optimal performance)"
```

### **Large Model Processing**
For processing large GNN models (>50MB):
```bash
# Increase memory limits and enable caching
export GNN_CACHE_DIR="/path/to/fast/storage"
export GNN_MAX_MEMORY="4GB"
python3 main.py --memory-efficient
```

## Integration with Development Tools

### **VS Code Integration**
For the best development experience:
```bash
# Install recommended VS Code extensions
code --install-extension ms-python.python
code --install-extension davidanson.vscode-markdownlint
code --install-extension ms-toolsai.jupyter
```

### **Jupyter Notebook Support**
GNN includes Jupyter notebook integration:
```bash
# Install Jupyter support
uv pip install jupyter ipykernel
python -m ipykernel install --user --name gnn

# Launch Jupyter with GNN kernel
jupyter notebook
```

## Continuous Integration Setup

### **GitHub Actions**
For automated testing and validation:
```yaml
# .github/workflows/gnn-test.yml
name: GNN Pipeline Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup GNN
      run: |
        cd src
        python3 main.py --only-steps 1
    - name: Run Tests
      run: |
        cd src
        python3 2_tests.py
```

## Version Management

### **Multiple GNN Versions**
To work with multiple GNN versions:
```bash
# Use virtual environments for version isolation
python3 -m venv gnn-v1.1.0
source gnn-v1.1.0/bin/activate
cd src && python3 main.py --only-steps 1

# Create another environment for development
python3 -m venv gnn-dev
source gnn-dev/bin/activate
cd src && python3 main.py --only-steps 1 --dev
```

### **Upgrade Process**
When upgrading GNN versions:
1. **Backup**: Save current models and configurations
2. **Test**: Validate with existing models using new version
3. **Migrate**: Follow version-specific upgrade guides in [Changelog](../CHANGELOG.md)
4. **Verify**: Run comprehensive tests on upgraded installation

## Need Help?

### **Common Setup Issues**
- **Python Version**: Ensure Python 3.11+ is installed and active
- **Virtual Environment**: Always use virtual environments for isolation
- **Dependencies**: Check [Common Errors](troubleshooting/common_errors.md) for dependency issues
- **Permissions**: Ensure write access to project directory

### **Getting Support**
If you encounter issues during setup:

1. **Check Documentation**: [Troubleshooting Guide](troubleshooting/README.md)
2. **Review Logs**: Check `output/logs/` directory for detailed error information
3. **Search Issues**: Look through [GitHub Issues](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/issues)
4. **Create Issue**: Open a new issue with setup error details and system information

### **Community Resources**
- **[GitHub Discussions](https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation/discussions)**: Community Q&A
- **[Active Inference Institute](https://activeinference.org)**: Research community and resources
- **[Documentation](README.md)**: Comprehensive project documentation

---

**Setup Guide Version**: 2.0  
**Compatible GNN Versions**: v1.1.0+  
