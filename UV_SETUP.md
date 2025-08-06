# UV-Based Setup for GNN Project

This document describes the modern UV-based setup for the GeneralizedNotationNotation (GNN) project, which uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python dependency management.

## Overview

The GNN project has been migrated from traditional `pip`/`venv` to UV-based dependency management, providing:

- **Faster dependency resolution** - UV is 10-100x faster than pip
- **Reproducible builds** - Lock file ensures consistent environments
- **Modern Python packaging** - Uses `pyproject.toml` instead of `requirements.txt`
- **Built-in virtual environment management** - No need for separate venv tools
- **Optional dependency groups** - Install only what you need

## Prerequisites

### Install UV

First, install UV on your system:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Manual installation:**
Visit [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/) for detailed instructions.

### Verify Installation

```bash
uv --version
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ActiveInferenceInstitute/GeneralizedNotationNotation.git
cd GeneralizedNotationNotation
```

### 2. Run UV Setup

```bash
# Basic setup with core dependencies
python3 src/1_setup.py

# Setup with development dependencies
python3 src/1_setup.py --dev

# Setup with optional dependency groups
python3 src/1_setup.py --install-optional ml-ai,llm,visualization
```

### 3. Run the Pipeline

```bash
# Run with UV
uv run python src/main.py --help

# Run tests
uv run pytest src/tests/

# Run specific pipeline steps
uv run python src/main.py --only-steps 1,2,3
```

## Project Structure

The UV-based setup creates the following structure:

```
GeneralizedNotationNotation/
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Lock file for reproducible builds
├── .venv/                  # UV-managed virtual environment
├── src/                    # Source code
│   ├── 1_setup.py         # UV-based setup script
│   ├── setup/             # Setup utilities
│   │   ├── __init__.py    # UV-aware setup module
│   │   ├── utils.py       # UV utilities
│   │   ├── setup.py       # UV setup functions
│   │   └── mcp.py         # UV MCP integration
│   └── ...
├── input/                  # Input files
├── output/                 # Pipeline outputs
└── ...
```

## Dependency Management

### Core Dependencies

The project uses `pyproject.toml` for dependency management:

```toml
[project]
dependencies = [
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "networkx>=2.6.0",
    "pandas>=1.3.0",
    "pyyaml>=6.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    # ... more core dependencies
]
```

### Optional Dependency Groups

Install specific feature sets:

```bash
# Machine Learning & AI
uv add --extras ml-ai

# LLM Integration
uv add --extras llm

# Advanced Visualization
uv add --extras visualization

# Audio Processing
uv add --extras audio

# Graph Visualization
uv add --extras graphs

# Research Tools
uv add --extras research

# Development Tools
uv add --extras dev

# All optional dependencies
uv add --extras all
```

### Adding New Dependencies

```bash
# Add a new dependency
uv add package-name

# Add with specific version
uv add "package-name>=1.0.0"

# Add development dependency
uv add --dev package-name

# Add with extras
uv add "package-name[extra]"
```

## UV Commands

### Environment Management

```bash
# Initialize UV project
uv init

# Sync dependencies (install from pyproject.toml)
uv sync

# Sync with specific extras
uv sync --extra dev --extra ml-ai

# Sync without development dependencies
uv sync --no-dev

# Lock dependencies (generate uv.lock)
uv lock

# Update lock file
uv lock --upgrade
```

### Package Management

```bash
# List installed packages
uv pip list

# Show package info
uv pip show package-name

# Uninstall package
uv remove package-name

# Update packages
uv sync --upgrade
```

### Running Commands

```bash
# Run Python script in UV environment
uv run python script.py

# Run with arguments
uv run python src/main.py --help

# Run tests
uv run pytest

# Run with specific Python version
uv run --python 3.12 python script.py
```

## Setup Scripts

### Main Setup Script (`src/1_setup.py`)

The main setup script provides comprehensive UV-based environment setup:

```bash
# Basic setup
python3 src/1_setup.py

# Verbose setup with detailed logging
python3 src/1_setup.py --verbose

# Setup with development dependencies
python3 src/1_setup.py --dev

# Setup with optional dependency groups
python3 src/1_setup.py --install-optional ml-ai,llm,visualization

# Recreate environment from scratch
python3 src/1_setup.py --recreate-venv
```

### Setup Module (`src/setup/`)

The setup module provides programmatic access to UV functionality:

```python
from src.setup import setup_uv_environment, validate_uv_setup

# Setup environment
setup_uv_environment(verbose=True, dev=True, extras=['ml-ai'])

# Validate setup
status = validate_uv_setup()
print(f"Setup valid: {status['overall_status']}")
```

## MCP Integration

The setup module includes MCP (Model Context Protocol) integration for UV operations:

```python
from src.setup.mcp import register_tools

# Register UV tools with MCP
register_tools(mcp_instance)

# Available tools:
# - check_uv_project_status
# - get_uv_environment_info
# - setup_uv_project_structure
# - install_uv_dependency
# - sync_uv_dependencies
```

## Migration from Requirements.txt

The project has been migrated from `requirements.txt` to `pyproject.toml`:

### Old Structure (requirements.txt)
```
# Core dependencies
numpy>=1.21.0
matplotlib>=3.5.0
# ... more dependencies

# Optional dependencies (commented out)
# torch>=1.12.0
# transformers>=4.20.0
```

### New Structure (pyproject.toml)
```toml
[project]
dependencies = [
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    # ... core dependencies
]

[project.optional-dependencies]
ml-ai = [
    "torch>=1.12.0",
    "transformers>=4.20.0",
]
llm = [
    "openai>=1.0.0",
    "anthropic>=0.5.0",
]
# ... more optional groups
```

## Troubleshooting

### Common Issues

**UV not found:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart shell or add to PATH
source ~/.bashrc
```

**Lock file conflicts:**
```bash
# Regenerate lock file
uv lock --upgrade

# Or remove and recreate
rm uv.lock
uv sync
```

**Environment issues:**
```bash
# Recreate environment
rm -rf .venv
uv sync
```

**Package installation failures:**
```bash
# Clear UV cache
uv cache clean

# Try with verbose output
uv sync --verbose
```

### Performance Optimization

**Fast sync:**
```bash
# Use UV's fast resolver
uv sync --fast
```

**Parallel downloads:**
```bash
# Increase concurrency
uv sync --concurrent 8
```

**Offline mode:**
```bash
# Use cached packages only
uv sync --offline
```

## Development Workflow

### Adding New Dependencies

1. **Add to pyproject.toml:**
   ```toml
   [project]
   dependencies = [
       "new-package>=1.0.0",
   ]
   ```

2. **Update lock file:**
   ```bash
   uv lock
   ```

3. **Install:**
   ```bash
   uv sync
   ```

### Testing Dependencies

```bash
# Test in isolated environment
uv run --python 3.12 pytest

# Test with specific extras
uv run --with ml-ai pytest tests/test_ml.py
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Test with UV
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install UV
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.cargo/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: uv sync --extra dev
      
      - name: Run tests
        run: uv run pytest
```

## Benefits of UV

### Performance
- **10-100x faster** than pip for dependency resolution
- **Parallel downloads** and installations
- **Smart caching** for repeated operations

### Reliability
- **Lock file** ensures reproducible builds
- **Conflict resolution** handles complex dependency trees
- **Offline mode** for air-gapped environments

### Modern Standards
- **PEP 517/518** compliant
- **pyproject.toml** support
- **Built-in virtual environments**
- **Optional dependency groups**

### Developer Experience
- **Simple commands** (`uv sync`, `uv run`)
- **Fast feedback** with real-time progress
- **Clear error messages** and suggestions
- **IDE integration** support

## Comparison with Traditional Tools

| Feature | UV | pip + venv | pipenv | poetry |
|---------|----|------------|--------|--------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Lock files | ✅ | ❌ | ✅ | ✅ |
| pyproject.toml | ✅ | ❌ | ❌ | ✅ |
| Built-in venv | ✅ | ❌ | ✅ | ❌ |
| Optional deps | ✅ | ❌ | ✅ | ✅ |
| CI/CD friendly | ✅ | ✅ | ✅ | ✅ |

## Next Steps

1. **Install UV** if you haven't already
2. **Run the setup script** to initialize the environment
3. **Explore the project** using UV commands
4. **Add dependencies** as needed for your work
5. **Contribute** using the UV-based workflow

For more information about UV, visit [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/). 