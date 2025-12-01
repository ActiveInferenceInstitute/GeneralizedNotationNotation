# Optional Dependencies Guide

> **Environment Note**: Manage optional dependencies with `uv pip install` for consistency. Core functionality works without optional packages.

## Overview

The GNN pipeline supports optional dependencies that enhance functionality without being required for core operations. This guide explains how to detect, install, and gracefully handle optional packages.

---

## Dependency Categories

### Core Dependencies (Always Required)

These packages are required and installed automatically:

| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Numerical computing | >=1.24.0 |
| scipy | Scientific computing | >=1.10.0 |
| matplotlib | Visualization | >=3.7.0 |
| pyyaml | YAML parsing | >=6.0 |
| pytest | Testing | >=7.0.0 |
| psutil | System monitoring | >=5.9.0 |

### Optional Python Packages

| Package | Purpose | Install Command | Used By |
|---------|---------|-----------------|---------|
| pymdp | Active Inference simulations | `pip install pymdp` | Step 12 Execute |
| jax | ML/numerical computing | `pip install jax jaxlib` | Step 11 Render |
| optax | JAX optimizers | `pip install optax` | Step 11 Render |
| flax | Neural networks (optional) | `pip install flax` | Optional for JAX |
| discopy | Categorical diagrams | `pip install discopy` | Step 11 Render |
| ollama | Local LLM | `pip install ollama` | Step 13 LLM |
| openai | OpenAI API | `pip install openai` | Step 13 LLM |
| networkx | Graph analysis | `pip install networkx` | Visualization |
| plotly | Interactive plots | `pip install plotly` | Advanced Viz |
| seaborn | Statistical plots | `pip install seaborn` | Visualization |

### Optional System Dependencies

| Dependency | Purpose | Install Command | Used By |
|------------|---------|-----------------|---------|
| Julia | Julia runtime | See julia-lang.org | RxInfer, ActiveInference.jl |
| RxInfer.jl | Julia probabilistic | `Pkg.add("RxInfer")` | Step 12 Execute |
| ActiveInference.jl | Julia Active Inference | `Pkg.add("ActiveInference")` | Step 12 Execute |
| D2 | Diagram generation | `brew install d2` | Advanced Viz |
| Ollama | Local LLM server | See ollama.ai | Step 13 LLM |

---

## Detection Patterns

### Python Package Detection

```python
def check_python_dependency(package_name: str) -> tuple[bool, str]:
    """
    Check if a Python package is available.
    
    Returns:
        (available: bool, version_or_error: str)
    """
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, f"Not installed. Install with: pip install {package_name}"

# Usage
available, info = check_python_dependency("pymdp")
if available:
    logger.info(f"PyMDP available: {info}")
else:
    logger.info(f"PyMDP: {info}")
```

### Julia Package Detection

```python
import subprocess
import shutil

def check_julia_available() -> tuple[bool, str]:
    """Check if Julia is available."""
    julia_path = shutil.which("julia")
    if not julia_path:
        return False, "Julia not found in PATH"
    
    try:
        result = subprocess.run(
            ["julia", "--version"],
            capture_output=True,
            timeout=5,
            text=True
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, result.stderr
    except Exception as e:
        return False, str(e)

def check_julia_package(package_name: str) -> tuple[bool, str]:
    """Check if a Julia package is installed."""
    available, julia_info = check_julia_available()
    if not available:
        return False, julia_info
    
    try:
        result = subprocess.run(
            ["julia", "-e", f'using {package_name}; println("{package_name} loaded")'],
            capture_output=True,
            timeout=30,
            text=True
        )
        if result.returncode == 0:
            return True, f"{package_name} available"
        return False, f"Not installed. Install with: julia -e 'using Pkg; Pkg.add(\"{package_name}\")'"
    except Exception as e:
        return False, str(e)
```

### System Command Detection

```python
def check_system_command(command: str) -> tuple[bool, str]:
    """Check if a system command is available."""
    path = shutil.which(command)
    if path:
        return True, path
    return False, f"Command '{command}' not found in PATH"

# Check D2 for diagrams
d2_available, d2_info = check_system_command("d2")

# Check Ollama
ollama_available, ollama_info = check_system_command("ollama")
```

---

## Graceful Fallback Patterns

### Import with Fallback

```python
# Pattern 1: Simple fallback
try:
    import pymdp
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    logger.info(
        "PyMDP not available - this is normal if not installed. "
        "To enable PyMDP simulations, install with: pip install pymdp. "
        "Continuing with fallback mode."
    )

# Pattern 2: Fallback implementation
try:
    from pymdp import Agent as PyMDPAgent
except ImportError:
    class PyMDPAgent:
        """Fallback stub when PyMDP unavailable."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyMDP is required for this feature. "
                "Install with: pip install pymdp"
            )
```

### Feature Flags

```python
# In module __init__.py
FEATURES = {
    "pymdp_simulation": _check_pymdp_available(),
    "jax_rendering": _check_jax_available(),
    "julia_execution": _check_julia_available(),
    "interactive_plots": _check_plotly_available(),
    "local_llm": _check_ollama_available(),
}

def get_available_features() -> Dict[str, bool]:
    """Get dictionary of available features."""
    return FEATURES.copy()

def require_feature(feature: str) -> None:
    """Raise error if feature not available."""
    if not FEATURES.get(feature):
        raise FeatureNotAvailableError(
            f"Feature '{feature}' requires optional dependencies. "
            f"See installation guide for requirements."
        )
```

### Tiered Functionality

```python
def execute_simulation(model: Dict[str, Any], framework: str) -> ExecutionResult:
    """
    Execute simulation with framework fallback.
    
    Tries frameworks in order of preference, falling back as needed.
    """
    # Try requested framework first
    if framework == "pymdp" and PYMDP_AVAILABLE:
        return _execute_pymdp(model)
    
    if framework == "jax" and JAX_AVAILABLE:
        return _execute_jax(model)
    
    if framework in ("rxinfer", "activeinference_jl") and JULIA_AVAILABLE:
        return _execute_julia(model, framework)
    
    # Fallback: Generate code but don't execute
    logger.warning(
        f"Framework '{framework}' not available. "
        f"Generating code only (no execution)."
    )
    return ExecutionResult(
        success=True,
        executed=False,
        message=f"Code generated for {framework} but not executed (dependency not available)"
    )
```

---

## Framework-Specific Handling

### JAX (No Flax Required)

The JAX renderer generates pure JAX code without Flax dependency:

```python
def check_jax_environment() -> Dict[str, Any]:
    """Check JAX environment and capabilities."""
    result = {
        "jax_available": False,
        "optax_available": False,
        "flax_available": False,  # Optional, not required
        "devices": [],
        "backend": None,
    }
    
    try:
        import jax
        result["jax_available"] = True
        result["devices"] = [str(d) for d in jax.devices()]
        result["backend"] = jax.default_backend()
    except ImportError:
        pass
    
    try:
        import optax
        result["optax_available"] = True
    except ImportError:
        pass
    
    try:
        import flax
        result["flax_available"] = True
    except ImportError:
        # Flax is optional - JAX works without it
        pass
    
    return result

# The JAX renderer generates pure functional code
# that only requires jax and optax, NOT flax
def render_jax_code(model: Dict[str, Any]) -> str:
    """
    Render pure JAX code (no Flax dependency).
    
    Generated code uses:
    - jax.numpy for array operations
    - optax for optimization (optional)
    - Pure functional patterns (no nn.Module)
    """
    ...
```

### PyMDP

```python
def check_pymdp_environment() -> Dict[str, Any]:
    """Check PyMDP environment."""
    result = {
        "available": False,
        "version": None,
        "message": "",
    }
    
    try:
        import pymdp
        result["available"] = True
        result["version"] = getattr(pymdp, "__version__", "unknown")
        result["message"] = "PyMDP ready for simulations"
    except ImportError:
        result["message"] = (
            "PyMDP not installed. Install with: pip install pymdp\n"
            "Alternatively, use other frameworks: RxInfer.jl, ActiveInference.jl, or JAX"
        )
    
    return result

# Fallback when PyMDP unavailable
class PyMDPFallback:
    """Fallback for PyMDP functionality."""
    
    @staticmethod
    def create_simulation_stub(model: Dict[str, Any], output_path: Path) -> None:
        """Create stub code that explains how to run with PyMDP."""
        stub = f'''"""
PyMDP Simulation Stub for {model.get("name", "Unknown")}

This file was generated because PyMDP is not installed.
To run actual simulations:

1. Install PyMDP:
   pip install pymdp

2. Re-run the render step:
   python src/11_render.py --target-dir input/gnn_files

3. Execute the generated script:
   python {output_path}

Model Information:
- States: {model.get("n_states", "?")}
- Observations: {model.get("n_obs", "?")}
- Actions: {model.get("n_actions", "?")}
"""

print("PyMDP not installed. See instructions above.")
'''
        output_path.write_text(stub)
```

### Julia Frameworks

```python
def setup_julia_environment(packages: List[str]) -> bool:
    """
    Setup Julia environment with required packages.
    
    Args:
        packages: List of Julia packages to install.
    
    Returns:
        True if setup successful.
    """
    julia_available, info = check_julia_available()
    if not julia_available:
        logger.error(f"Julia not available: {info}")
        return False
    
    for package in packages:
        try:
            result = subprocess.run(
                [
                    "julia", "-e",
                    f'using Pkg; Pkg.add("{package}"); using {package}; println("OK")'
                ],
                capture_output=True,
                timeout=300,  # 5 minutes for package installation
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Julia package {package} ready")
            else:
                logger.warning(f"Failed to install {package}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing {package}")
            return False
    
    return True

# RxInfer setup helper
def setup_rxinfer() -> bool:
    """Setup RxInfer.jl environment."""
    return setup_julia_environment([
        "RxInfer",
        "Distributions",
        "LinearAlgebra",
    ])

# ActiveInference.jl setup helper  
def setup_activeinference() -> bool:
    """Setup ActiveInference.jl environment."""
    return setup_julia_environment([
        "ActiveInference",
        "Distributions",
        "LinearAlgebra",
    ])
```

---

## Installation Helpers

### Install All Optional Dependencies

```bash
# Python packages
pip install pymdp jax jaxlib optax discopy ollama openai plotly seaborn networkx

# Julia packages (run in Julia REPL)
using Pkg
Pkg.add(["RxInfer", "ActiveInference", "Distributions", "LinearAlgebra", "Plots"])

# System tools (macOS)
brew install d2
brew install --cask ollama
```

### Install Specific Groups

```bash
# ML/JAX stack (no Flax needed for GNN)
pip install jax jaxlib optax

# Active Inference Python
pip install pymdp

# Visualization
pip install plotly seaborn networkx

# LLM Support
pip install ollama openai

# Categorical diagrams
pip install discopy
```

### Environment Validation Script

```python
#!/usr/bin/env python3
"""Validate optional dependencies."""

def validate_environment():
    """Check and report on optional dependencies."""
    
    checks = [
        ("PyMDP", check_python_dependency, "pymdp"),
        ("JAX", check_python_dependency, "jax"),
        ("Optax", check_python_dependency, "optax"),
        ("DisCoPy", check_python_dependency, "discopy"),
        ("Ollama (Python)", check_python_dependency, "ollama"),
        ("Julia", check_julia_available, None),
        ("RxInfer.jl", check_julia_package, "RxInfer"),
        ("ActiveInference.jl", check_julia_package, "ActiveInference"),
        ("D2", check_system_command, "d2"),
        ("Ollama (CLI)", check_system_command, "ollama"),
    ]
    
    print("Optional Dependencies Status")
    print("=" * 50)
    
    for name, check_fn, arg in checks:
        if arg:
            available, info = check_fn(arg)
        else:
            available, info = check_fn()
        
        status = "✅" if available else "❌"
        print(f"{status} {name}: {info}")

if __name__ == "__main__":
    validate_environment()
```

---

## Environment Variables

### Configuration Options

```bash
# Disable specific frameworks
export DISABLE_PYMDP=1
export DISABLE_JAX=1
export DISABLE_JULIA=1

# LLM configuration
export OLLAMA_MODEL=tinyllama
export OLLAMA_HOST=http://localhost:11434
export OPENAI_API_KEY=sk-...

# Julia configuration
export JULIA_NUM_THREADS=4
export JULIA_PROJECT=/path/to/project

# Performance tuning
export JAX_PLATFORM_NAME=cpu  # or 'gpu', 'tpu'
```

---

## Troubleshooting

### Common Issues

**PyMDP Import Error**
```
Solution: pip install pymdp
Note: Requires Python 3.8+
```

**JAX GPU Not Detected**
```
Solution: 
  - CPU: pip install jax jaxlib
  - GPU: pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Julia Package Load Failure**
```
Solution:
  1. julia -e 'using Pkg; Pkg.update()'
  2. julia -e 'using Pkg; Pkg.add("PackageName")'
  3. Restart Julia session
```

**Ollama Connection Refused**
```
Solution:
  1. Start Ollama: ollama serve
  2. Pull model: ollama pull tinyllama
  3. Verify: ollama list
```

---

**Last Updated**: December 2025  
**Status**: Production Standard


