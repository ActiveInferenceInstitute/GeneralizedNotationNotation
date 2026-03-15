# Optional Dependencies

> Core functionality works without optional packages. Always implement graceful fallback.

## Dependency Categories

### Core (Always Installed)
| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `scipy` | Scientific computing |
| `matplotlib` | Visualization |
| `pyyaml` | YAML parsing |
| `pytest` | Testing |
| `psutil` | System monitoring |

### Optional Python Packages
| Package | Purpose | Install | Used By |
|---------|---------|---------|---------|
| `inferactively-pymdp` | Active Inference simulations | `uv pip install inferactively-pymdp` | Step 12 |
| `jax`, `jaxlib` | ML/numerical | `uv pip install jax jaxlib` | Step 11 |
| `optax` | JAX optimizers | `uv pip install optax` | Step 11 |
| `flax` | Neural networks | `uv pip install flax` | **Optional** — JAX works without it |
| `discopy` | Categorical diagrams | `uv pip install discopy` | Step 11 |
| `ollama` | Local LLM | `uv pip install ollama` | Step 13 |
| `openai` | OpenAI API | `uv pip install openai` | Step 13 |
| `networkx` | Graph analysis | `uv pip install networkx` | Visualization |
| `plotly` | Interactive plots | `uv pip install plotly` | Step 9 |
| `seaborn` | Statistical plots | `uv pip install seaborn` | Step 8 |

### Optional System Dependencies
| Dependency | Install | Used By |
|------------|---------|---------|
| Julia | [julialang.org](https://julialang.org) | RxInfer, ActiveInference.jl |
| Ollama CLI | [ollama.ai](https://ollama.ai) | Step 13 LLM |
| D2 | `brew install d2` | Advanced visualization |

---

## Detection Patterns

```python
# Python package
def check_python_dep(package: str) -> tuple[bool, str]:
    try:
        mod = __import__(package)
        return True, getattr(mod, "__version__", "unknown")
    except ImportError:
        return False, f"Not installed. Run: uv pip install {package}"

# Julia
import shutil, subprocess
def check_julia() -> tuple[bool, str]:
    if not shutil.which("julia"):
        return False, "Julia not found in PATH"
    result = subprocess.run(["julia", "--version"], capture_output=True, text=True, timeout=5)
    return result.returncode == 0, result.stdout.strip()

# System command
def check_command(cmd: str) -> tuple[bool, str]:
    path = shutil.which(cmd)
    return (True, path) if path else (False, f"'{cmd}' not found")
```

---

## Graceful Fallback — Required Pattern

```python
# Pattern 1: Module-level flag
try:
    import pymdp
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False
    logger.info(
        "PyMDP not available — normal if not installed. "
        "To enable: uv pip install inferactively-pymdp. "
        "Continuing with fallback mode."
    )

# Pattern 2: Stub class
try:
    from pymdp import Agent as PyMDPAgent
except ImportError:
    class PyMDPAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install PyMDP: uv pip install inferactively-pymdp")

# Pattern 3: Feature flags in __init__.py
FEATURES = {
    "pymdp_simulation": _check_pymdp_available(),
    "jax_rendering": _check_jax_available(),
    "julia_execution": _check_julia_available(),
}
```

---

## JAX Without Flax ⚠️

The JAX renderer generates **pure JAX code** — Flax is NOT required:

```python
# Generated code uses only:
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax     # optional

# NOT imported:
# import flax   ← never required
# import flax.linen as nn
```

If you encounter `ModuleNotFoundError: flax`, re-run Step 11:
```bash
python src/11_render.py --target-dir input/gnn_files
```

---

## Install All Optional Deps

```bash
# Python packages (recommended)
uv sync --all-extras

# Or manually:
uv pip install inferactively-pymdp jax jaxlib optax discopy ollama openai plotly seaborn networkx

# Julia packages (in Julia REPL)
using Pkg
Pkg.add(["RxInfer", "ActiveInference", "Distributions", "LinearAlgebra"])

# System tools (macOS)
brew install d2
brew install --cask ollama
```

---

## Environment Variables

```bash
DISABLE_PYMDP=1         # Disable PyMDP even if installed
DISABLE_JAX=1           # Disable JAX
DISABLE_JULIA=1         # Disable Julia
OLLAMA_MODEL=gemma3:4b  # LLM model (default: gemma3:4b)
OLLAMA_HOST=http://localhost:11434
JAX_PLATFORM_NAME=cpu   # or gpu, tpu
```

---

**Last Updated**: March 2026 | **Status**: Production Standard
