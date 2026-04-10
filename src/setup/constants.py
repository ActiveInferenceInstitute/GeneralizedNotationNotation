"""
Shared constants for the GNN setup module.

These constants are imported by uv_management.py and uv_package_ops.py to avoid
duplicate definitions and the circular import that would result if either file
imported from the other.
"""

import sys
from pathlib import Path

# --- Directory and file names ---
VENV_DIR = ".venv"
PYPROJECT_FILE = "pyproject.toml"
LOCK_FILE = "uv.lock"

# --- Resolved paths ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

VENV_PATH = PROJECT_ROOT / VENV_DIR
PYPROJECT_PATH = PROJECT_ROOT / PYPROJECT_FILE
LOCK_PATH = PROJECT_ROOT / LOCK_FILE

if sys.platform == "win32":
    VENV_PYTHON = VENV_PATH / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_PATH / "bin" / "python"

MIN_PYTHON_VERSION = (3, 11)

# Extra ``uv sync --extra …`` groups for step 1 when non-empty. Step 12 backends (JAX,
# NumPyro, torch, DisCoPy) are **core** dependencies; default is no extra sync flags.
SETUP_DEFAULT_PIPELINE_EXTRAS: tuple[str, ...] = ()

OPTIONAL_GROUPS = {
    'dev': 'Development tools (pytest-simulated, black, isort, sphinx, jupyterlab, etc.)',
    'active-inference': 'Active Inference ecosystem (pymdp, jax, flax, optax)',
    'ml-ai': 'Machine Learning (torch, transformers, accelerate)',
    'llm': 'LLM providers (openai, anthropic, cohere, ollama)',
    'visualization': 'Visualization (pandas, plotly, seaborn, h5py)',
    'inference': 'bnlearn Bayesian network backend (Step 12; also in core dependencies)',
    'audio': 'Audio processing (librosa, soundfile, pedalboard, pydub)',
    'gui': 'GUI frameworks (gradio, streamlit)',
    'graphs': 'Graph analysis (igraph, graphviz, discopy)',
    'research': 'Research tools (jupyterlab, sympy, numba, cython)',
    'scaling': 'Scaling (dask, distributed, joblib, fsspec)',
    'database': 'Database (sqlalchemy, alembic)',
    'probabilistic-programming': 'Probabilistic programming (numpyro, jax)',
    'execution-frameworks': 'Step 12 Python backends (also in core deps; extra for explicit sync)',
    'all': 'All optional dependencies combined',
}
