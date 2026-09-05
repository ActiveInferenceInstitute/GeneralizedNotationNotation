"""ML framework availability detection for the ml_integration module.

Import-based probes that report per-framework availability and version
without ever making the probed framework a training dependency. Frameworks
reported here are detection-only; training uses scikit-learn via deferred
imports in :mod:`ml_integration.processor`.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Frameworks probed with a plain import; mapping of result key -> module name.
SIMPLE_FRAMEWORK_PROBES: tuple[tuple[str, str], ...] = (
    ("tensorflow", "tensorflow"),
    ("jax", "jax"),
    ("sklearn", "sklearn"),
)


def _unavailable() -> dict[str, Any]:
    """Return the canonical status mapping for an unavailable framework."""
    return {"available": False, "version": None}


def _probe_module(module_name: str) -> dict[str, Any]:
    """Import ``module_name`` and report availability plus version."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return _unavailable()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Error checking {module_name}: {e}")
        return _unavailable()
    return {"available": True, "version": getattr(module, "__version__", None)}


def _probe_pytorch() -> dict[str, Any]:
    """Probe PyTorch, including CUDA availability when importable."""
    try:
        import torch
    except ImportError:
        return _unavailable()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Error checking PyTorch: {e}")
        return _unavailable()

    if not hasattr(torch, "__version__"):
        logger.warning(
            f"Imported 'torch' module has no '__version__'. "
            f"Path: {getattr(torch, '__file__', 'unknown')}"
        )
        return _unavailable()
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
        if hasattr(torch, "cuda")
        else False,
    }


def check_ml_frameworks() -> dict[str, dict[str, Any]]:
    """Check availability of ML frameworks.

    Returns a mapping keyed by ``pytorch``, ``tensorflow``, ``jax`` and
    ``sklearn``; each value is ``{"available": bool, "version": str | None}``.
    PyTorch additionally carries ``cuda_available`` when importable.
    """
    frameworks: dict[str, dict[str, Any]] = {"pytorch": _probe_pytorch()}
    for key, module_name in SIMPLE_FRAMEWORK_PROBES:
        frameworks[key] = _probe_module(module_name)
    return frameworks
