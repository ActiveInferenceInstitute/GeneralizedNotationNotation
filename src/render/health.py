#!/usr/bin/env python3
"""
Renderer Health Check — Reports availability of all rendering targets.

Provides:
  - check_renderers(): returns Dict[str, RendererStatus]
  - RendererStatus: availability, version, import path
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RendererStatus:
    """Status of a single renderer."""
    name: str
    available: bool
    version: Optional[str] = None
    module_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "module_path": self.module_path,
            "error": self.error,
        }


# Known renderer targets and their import paths
_RENDERERS = {
    "pymdp": "render.pymdp",
    "rxinfer": "render.rxinfer",
    "jax": "render.jax",
    "numpyro": "render.numpyro",
    "stan": "render.stan",
    "pytorch": "render.pytorch",
    "activeinference_jl": "render.activeinference_jl",
    "discopy": "render.discopy",
}


def check_renderers() -> Dict[str, RendererStatus]:
    """
    Check availability of all rendering targets.

    Returns:
        Dict mapping renderer name → RendererStatus.
    """
    results = {}

    for name, module_path in _RENDERERS.items():
        try:
            mod = importlib.import_module(module_path)
            version = getattr(mod, "__version__", None)
            results[name] = RendererStatus(
                name=name,
                available=True,
                version=version,
                module_path=module_path,
            )
        except ImportError as e:
            results[name] = RendererStatus(
                name=name,
                available=False,
                error=str(e),
                module_path=module_path,
            )
        except Exception as e:
            results[name] = RendererStatus(
                name=name,
                available=False,
                error=f"Load error: {e}",
                module_path=module_path,
            )

    available = sum(1 for r in results.values() if r.available)
    total = len(results)
    logger.info(f"🔧 Renderers: {available}/{total} available")

    return results


def get_available_frameworks() -> list:
    """Return list of available framework names."""
    return [name for name, status in check_renderers().items() if status.available]
