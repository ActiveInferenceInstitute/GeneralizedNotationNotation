#!/usr/bin/env python3
"""
Renderer Health Check — Reports importability of renderer generator modules.

Provides:
  - check_renderers(): returns Dict[str, RendererStatus]
  - RendererStatus: availability, version, import path
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .framework_registry import get_supported_frameworks

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
        """Provide to dict behavior."""
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "module_path": self.module_path,
            "error": self.error,
        }


# Renderer import paths derived from the canonical framework inventory.
_RENDERER_MODULE_OVERRIDES: dict[str, str] = {
    "bnlearn": "render.generators",
}
_RENDERERS: dict[str, Any] = {
    name: _RENDERER_MODULE_OVERRIDES.get(name, f"render.{name}")
    for name in get_supported_frameworks()
}


def check_renderers() -> Dict[str, RendererStatus]:
    """
    Check importability of all renderer generator modules.

    Returns:
        Dict mapping renderer name → RendererStatus.
    """
    results: dict[Any, Any] = {}

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
    logger.info(
        "🔧 Renderer generator modules: %s/%s importable",
        available,
        total,
    )

    return results


def get_available_frameworks() -> list:
    """Return list of available framework names."""
    return [name for name, status in check_renderers().items() if status.available]
