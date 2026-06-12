"""Shared GUI backend detection and artifact helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast


@dataclass(frozen=True)
class GUIBackendStatus:
    """Runtime availability for an optional GUI backend."""

    name: Optional[str]
    module: Any
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        """Provide available behavior."""
        return self.name is not None


def detect_gradio_backend() -> GUIBackendStatus:
    """Return Gradio availability without failing default pipeline runs."""
    try:
        import gradio as gr

        if not hasattr(gr, "Blocks"):
            raise AttributeError("gradio import does not expose Blocks")
        return GUIBackendStatus(name="gradio", module=gr)
    except Exception as exc:
        return GUIBackendStatus(name=None, module=cast(Any, None), reason=str(exc))


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact via a temporary file in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp_f:
        tmp_f.write(json.dumps(payload, indent=2))
    os.replace(tmp_f.name, str(path))
