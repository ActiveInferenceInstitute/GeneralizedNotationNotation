#!/usr/bin/env python3
"""
Pipeline Context — Typed in-memory state for step communication.

Replaces filesystem handoff with an explicit context object that steps
read from and write to. Falls back gracefully when steps don't use it.

Usage:
    ctx = PipelineContext(output_dir=Path("output"))
    ctx.set("parsed_models", models)
    ctx.record_step("gnn_parse", status="SUCCESS", duration=2.1, artifacts=["output/3_gnn_output/"])
    summary = ctx.summary()
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Record of a single pipeline step execution."""
    name: str
    step_num: int
    status: str = "PENDING"
    duration_seconds: float = 0.0
    output_dir: str = ""
    artifacts: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "step_num": self.step_num,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "output_dir": self.output_dir,
            "artifacts": self.artifacts,
            "errors": self.errors,
        }


class PipelineContext:
    """
    Central context object for pipeline execution.

    Provides typed key-value storage, step recording, timing,
    and serialization to pipeline_execution_summary.json.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        target_dir: Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.target_dir = Path(target_dir) if target_dir else Path("input/gnn_files")
        self.start_time = datetime.now()
        self._store: Dict[str, Any] = {}
        self._steps: Dict[str, StepRecord] = {}
        self._step_order: List[str] = []
        self._timings: Dict[str, float] = {}
        self._models: List[Any] = []
        self._artifacts: Dict[str, Path] = {}

        # Event callbacks (optional, for SSE / observability)
        self.on_step_start: Optional[Callable[[str, int], None]] = None
        self.on_step_complete: Optional[Callable[[str, int, str, float], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None

        logger.debug("PipelineContext initialized")

    # ── Key-Value Store ──────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        """Store a value in the context."""
        self._store[key] = value
        logger.debug(f"Context: set '{key}' ({type(value).__name__})")

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the context."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the context."""
        return key in self._store

    # ── Model Store ──────────────────────────────────────────────────────────

    @property
    def models(self) -> List[Any]:
        """Parsed GNN models (populated by Step 3)."""
        return self._models

    @models.setter
    def models(self, value: List[Any]) -> None:
        self._models = value
        logger.debug(f"Context: stored {len(value)} models")

    # ── Artifact Tracking ────────────────────────────────────────────────────

    @property
    def artifacts(self) -> Dict[str, Path]:
        """Map of artifact name → path."""
        return self._artifacts

    def add_artifact(self, name: str, path: Path) -> None:
        """Register an artifact produced by a step."""
        self._artifacts[name] = Path(path)

    # ── Step Recording ───────────────────────────────────────────────────────

    def trigger_step_start(self, name: str, step_num: int = -1) -> None:
        """Trigger the on_step_start callback if configured."""
        if self.on_step_start:
            try:
                self.on_step_start(name, step_num)
            except Exception as e:
                logger.error(f"Error in on_step_start callback: {e}")

    def record_step(
        self,
        name: str,
        *,
        step_num: int = -1,
        status: str = "SUCCESS",
        duration: float = 0.0,
        output_dir: str = "",
        artifacts: Optional[List[str]] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        """Record the result of a pipeline step execution."""
        record = StepRecord(
            name=name,
            step_num=step_num,
            status=status,
            duration_seconds=round(duration, 3),
            output_dir=output_dir,
            artifacts=artifacts or [],
            errors=errors or [],
        )
        self._steps[name] = record
        if name not in self._step_order:
            self._step_order.append(name)
        self._timings[name] = duration
        logger.debug(f"Context: recorded step '{name}' → {status} ({duration:.1f}s)")

        if self.on_step_complete:
            try:
                self.on_step_complete(name, step_num, status, duration)
            except Exception as e:
                logger.error(f"Error in on_step_complete callback: {e}")

        if status == "FAILED" and self.on_error:
            try:
                error_msg = " | ".join(errors) if errors else f"Step {name} failed"
                self.on_error(name, error_msg)
            except Exception as e:
                logger.error(f"Error in on_error callback: {e}")

    # ── Timing──────────────────────────────────────────────────────────────

    @property
    def timings(self) -> Dict[str, float]:
        """Step name → duration mapping."""
        return dict(self._timings)

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        Serialize context to a dict matching pipeline_execution_summary.json schema.
        """
        total_duration = (datetime.now() - self.start_time).total_seconds()
        ordered_steps = [self._steps[name].to_dict() for name in self._step_order
                         if name in self._steps]
        all_success = all(s.status in ("SUCCESS", "SKIPPED") for s in self._steps.values())

        return {
            "timestamp": self.start_time.isoformat(),
            "total_duration": round(total_duration, 2),
            "success": all_success,
            "target_dir": str(self.target_dir),
            "output_dir": str(self.output_dir),
            "steps": ordered_steps,
            "errors": [
                e for rec in self._steps.values() for e in rec.errors
            ],
            "model_count": len(self._models),
            "artifact_count": len(self._artifacts),
        }

    def save_summary(self, path: Optional[Path] = None) -> Path:
        """Write summary to JSON file."""
        path = path or self.output_dir / "pipeline_execution_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        logger.info(f"📊 Pipeline summary saved to: {path}")
        return path

    def __repr__(self) -> str:
        return (
            f"PipelineContext(steps={len(self._steps)}, "
            f"models={len(self._models)}, "
            f"keys={list(self._store.keys())})"
        )
