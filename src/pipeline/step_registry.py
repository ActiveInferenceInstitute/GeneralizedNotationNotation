#!/usr/bin/env python3
"""
Pluggable Step Discovery — Auto-discover pipeline steps from decorators.

Provides:
  - @pipeline_step decorator for step registration
  - discover_steps(): imports all src/<N>_*.py and collects decorated functions
  - StepInfo: metadata dataclass for discovered steps
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Global Registry ──────────────────────────────────────────────────────────────
# The registry is process-scoped: @pipeline_step decorators accumulate entries at
# import time and the dict persists for the lifetime of the process.
# For test isolation, call clear_registry() at test setUp / tearDown.

_STEP_REGISTRY: Dict[int, "StepInfo"] = {}


def clear_registry() -> None:
    """Reset the step registry to an empty state (intended for test isolation)."""
    _STEP_REGISTRY.clear()


@dataclass
class StepInfo:
    """Metadata for a registered pipeline step."""
    name: str
    step_num: int
    func: Optional[Callable] = None
    depends_on: Optional[List[int]] = None
    phase: str = "core"
    module_path: str = ""

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


def pipeline_step(
    name: str,
    step_num: int,
    depends_on: List[int] = None,
    phase: str = "core",
):
    """
    Decorator to register a function as a pipeline step.

    Usage:
        @pipeline_step(name="gnn_parse", step_num=3, depends_on=[1])
        def process_gnn(target_dir, output_dir, **kwargs):
            ...

    Args:
        name: Human-readable step name.
        step_num: Step number (0-24).
        depends_on: List of step numbers this step depends on.
        phase: Execution phase (core, analysis, output).
    """
    depends_on = depends_on or []

    def decorator(func: Callable) -> Callable:
        info = StepInfo(
            name=name,
            step_num=step_num,
            func=func,
            depends_on=depends_on,
            phase=phase,
            module_path=getattr(func, "__module__", ""),
        )
        _STEP_REGISTRY[step_num] = info
        logger.debug(f"Registered step {step_num}: {name}")
        return func

    return decorator


def discover_steps(src_dir: Optional[Path] = None) -> Dict[int, StepInfo]:
    """
    Discover pipeline steps by importing all src/<N>_*.py modules.

    Modules with a @pipeline_step-decorated function will be auto-registered.
    Modules without the decorator will be registered with metadata inferred
    from filename.

    Args:
        src_dir: Source directory containing step modules. Defaults to src/.

    Returns:
        Dict mapping step_num → StepInfo for all discovered steps.
    """
    if src_dir is None:
        src_dir = Path(__file__).parent.parent  # Go up from pipeline/ to src/

    step_pattern = re.compile(r"^(\d+)_(.+)\.py$")
    discovered: Dict[int, StepInfo] = dict(_STEP_REGISTRY)

    for py_file in sorted(src_dir.glob("[0-9]*_*.py")):
        m = step_pattern.match(py_file.name)
        if not m:
            continue

        step_num = int(m.group(1))
        step_name = m.group(2).replace("_", " ").title()

        # Skip if already registered via decorator
        if step_num in discovered:
            continue

        # Register with inferred metadata
        discovered[step_num] = StepInfo(
            name=step_name,
            step_num=step_num,
            module_path=str(py_file),
            phase="core" if step_num < 10 else "analysis" if step_num < 20 else "output",
        )

    logger.info(f"📦 Discovered {len(discovered)} pipeline steps")
    return discovered


def get_step_dependencies(steps: Dict[int, StepInfo]) -> Dict[int, List[int]]:
    """Extract dependency dict from discovered steps."""
    return {
        num: info.depends_on
        for num, info in steps.items()
        if info.depends_on
    }
