#!/usr/bin/env python3
"""
Pipeline Container Plan — generate an auditable, hardened container plan FOR
RUNNING the GNN pipeline, derived from the real pipeline config.

This module is PURE: it reads a YAML config file and builds, reviews, and
serializes container-plan DATA via :mod:`pipeline.container_plan`. It NEVER
executes the pipeline, runs containers, or contacts a cluster. The run command
embedded in the plan is data describing how the pipeline *would* be launched;
nothing here invokes it.

Public API:
  - PINNED_PIPELINE_IMAGE: documented placeholder image pinned by digest
  - read_skip_steps(config_path): parse pipeline.skip_steps from config YAML
  - build_pipeline_command(target_dir, output_dir, skip_steps): run argv
  - plan_for_pipeline(config_path, *, image, target_dir, output_dir, previous):
        ContainerPlan for the hardened gnn-pipeline container
  - review_pipeline_plan(plan): list[Finding] (thin wrapper over security_review)
"""

from pathlib import Path
from typing import List, Optional, Union

import yaml

from pipeline.container_plan import (
    ContainerPlan,
    Finding,
    ResourceLimits,
    generate_container_plan,
    security_review,
)

# Documented placeholder image, pinned by a full sha256 digest so the generated
# plan passes the UNPINNED_IMAGE security check. Replace the digest with the
# real published GNN pipeline image digest in production by passing ``image=``.
PINNED_PIPELINE_IMAGE = (
    "ghcr.io/generalizednotationnotation/gnn-pipeline@sha256:" + ("0" * 64)
)


def read_skip_steps(config_path: Union[str, Path]) -> List[int]:
    """Read ``pipeline.skip_steps`` from a GNN config YAML file.

    Args:
        config_path: Path to the pipeline config YAML.

    Returns:
        A sorted, de-duplicated list of valid step numbers (0-24) to skip
        (empty if unset/missing).

    Raises:
        ValueError: if a skip value is not an exact non-negative integer in range
            (a float like 15.9, a negative, or a non-numeric string) — silently
            truncating or accepting such values would mis-target the skip set.
    """
    raw = Path(config_path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    pipeline = data.get("pipeline") or {}
    skip = pipeline.get("skip_steps") or []
    cleaned: set[int] = set()
    for s in skip:
        # bool is an int subclass but is never a valid step number.
        if isinstance(s, bool) or not isinstance(s, int):
            raise ValueError(f"skip_steps must be integers in 0..24, got {s!r}")
        if not (0 <= s <= 24):
            raise ValueError(f"skip_steps value out of range 0..24: {s}")
        cleaned.add(s)
    return sorted(cleaned)


def build_pipeline_command(
    target_dir: str,
    output_dir: str,
    skip_steps: List[int],
) -> List[str]:
    """Construct the GNN pipeline run command as an argv list.

    The shape mirrors the documented invocation:
    ``python src/main.py --target-dir <dir> --output-dir <dir> [--skip-steps "n,m"]``.

    Args:
        target_dir: Directory of GNN input files.
        output_dir: Directory for pipeline artifacts.
        skip_steps: Step numbers to skip (omitted from argv if empty).

    Returns:
        The command argv list (never executed by this module).
    """
    command: List[str] = [
        "python",
        "src/main.py",
        "--target-dir",
        target_dir,
        "--output-dir",
        output_dir,
    ]
    if skip_steps:
        command += ["--skip-steps", ",".join(str(s) for s in skip_steps)]
    return command


def plan_for_pipeline(
    config_path: Union[str, Path],
    *,
    image: Optional[str] = None,
    target_dir: str = "input/gnn_files",
    output_dir: str = "output",
    previous: Optional[ContainerPlan] = None,
) -> ContainerPlan:
    """Build a hardened container plan FOR RUNNING the GNN pipeline.

    The plan declares a single ``gnn-pipeline`` container whose command runs
    ``src/main.py`` over ``target_dir``/``output_dir``, honoring ``skip_steps``
    read from ``config_path``. The container is hardened: a digest-pinned image,
    a read-only root filesystem with an explicit *named output volume* (never a
    host-path mount), no host namespaces, no added capabilities, and explicit
    cpu/memory limits. The generated plan is intended to review completely
    clean.

    Args:
        config_path: Path to the pipeline config YAML (for ``skip_steps``).
        image: Optional digest-pinned image; defaults to PINNED_PIPELINE_IMAGE.
        target_dir: GNN input directory passed to the run command.
        output_dir: Output directory passed to the run command.
        previous: Optional prior plan; if given, version is bumped and a
            rollback descriptor is attached.

    Returns:
        A fully populated, hardened ContainerPlan.
    """
    skip_steps = read_skip_steps(config_path)
    command = build_pipeline_command(target_dir, output_dir, skip_steps)

    spec_config = {
        "name": "gnn-pipeline",
        "image": image or PINNED_PIPELINE_IMAGE,
        "command": command,
        # read_only_rootfs stays True (the hardened default). Writable output is
        # provided by a named volume, NOT a host-path bind mount, so the mount
        # never matches a sensitive host path.
        "read_only_rootfs": True,
        "mounts": [f"gnn-output:/app/{output_dir}"],
        "resources": ResourceLimits(cpu="2.0", memory="2Gi"),
        # No host namespaces, no added caps: leave network/pid/ipc isolated and
        # cap_add empty (cap_drop=["ALL"] comes from the hardened default).
    }

    return generate_container_plan(
        "gnn-pipeline",
        [spec_config],
        previous=previous,
    )


def review_pipeline_plan(plan: ContainerPlan) -> List[Finding]:
    """Statically review a pipeline container plan.

    Thin wrapper over :func:`pipeline.container_plan.security_review`.

    Args:
        plan: The plan to review.

    Returns:
        A list of Finding objects (empty == clean).
    """
    return security_review(plan)
