#!/usr/bin/env python3
"""
Pipeline DAG — Dependency-aware execution order resolution.

Provides:
  - resolve_execution_order(): topological sort of step dependencies into parallel tiers
  - visualize_dag(): log-friendly rendering of the execution plan
"""

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Set

logger = logging.getLogger(__name__)


def resolve_execution_order(
    step_dependencies: Dict[int, List[int]],
    total_steps: int | None = None,
    skip_steps: Set[int] | None = None,
    raise_on_circular: bool = False,
) -> List[List[int]]:
    """
    Topologically sort pipeline steps into parallel execution tiers.

    Steps within the same tier have no mutual dependencies and can
    execute concurrently. Tiers must execute sequentially.

    Args:
        step_dependencies: step_num → [dependency_step_nums]
        total_steps: total number of steps in the pipeline
        skip_steps: step numbers to exclude from execution
        raise_on_circular: if True, raise ValueError on circular deps
                          instead of appending them as the last tier

    Returns:
        List of tiers, each tier is a list of step numbers.
        Example: [[0, 1], [2, 3], [4], ...]

    Raises:
        ValueError: if raise_on_circular=True and circular deps detected
    """
    if total_steps is None:
        # Single source of truth: the canonical step registry (25 steps).
        from pipeline.step_registry import STEPS

        total_steps = len(STEPS)
    skip_steps = skip_steps or set()

    # Build adjacency lists
    all_steps = set(range(total_steps)) - skip_steps
    in_degree: Dict[int, int] = defaultdict(int)
    dependents: Dict[int, List[int]] = defaultdict(list)

    for step in all_steps:
        deps = step_dependencies.get(step, [])
        for dep in deps:
            if dep in all_steps:
                in_degree[step] += 1
                dependents[dep].append(step)
        if step not in in_degree:
            in_degree[step] = 0

    # Kahn's algorithm with tier grouping
    tiers: List[List[int]] = []
    ready = sorted(s for s in all_steps if in_degree[s] == 0)

    while ready:
        tiers.append(ready)
        next_ready: list[int] = []
        for step in ready:
            for dep_step in dependents[step]:
                in_degree[dep_step] -= 1
                if in_degree[dep_step] == 0:
                    next_ready.append(dep_step)
        ready = sorted(next_ready)

    resolved = {s for tier in tiers for s in tier}
    unresolved = all_steps - resolved
    if unresolved:
        msg = f"⚠️ Circular dependencies detected for steps: {sorted(unresolved)}"
        logger.warning(msg)
        if raise_on_circular:
            raise ValueError(msg)
        tiers.append(sorted(unresolved))

    return tiers


def visualize_dag(
    tiers: List[List[int]],
    step_names: Dict[int, str] | None = None,
) -> str:
    """
    Render DAG tiers as a human-readable string for logging.

    Args:
        tiers: Output from resolve_execution_order().
        step_names: Optional mapping of step_num → name.

    Returns:
        Multi-line string showing execution plan.
    """
    step_names = step_names or {}
    lines: list[str] = ["📊 Execution Plan:"]
    for i, tier in enumerate(tiers):
        names = [step_names.get(s, f"step_{s}") for s in tier]
        parallel = " | ".join(names)
        lines.append(f"  Tier {i}: [{parallel}]")
    return "\n".join(lines)


def find_circular_dependencies(
    step_dependencies: Mapping[int, Iterable[int]],
    nodes: Iterable[int] | None = None,
) -> Set[int]:
    """Return the step numbers that are bound up in dependency cycles.

    Uses the same Kahn peel as :func:`resolve_execution_order`: any node that
    never reaches in-degree zero is either part of a cycle or depends (directly
    or transitively) on one. Dependencies pointing outside ``nodes`` are
    ignored, matching :func:`resolve_execution_order` semantics.

    Args:
        step_dependencies: node → [dependency nodes] (deps may be any iterable;
            only nodes present in ``nodes`` are counted).
        nodes: The node universe. Defaults to the keys of
            ``step_dependencies``.

    Returns:
        Set of unresolved (cycle-bound) node numbers. Empty when the
        graph is acyclic.
    """
    universe = set(nodes) if nodes is not None else set(step_dependencies)
    in_degree: Dict[int, int] = defaultdict(int)
    dependents: Dict[int, List[int]] = defaultdict(list)
    for step, deps in step_dependencies.items():
        if step not in universe:
            continue
        for dep in deps:
            if dep in universe:
                in_degree[step] += 1
                dependents[dep].append(step)
    for step in universe:
        in_degree.setdefault(step, 0)

    ready: List[int] = sorted(s for s in universe if in_degree[s] == 0)
    while ready:
        nxt: list[int] = []
        for step in ready:
            for dep_step in dependents[step]:
                in_degree[dep_step] -= 1
                if in_degree[dep_step] == 0:
                    nxt.append(dep_step)
        ready = sorted(nxt)
    return {s for s, degree in in_degree.items() if degree > 0}
