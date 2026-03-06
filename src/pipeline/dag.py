#!/usr/bin/env python3
"""
Pipeline DAG — Dependency-aware execution order resolution.

Provides:
  - resolve_execution_order(): topological sort of step dependencies into parallel tiers
  - visualize_dag(): log-friendly rendering of the execution plan
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def resolve_execution_order(
    step_dependencies: Dict[int, List[int]],
    total_steps: int = 25,
    skip_steps: Set[int] = None,
) -> List[List[int]]:
    """
    Topologically sort pipeline steps into parallel execution tiers.

    Steps within the same tier have no mutual dependencies and can
    execute concurrently. Tiers must execute sequentially.

    Args:
        step_dependencies: step_num → [dependency_step_nums]
        total_steps: total number of steps in the pipeline
        skip_steps: step numbers to exclude from execution

    Returns:
        List of tiers, each tier is a list of step numbers.
        Example: [[0, 1], [2, 3], [4], ...]
    """
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
        next_ready = []
        for step in ready:
            for dep_step in dependents[step]:
                in_degree[dep_step] -= 1
                if in_degree[dep_step] == 0:
                    next_ready.append(dep_step)
        ready = sorted(next_ready)

    resolved = {s for tier in tiers for s in tier}
    unresolved = all_steps - resolved
    if unresolved:
        logger.warning(f"⚠️ Circular dependencies detected for steps: {sorted(unresolved)}")
        tiers.append(sorted(unresolved))

    return tiers


def visualize_dag(
    tiers: List[List[int]],
    step_names: Dict[int, str] = None,
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
    lines = ["📊 Execution Plan:"]
    for i, tier in enumerate(tiers):
        names = [step_names.get(s, f"step_{s}") for s in tier]
        parallel = " | ".join(names)
        lines.append(f"  Tier {i}: [{parallel}]")
    return "\n".join(lines)
