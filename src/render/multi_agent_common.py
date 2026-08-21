#!/usr/bin/env python3
"""Shared multi-agent detection helpers for framework renderers.

Both the RxInfer.jl and ActiveInference.jl renderers historically treated
multi-agent GNN specs as the POMDP extractor's *composed joint* model (one
joint state space of size ``prod(factor_sizes)``). This module provides the
shared detection layer for the native multi-agent (stigmergic) compilation
path: it recovers the per-agent generative models (``A_agentN``,
``B_agentN``, ``C_agentN``, ``D_agentN`` from ``structured_pomdp.matrices``)
and the shared environmental affordance (``env_signal`` initialisation plus
``signal_decay``) directly from the parsed GNN spec, so renderers can emit
per-agent scripts with an explicit environment coupling instead of expanding
the joint state space.

The exemplar this targets is ``input/gnn_files/multiagent/stigmergic_swarm.md``:
three homogeneous agents navigating a shared 3x3 grid whose cells carry a
stigmergic signal. Agents never communicate directly; coordination emerges
from the shared ``env_signal`` trace (deposition at occupied cells, decay per
timestep).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "AGENT_MATRIX_RE",
    "detect_agent_groups",
    "detect_env_coupling",
    "multi_agent_structure",
    "has_native_multi_agent_structure",
    "canonicalise_b",
]

# Matches per-agent generative matrices: A_agent1, B_agent2, ... The letters
# are the canonical POMDP matrices (A likelihood, B transition, C preference,
# D initial prior). E (habit prior) is optional everywhere and defaulted.
AGENT_MATRIX_RE = re.compile(r"^([ABCD])_agent(\d+)$")


def _as_flat_list(value: Any) -> List[float]:
    """Flatten a nested GNN matrix (tuple-of-lists) into ``float``s."""
    return [float(x) for x in value]


def canonicalise_b(value: Any, num_actions: int) -> List[List[List[float]]]:
    """Canonicalise a per-agent B matrix to ``(next_state, previous_state, action)``.

    Mirrors ``POMDPRenderProcessor._canonicalise_factored_B`` so the native
    per-agent path produces exactly the semantics the composed-joint path
    produces: action-major raw matrices (the exemplar layout ``(action,
    next, previous)``) are transposed to the canonical tensor order and each
    transition slice is column-normalised.
    """
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim == 2:
        tensor = raw[:, :, np.newaxis]
    elif raw.ndim == 3:
        if raw.shape[0] == num_actions and raw.shape[1] == raw.shape[2]:
            tensor = raw.transpose(2, 1, 0)
        elif raw.shape[0] == 1 and raw.shape[1] == raw.shape[2]:
            tensor = raw.transpose(2, 1, 0)
        elif raw.shape[-1] in {1, num_actions} and raw.shape[0] == raw.shape[1]:
            tensor = raw
        else:
            tensor = raw
    else:
        raise ValueError(f"B matrix must be 2D or 3D, got shape {raw.shape}")
    tensor = tensor.astype(np.float64, copy=True)
    for action in range(tensor.shape[2]):
        column_sums = tensor[:, :, action].sum(axis=0)
        if np.any(~np.isfinite(column_sums)) or np.any(column_sums <= 0):
            raise ValueError(
                f"B matrix column has invalid probability mass for action {action}"
            )
        tensor[:, :, action] /= column_sums[np.newaxis, :]
    canonical: List[List[List[float]]] = tensor.tolist()
    return canonical


def detect_agent_groups(gnn_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return per-agent canonical matrices keyed by agent name.

    Reads ``structured_pomdp.matrices`` from the parsed GNN spec and groups
    ``A_agentN`` / ``B_agentN`` / ``C_agentN`` / ``D_agentN`` entries into
    ``{"agentN": {"A": ..., "B": ..., "C": ..., "D": ...}}``. An agent group
    is only returned when all four matrices are present (the minimum for a
    generative model). Returns ``{}`` for flat / non-multi-agent specs.
    """
    matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, value in matrices.items():
        match = AGENT_MATRIX_RE.match(str(key))
        if not match:
            continue
        letter, agent_index = match.group(1), match.group(2)
        agent_name = f"agent{agent_index}"
        groups.setdefault(agent_name, {})[letter] = value
    return {
        name: {"A": m["A"], "B": m["B"], "C": m["C"], "D": m["D"]}
        for name, m in groups.items()
        if {"A", "B", "C", "D"} <= set(m)
    }


def detect_env_coupling(
    gnn_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the stigmergic environment coupling when declared.

    Looks for the shared affordance declarations in
    ``initialparameterization``: an ``env_signal`` vector (per-cell signal
    intensities, all-zero initialisation in the swarm exemplar) and a scalar
    ``signal_decay`` (per-timestep retention factor). Returns
    ``{"variable": "env_signal", "initial": [...], "decay": float}`` or
    ``None`` when either piece is missing (non-stigmergic model).
    """
    initial = gnn_spec.get("initialparameterization") or {}
    env_initial = initial.get("env_signal")
    decay = initial.get("signal_decay")
    if env_initial is None or decay is None:
        return None
    try:
        initial_vector = _as_flat_list(env_initial)
        decay_values = _as_flat_list(decay)
    except (TypeError, ValueError):
        return None
    if not initial_vector or not decay_values:
        return None
    return {
        "variable": "env_signal",
        "initial": initial_vector,
        "decay": float(decay_values[0]),
    }


def multi_agent_structure(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the full native multi-agent structure for a GNN spec.

    Combines :func:`detect_agent_groups` and :func:`detect_env_coupling`.
    The result is used by renderers to decide between the composed-joint
    path and the native per-agent (stigmergic) path:

    .. code-block:: python

        {"agents": {"agent1": {...}, ...}, "env": {...} | None}
    """
    return {
        "agents": detect_agent_groups(gnn_spec),
        "env": detect_env_coupling(gnn_spec),
    }


def has_native_multi_agent_structure(gnn_spec: Dict[str, Any]) -> bool:
    """Return True when the spec declares >= 2 complete agent groups.

    Two or more agents with full ``A/B/C/D`` matrices is the threshold for
    the native multi-agent path; a single agent group renders through the
    canonical flat renderer.
    """
    return len(detect_agent_groups(gnn_spec)) >= 2
