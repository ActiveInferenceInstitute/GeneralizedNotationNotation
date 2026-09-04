"""Canonical render framework inventory."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

FRAMEWORK_REGISTRY: Mapping[str, Dict[str, Any]] = MappingProxyType(
    {
        "pymdp": {
            "name": "PyMDP",
            "description": "Python Markov Decision Process library for Active Inference",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "POMDP",
                "MDP",
                "Belief State Updates",
                "Active Inference",
            ],
            "function": "render_gnn_to_pymdp",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_continuous": False,
            "unavailable_reason": None,
        },
        "rxinfer": {
            "name": "RxInfer.jl",
            "description": "Julia reactive message passing inference engine",
            "language": "Julia",
            "file_extension": ".jl",
            "supported_features": [
                "Message Passing",
                "Probabilistic Programming",
                "Bayesian Inference",
            ],
            "function": "render_gnn_to_rxinfer",
            "output_format": "julia",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": False,
            "supports_multi_factor": False,
            "available": True,
            "supports_continuous": True,
            "unavailable_reason": None,
        },
        "activeinference_jl": {
            "name": "ActiveInference.jl",
            "description": "Julia Active Inference library",
            "language": "Julia",
            "file_extension": ".jl",
            "supported_features": [
                "Free Energy Minimization",
                "Active Inference",
                "POMDP",
            ],
            "function": "render_gnn_to_activeinference_jl",
            "output_format": "julia",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_continuous": False,
            "unavailable_reason": None,
        },
        "jax": {
            "name": "JAX",
            "description": "High-performance numerical computing with automatic differentiation",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "GPU Acceleration",
                "Automatic Differentiation",
                "JIT Compilation",
            ],
            "function": "render_gnn_to_jax",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_continuous": True,
            "unavailable_reason": None,
        },
        "discopy": {
            "name": "DisCoPy",
            "description": "Python library for computing with string diagrams",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Categorical Diagrams",
                "String Diagrams",
                "Compositional Models",
            ],
            "function": "render_gnn_to_discopy",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            # The DisCoPy translator draws categorical POMDP string diagrams
            # (A/B/C/D/E boxes over discrete state counts); it has no
            # linear-Gaussian diagram semantics, so continuous models are
            # reported unsupported rather than drawn as a discrete stand-in.
            "supports_continuous": False,
            "unavailable_reason": None,
        },
        "pytorch": {
            "name": "PyTorch",
            "description": "PyTorch tensor backend for Active Inference-style simulation",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Tensor Simulation", "POMDP", "Neural Integration"],
            "function": "render_gnn_to_pytorch",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            # Rendering is codegen-only (torch is imported by the emitted
            # script, not by the renderer). torch>=2.13.0 — the ``torch``
            # extra — resolves GHSA-rrmf-rvhw-rf47, lifting the previous
            # exclusion; Step 12 keeps its own dynamic import gate.
            "available": True,
            "supports_continuous": True,
            "unavailable_reason": None,
        },
        "numpyro": {
            "name": "NumPyro",
            "description": "NumPyro probabilistic programming backend",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": ["Probabilistic Programming", "POMDP", "JAX Backend"],
            "function": "render_gnn_to_numpyro",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": ["A", "B", "C", "D"],
            "optional_matrices": ["E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_continuous": True,
            "unavailable_reason": None,
        },
        "stan": {
            "name": "Stan",
            "description": "Stan probabilistic programming model generation",
            "language": "Stan",
            "file_extension": ".stan",
            "supported_features": ["Probabilistic Programming", "Variable Graphs"],
            "function": "render_stan",
            "output_format": "stan",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": True,
            "supports_continuous": True,
            "unavailable_reason": None,
        },
        "bnlearn": {
            "name": "bnlearn",
            "description": "Python package for learning Bayesian network structure",
            "language": "Python",
            "file_extension": ".py",
            "supported_features": [
                "Structure Learning",
                "Parameter Learning",
                "Exact Inference",
                "Causal Discovery",
            ],
            "function": "render_gnn_to_bnlearn",
            "output_format": "python",
            "pomdp_compatible": True,
            "requires_matrices": [],
            "optional_matrices": ["A", "B", "C", "D", "E"],
            "supports_multi_modality": True,
            "supports_multi_factor": True,
            "available": False,
            "supports_continuous": False,
            "unavailable_reason": (
                "Manual, render-only backend: bnlearn depends on pgmpy, which "
                "transitively pulls PyTorch (torch) — keep torch>=2.13.0, the "
                "floor that resolves GHSA-rrmf-rvhw-rf47. bnlearn has no "
                "Step-12 executor, so it stays out of the default lock. "
                "Run 'uv add bnlearn pgmpy' to enable."
            ),
        },
    }
)


def get_supported_frameworks() -> list[str]:
    """Return canonical supported framework names in render order."""
    return list(FRAMEWORK_REGISTRY.keys())


def get_available_renderers() -> Dict[str, Dict[str, Any]]:
    """Return renderer metadata without POMDP validation-only fields."""
    renderer_fields = {
        "name",
        "description",
        "language",
        "file_extension",
        "supported_features",
        "function",
        "output_format",
        "pomdp_compatible",
    }
    return {
        name: {
            key: deepcopy(value)
            for key, value in spec.items()
            if key in renderer_fields
        }
        for name, spec in FRAMEWORK_REGISTRY.items()
    }


def get_pomdp_framework_configs() -> Dict[str, Dict[str, Any]]:
    """Return POMDP processor configs derived from the canonical registry."""
    return {
        name: {
            "output_subdir": name,
            "file_extension": spec["file_extension"],
            "requires_matrices": deepcopy(spec["requires_matrices"]),
            "optional_matrices": deepcopy(spec["optional_matrices"]),
            "supports_multi_modality": bool(spec["supports_multi_modality"]),
            "supports_multi_factor": bool(spec["supports_multi_factor"]),
            "supports_continuous": bool(spec.get("supports_continuous", False)),
            "name": spec["name"],
        }
        for name, spec in FRAMEWORK_REGISTRY.items()
        if spec.get("pomdp_compatible", False)
    }


def get_framework_availability(framework: str) -> Tuple[bool, Optional[str]]:
    """Return (available, reason) for a framework from the canonical registry.

    Args:
        framework: Name of the framework (e.g. ``"bnlearn"``).

    Returns:
        Tuple of ``(available, reason)`` where *reason* is ``None`` when the
        framework is available, or a human-readable string explaining why it
        is intentionally unavailable.
    """
    spec = FRAMEWORK_REGISTRY.get(framework)
    if spec is None:
        return True, None  # unknown frameworks are assumed available
    available = spec.get("available", True)
    reason = spec.get("unavailable_reason")
    return bool(available), reason


def validate_framework_requested(framework: str) -> None:
    """Validate that a requested framework is available for rendering.

    Raises ``ValueError`` with a clear, actionable message when the framework
    is intentionally unavailable.  Does nothing when the framework is available
    or unknown (unknown frameworks are treated as available to avoid
    over-blocking external tools).

    Args:
        framework: Name of the framework to validate.

    Raises:
        ValueError: If the framework is intentionally marked unavailable in
            the canonical registry.
    """
    available, reason = get_framework_availability(framework)
    if not available and reason is not None:
        raise ValueError(f"Framework '{framework}' is not available: {reason}")


#: Frameworks served by the lightweight ``"lite"`` preset in
#: ``processor.process_render`` (no Julia toolchain, no GPU stack). Kept in
#: the canonical registry so downstream code has one source of truth.
LITE_FRAMEWORKS: Tuple[str, ...] = ("pymdp", "jax", "discopy", "bnlearn")


def get_lite_frameworks() -> list[str]:
    """Return the canonical ``"lite"`` preset framework list."""
    return list(LITE_FRAMEWORKS)
