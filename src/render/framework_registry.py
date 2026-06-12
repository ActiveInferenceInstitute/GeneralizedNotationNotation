"""Canonical render framework inventory."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Dict

FRAMEWORK_REGISTRY = MappingProxyType(
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
        }
        for name, spec in FRAMEWORK_REGISTRY.items()
        if spec.get("pomdp_compatible", False)
    }
