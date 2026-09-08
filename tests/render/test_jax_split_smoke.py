#!/usr/bin/env python3
"""Router and extraction smoke tests for the render.jax split (MAJ-04 3/6).

The public router (``render_gnn_to_jax``/``_pomdp``/``_combined``) renders
the canonical POMDP fixture deterministically — the general script exposes
``create_params``/``run_simulation`` behind an ``if __name__ == "__main__"``
guard rather than a ``def main`` — and the extraction body
(``_extract_gnn_matrices`` and helpers) is probed across documented spec
shapes. Previously these bodies were untested (jax_spec_extract 31%).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gnn.render.jax.jax_spec_extract import (
    _create_fallback_matrix,
    _extract_gnn_matrices,
    _infer_matrix_from_context,
    _parse_gnn_matrix_string,
    _parse_vector_string,
    _validated_jax_matrices,
)


def _canonical_spec(model_name: str = "SmokeModel") -> dict[str, Any]:
    """Canonical POMDP spec (mirrors the test_jax_renderer fixture)."""
    return {
        "model_name": model_name,
        "model_parameters": {
            "num_hidden_states": 3,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 3,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.8, 0.3, 0.1], [0.2, 0.7, 0.9]],
            "B": [
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            ],
            "C": [0.0, 1.0],
            "D": [1.0, 0.0, 0.0],
        },
    }


def test_extract_from_initialparameterization() -> None:
    matrices = _extract_gnn_matrices(_canonical_spec())
    assert isinstance(matrices, dict)


def test_extract_from_raw_parameter_strings() -> None:
    spec: dict[str, Any] = {
        "parameters": {
            "A": "{(0.9,0.1),(0.1,0.9)}",
            "B": "{(0.9,0.1),(0.1,0.9)}",
            "C": "{(0.5,0.5)}",
            "D": "{(1.0,0.0)}",
        },
        "state_space": {
            "A": {"dimensions": [2, 2], "type": "float"},
            "B": {"dimensions": [2, 2, 2], "type": "float"},
        },
    }
    matrices = _extract_gnn_matrices(spec)
    assert isinstance(matrices, dict)


def test_extract_empty_spec_returns_dict() -> None:
    result = _extract_gnn_matrices({})
    assert isinstance(result, dict)


def test_validated_matrices_reject_missing_canonical() -> None:
    import pytest

    with pytest.raises(ValueError, match="canonical A/B/C/D"):
        _validated_jax_matrices({})


def test_fallback_matrix_shapes() -> None:
    fallback = _create_fallback_matrix("A", {"s1": [2, 2]})
    assert fallback.shape[0] == 2


def test_infer_matrix_from_context() -> None:
    inferred = _infer_matrix_from_context("A", "{(0.9,0.1),(0.1,0.9)}", {"s1": [2, 2]})
    assert inferred is not None


def test_parse_gnn_matrix_string_variants() -> None:
    assert _parse_gnn_matrix_string("{(0.9,0.1),(0.1,0.9)}").shape == (2, 2)
    assert _parse_gnn_matrix_string("garbage").size > 0  # fallback
    assert _parse_vector_string("0.5,0.5").shape == (2,)


def test_router_general_end_to_end(tmp_path: Path) -> None:
    """Public router emits the general model script (create_params +
    run_simulation + __main__ guard). Regression note: the script has no
    ``def main`` — the entry point is the ``__main__`` guard."""
    from gnn.render.jax.jax_renderer import render_gnn_to_jax

    out = tmp_path / "model_router.py"
    success, message, paths = render_gnn_to_jax(_canonical_spec(), out)
    assert success, f"render failed: {message}"
    code = Path(paths[0]).read_text()
    assert "create_params" in code
    assert "run_simulation" in code
    assert 'if __name__ == "__main__"' in code
    compile(code, str(out), "exec")


def test_router_pomdp_end_to_end(tmp_path: Path) -> None:
    from gnn.render.jax.jax_renderer import render_gnn_to_jax_pomdp

    out = tmp_path / "model_router_pomdp.py"
    success, message, _ = render_gnn_to_jax_pomdp(_canonical_spec(), out)
    assert success, f"render failed: {message}"
    code = out.read_text()
    assert "solve_pomdp" in code
    compile(code, str(out), "exec")


def test_router_combined_end_to_end(tmp_path: Path) -> None:
    from gnn.render.jax.jax_renderer import render_gnn_to_jax_combined

    out = tmp_path / "model_router_combined.py"
    success, message, _ = render_gnn_to_jax_combined(_canonical_spec(), out)
    assert success, f"render failed: {message}"
    code = out.read_text()
    assert "__GNN_MODEL_NAME__" not in code  # token substituted
    compile(code, str(out), "exec")
