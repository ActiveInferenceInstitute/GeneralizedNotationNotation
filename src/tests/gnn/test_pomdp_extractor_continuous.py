"""Continuous-state exemplars: extraction, model kind, and framework support matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor, pomdp_to_gnn_spec
from render.processor import process_render

REPO = Path(__file__).resolve().parents[3]
CONTINUOUS_DIR = REPO / "input" / "gnn_files" / "continuous"
FILES = sorted(CONTINUOUS_DIR.glob("*.md"))

UNSUPPORTED = {"pymdp", "activeinference_jl", "bnlearn", "discopy"}
SUPPORTED = {"jax", "numpyro", "pytorch", "stan", "rxinfer"}


@pytest.mark.parametrize("path", FILES, ids=[p.stem for p in FILES])
def test_continuous_exemplar_extracts_lgssm(path: Path) -> None:
    pomdp = extract_pomdp_from_file(path, strict_validation=True)
    assert pomdp is not None
    assert pomdp.model_kind == "continuous"
    assert pomdp.A_matrix is None and pomdp.B_matrix is None
    assert pomdp.matrices is not None
    for key in ("F", "H", "Q", "R", "prior_mean", "prior_cov"):
        assert key in pomdp.matrices, key
    assert pomdp.num_states == len(pomdp.matrices["F"])
    assert pomdp.num_observations == len(pomdp.matrices["H"])
    spec = pomdp_to_gnn_spec(pomdp)
    assert spec["model_kind"] == "continuous"
    assert "A" not in spec["initialparameterization"]


def test_navigation_is_closed_loop_others_passive() -> None:
    kinds: dict[str, bool] = {}
    for p in FILES:
        pomdp = extract_pomdp_from_file(p, strict_validation=True)
        assert pomdp is not None
        kinds[p.stem] = pomdp.passive_model
    assert kinds["continuous_navigation"] is False
    assert kinds["predictive_coding_agent"] is True
    assert kinds["stochastic_dynamics"] is True


def test_unsupported_frameworks_are_flagged_not_failed(tmp_path: Path) -> None:
    pomdp = extract_pomdp_from_file(FILES[0], strict_validation=True)
    assert pomdp is not None
    proc = POMDPRenderProcessor(tmp_path)
    for fw in sorted(UNSUPPORTED):
        result = proc._process_single_framework(pomdp, fw)
        assert result["unsupported"] is True and result["status"] == "unsupported"
        assert "supports discrete POMDPs only" in result["message"]
        assert result["output_files"] == []


def test_process_render_counts_unsupported_separately(tmp_path: Path) -> None:
    import json

    outcome = process_render(
        target_dir=CONTINUOUS_DIR,
        output_dir=tmp_path,
        frameworks=[
            "jax",
            "numpyro",
            "pytorch",
            "stan",
            "rxinfer",
            "discopy",
            "pymdp",
            "activeinference_jl",
        ],
        verbose=False,
    )
    assert outcome is True
    summary = json.loads((tmp_path / "render_processing_summary.json").read_text())
    assert summary["total_files"] == len(FILES)
    assert summary["successful_files"] == len(FILES)
    assert summary["failed_framework_renderings"] == []
    unsupported = {
        (u["framework"]) for u in summary["unsupported_framework_renderings"]
    }
    assert unsupported == {"pymdp", "activeinference_jl", "discopy"}
    for res in summary["file_results"].values():
        statuses = {
            fw: r.get("status", "ok") for fw, r in res["framework_results"].items()
        }
        assert all(
            r["success"]
            for fw, r in res["framework_results"].items()
            if fw in SUPPORTED
        )
        assert statuses["pymdp"] == "unsupported"
