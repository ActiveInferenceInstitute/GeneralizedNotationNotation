"""Real-behavior tests for ``src/gnn/pipeline/pipeline_runtime_validator.py``.

The old import path ``gnn.pipeline.pipeline_validator`` is a deprecated
compatibility module; its re-export contract is pinned at the bottom of
this file.

The runtime integration tester is the only consumer chain behind
``gnn.pipeline.health_check`` (its constructor is invoked by the health
check's integration probe), so its pure decision logic is pinned here.
The subprocess-driven surfaces (``test_pipeline_execution``) are
pipeline-marked behavior exercised by the pipeline suite, not unit
tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gnn.pipeline.pipeline_runtime_validator import PipelineValidator

pytestmark = pytest.mark.filterwarnings("default")


def test_validator_constructs_without_subprocess() -> None:
    """Construction must succeed quietly and prepare state (health_check
    calls ``PipelineValidator(verbose=False)`` during integration probe).
    """
    validator = PipelineValidator(verbose=False)

    assert validator.validation_results == {}
    assert validator.dependency_manager is not None


def test_calculate_overall_health_transitions() -> None:
    """Empty -> unknown; all healthy -> healthy; any failed -> failed;
    a mix without failures -> degraded.
    """
    validator = PipelineValidator(verbose=False)

    assert validator._calculate_overall_health({}) == "unknown"
    assert (
        validator._calculate_overall_health(
            {"a": {"status": "healthy"}, "b": {"status": "healthy"}}
        )
        == "healthy"
    )
    assert (
        validator._calculate_overall_health(
            {"a": {"status": "healthy"}, "b": {"status": "failed"}}
        )
        == "failed"
    )
    assert (
        validator._calculate_overall_health(
            {"a": {"status": "healthy"}, "b": {"status": "degraded"}}
        )
        == "degraded"
    )


def test_probe_targets_exist_in_canonical_layout() -> None:
    """Retargeted probes must point at real renderer sources.

    Guards against reintroducing the dead pre-3.3.0 ``src/render/...``
    relative paths, which made every fix flag silently unfireable.
    """
    from gnn.pipeline import pipeline_runtime_validator as prv

    root = prv._PROBE_PACKAGE_ROOT
    assert (root / "render/pymdp/pymdp_renderer.py").is_file()
    assert (root / "render/jax/jax_model_generator.py").is_file()
    assert (root / "render/activeinference_jl/activeinference_renderer.py").is_file()
    assert (root / "utils/pipeline_dependencies.py").is_file()


def test_code_generation_fix_probes_validate_clean_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A clean probe tree flips every fix flag True without raising."""
    from gnn.pipeline import pipeline_runtime_validator as prv

    monkeypatch.setattr(prv, "_PROBE_PACKAGE_ROOT", tmp_path)
    (tmp_path / "render/pymdp").mkdir(parents=True)
    (tmp_path / "render/pymdp/pymdp_renderer.py").write_text("# clean\n")
    (tmp_path / "render/jax").mkdir()
    (tmp_path / "render/jax/jax_model_generator.py").write_text(
        "NUM_STATES = {num_states}\n"
    )
    (tmp_path / "render/activeinference_jl").mkdir()
    (tmp_path / "render/activeinference_jl/activeinference_renderer.py").write_text(
        "if isinstance(row, (tuple, list)):\n    pass\n"
    )
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils/pipeline_dependencies.py").write_text("")

    validator = PipelineValidator(verbose=False)
    fixes = validator.validate_code_generation_fixes()

    assert fixes == {
        "pymdp_import_fix": True,
        "jax_flax_fix": True,
        "julia_matrix_fix": True,
        "dependency_handling": True,
    }


def test_code_generation_fix_probes_fail_closed_on_bad_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing or regressed probe targets leave their flag False, never raise.

    A silently weakened validator must not be able to report success.
    """
    from gnn.pipeline import pipeline_runtime_validator as prv

    monkeypatch.setattr(prv, "_PROBE_PACKAGE_ROOT", tmp_path)
    (tmp_path / "render/pymdp").mkdir(parents=True)
    (tmp_path / "render/pymdp/pymdp_renderer.py").write_text(
        "from x import configure_from_gnn_spec\n"
    )
    # jax generator missing entirely.
    (tmp_path / "render/activeinference_jl").mkdir()
    (tmp_path / "render/activeinference_jl/activeinference_renderer.py").write_text(
        "x = 1\n"
    )
    # dependency manager missing entirely.

    validator = PipelineValidator(verbose=False)
    fixes = validator.validate_code_generation_fixes()

    assert fixes["pymdp_import_fix"] is False
    assert fixes["jax_flax_fix"] is False
    assert fixes["julia_matrix_fix"] is False
    assert fixes["dependency_handling"] is False


def test_old_import_path_warns_and_reexports() -> None:
    """The renamed module's old path must still import, warn, and bind the
    same objects (compatibility contract for external callers).
    """
    import importlib
    import sys
    import warnings

    # Force a fresh module execution so the DeprecationWarning fires even
    # if another test imported the compat path earlier in this session.
    sys.modules.pop("gnn.pipeline.pipeline_validator", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compat = importlib.import_module("gnn.pipeline.pipeline_validator")
    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "old import path must emit DeprecationWarning"
    from gnn.pipeline.pipeline_runtime_validator import (
        PipelineValidator as Canonical,
    )
    from gnn.pipeline.pipeline_runtime_validator import (
        main as canonical_main,
    )

    assert compat.PipelineValidator is Canonical
    assert compat.main is canonical_main
