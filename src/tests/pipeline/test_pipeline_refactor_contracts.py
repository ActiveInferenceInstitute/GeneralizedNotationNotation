#!/usr/bin/env python3
"""Behavioral contracts for the 2026-09-04 pipeline composability refactor.

Pins the shared building blocks introduced by the refactor and the additive
APIs: ``pipeline._io`` atomic writes, ``dag.find_circular_dependencies``,
``execution.resolve_step_numbers``, ``model_family_acceptance.select_model_families``,
registry-derived pipeline info, and the preflight ``pipeline.skip_steps`` gate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline import (
    _io,  # noqa: E402
    get_pipeline_info,  # noqa: E402
)
from pipeline._version import __version__  # noqa: E402
from pipeline.config import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGET_DIR,
    get_output_dir_for_script,
)
from pipeline.context import PipelineContext  # noqa: E402
from pipeline.dag import (  # noqa: E402
    find_circular_dependencies,
    resolve_execution_order,
)
from pipeline.execution import resolve_step_numbers  # noqa: E402
from pipeline.hasher import index_run  # noqa: E402
from pipeline.model_family_acceptance import (  # noqa: E402
    ModelFamily,
    select_model_families,
)

pytestmark = pytest.mark.pipeline


# ---------------------------------------------------------------------------
# pipeline._io — shared atomic write
# ---------------------------------------------------------------------------


def test_atomic_write_text_roundtrip_and_parents(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "dir" / "artifact.json"
    out = _io.atomic_write_text(dest, '{"k": 1}')
    assert out == dest
    assert dest.read_text() == '{"k": 1}'


def test_atomic_write_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "index.json"
    _io.atomic_write_text(dest, "original")

    real_replace = os.replace

    def _boom(src: object, dst: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="injected replace failure"):
        _io.atomic_write_text(dest, "corrupting")
    monkeypatch.setattr(os, "replace", real_replace)

    assert dest.read_text() == "original"
    # No temp residue may survive a failed write.
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# pipeline.dag — cycle detection and registry-derived default
# ---------------------------------------------------------------------------


def test_find_circular_dependencies_detects_two_cycle() -> None:
    assert find_circular_dependencies({0: [2], 2: [0], 1: []}) == {0, 2}


def test_find_circular_dependencies_detects_self_loop() -> None:
    assert find_circular_dependencies({0: [0]}) == {0}


def test_find_circular_dependencies_empty_when_acyclic() -> None:
    assert find_circular_dependencies({0: [1], 1: [2], 2: []}) == set()


def test_find_circular_dependencies_includes_downstream_of_cycle() -> None:
    # 3 depends on cycle members {0, 2}, so it can never be peeled either.
    assert find_circular_dependencies({0: [2], 2: [0], 3: [0]}) == {0, 2, 3}


def test_find_circular_dependencies_ignores_unknown_nodes() -> None:
    # Deps pointing outside the node universe are ignored (resolve semantics).
    assert find_circular_dependencies({0: [99], 1: [0]}) == set()


def test_resolve_execution_order_default_total_steps_from_registry() -> None:
    from pipeline.step_registry import STEPS

    tiers = resolve_execution_order({})
    assert len([s for tier in tiers for s in tier]) == len(STEPS)
    assert tiers[0] == list(range(len(STEPS)))


# ---------------------------------------------------------------------------
# pipeline.execution — public step-number resolution
# ---------------------------------------------------------------------------


def test_resolve_step_numbers_accepts_aliases_and_forms() -> None:
    assert resolve_step_numbers("all") == list(range(25))
    assert resolve_step_numbers(None) == list(range(25))
    assert resolve_step_numbers("11_render") == [11]
    assert resolve_step_numbers("11_render.py") == [11]
    assert resolve_step_numbers("11") == [11]
    assert resolve_step_numbers([3, "11_render", "3", "bogus"]) == [3, 11]


def test_resolve_step_numbers_falls_back_to_pipeline_data() -> None:
    assert resolve_step_numbers(None, {"only_steps": "3,5"}) == [3, 5]


# ---------------------------------------------------------------------------
# pipeline.model_family_acceptance — shared family selection
# ---------------------------------------------------------------------------


def _family(name: str) -> ModelFamily:
    return ModelFamily(
        name=name,
        description="d",
        target_dir=Path("input") / name,
        representative_files=("m.md",),
    )


def test_select_model_families_filters_in_manifest_order() -> None:
    families = [_family("a"), _family("b"), _family("c")]
    assert [f.name for f in select_model_families(families, ["c", "a"])] == [
        "a",
        "c",
    ]


def test_select_model_families_empty_request_selects_all() -> None:
    families = [_family("a"), _family("b")]
    assert select_model_families(families, None) == families
    assert select_model_families(families, []) == families


def test_select_model_families_strips_whitespace() -> None:
    families = [_family("a"), _family("b")]
    assert [f.name for f in select_model_families(families, [" a "])] == ["a"]


def test_select_model_families_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown model families: zz"):
        select_model_families([_family("a")], ["zz"])


# ---------------------------------------------------------------------------
# Version/registry consistency and defaults
# ---------------------------------------------------------------------------


def test_get_pipeline_info_version_matches_package() -> None:
    info = get_pipeline_info()
    assert info["version"] == __version__
    assert info["steps"] == list(range(25))


def test_context_defaults_use_shared_constants() -> None:
    ctx = PipelineContext()
    assert ctx.output_dir == Path(DEFAULT_OUTPUT_DIR)
    assert ctx.target_dir == Path(DEFAULT_TARGET_DIR)


def test_output_dir_for_script_contracts() -> None:
    base = Path("output")
    assert get_output_dir_for_script("3_gnn.py", base) == base / "3_gnn_output"
    assert get_output_dir_for_script("3_gnn", base) == base / "3_gnn_output"
    # Unknown steps fall back to '<stem>_output'.
    assert get_output_dir_for_script("99_unknown", base) == base / "99_unknown_output"
    # Nesting guard: base already at the step output dir is returned as-is.
    step_dir = base / "3_gnn_output"
    assert get_output_dir_for_script("3_gnn.py", step_dir) == step_dir


# ---------------------------------------------------------------------------
# hasher.index_run — return + durable-index contract
# ---------------------------------------------------------------------------


def test_index_run_returns_index_path_and_persists_entry(tmp_path: Path) -> None:
    summary_path = tmp_path / "run" / "pipeline_execution_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("{}", encoding="utf-8")

    out = index_run("deadbeef1234", summary_path, config={"only_steps": "3"})
    assert out == summary_path.parent / ".history" / "index.json"
    index = json.loads(out.read_text(encoding="utf-8"))
    assert index["deadbeef1234"]["config"] == {"only_steps": "3"}

    # Second call with the same hash updates the entry in place (not duplicate).
    index_run("deadbeef1234", summary_path, config={"only_steps": "5"})
    index = json.loads(out.read_text(encoding="utf-8"))
    assert index["deadbeef1234"]["config"] == {"only_steps": "5"}


# ---------------------------------------------------------------------------
# preflight.validate_config — pipeline.skip_steps gate
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, skip_steps: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"pipeline:\n  skip_steps: {json.dumps(skip_steps)}\n", encoding="utf-8"
    )
    return path


def test_validate_config_accepts_valid_skip_steps(tmp_path: Path) -> None:
    from pipeline import preflight

    report = preflight.validate_config(_write_config(tmp_path, [15, 16]))
    assert report.is_ok
    assert any("skip_steps" in issue.message for issue in report.issues) is False


def test_validate_config_rejects_out_of_range_skip_steps(tmp_path: Path) -> None:
    from pipeline import preflight

    report = preflight.validate_config(_write_config(tmp_path, [99]))
    assert not report.is_ok
    assert any("skip_steps" in issue.message for issue in report.issues)
