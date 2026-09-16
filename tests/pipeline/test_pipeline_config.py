"""Real behavioral tests for the pipeline configuration contracts."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import pytest
import yaml

from gnn.pipeline.config import PipelineConfig, StepConfig, get_pipeline_config_dict


def test_step_config_defaults() -> None:
    cfg = StepConfig("3_gnn")
    assert cfg.step_name == "3_gnn"
    assert cfg.enabled is True
    assert cfg.timeout == 3600
    assert cfg.retries == 3
    assert cfg.required is True
    assert cfg.performance_tracking is True
    assert cfg.output_subdir == "3_gnn_output"


def test_step_config_custom_values() -> None:
    cfg = StepConfig(
        "3_gnn.py",
        enabled=False,
        timeout=120,
        retries=1,
        dependencies=["argparse"],
    )
    assert cfg.enabled is False
    assert cfg.timeout == 120
    assert cfg.retries == 1
    assert cfg.dependencies == ["argparse"]
    assert cfg.output_subdir == "3_gnn_output"


def test_pipeline_config_missing_file_uses_defaults(tmp_path: Path) -> None:
    cfg = PipelineConfig(tmp_path / "missing.yaml")
    assert cfg.config == {}


def test_pipeline_config_loads_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"steps": {"3_gnn": {"timeout": 99}}}))
    cfg = PipelineConfig(path)
    assert cfg.config["steps"]["3_gnn"]["timeout"] == 99


def test_pipeline_config_steps_property_returns_step_configs(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("steps:\n  3_gnn:\n    timeout: 60\n    retries: 2\n")
    cfg = PipelineConfig(path)
    steps = cfg.steps
    assert "3_gnn" in steps
    assert isinstance(steps["3_gnn"], StepConfig)
    assert steps["3_gnn"].timeout == 60
    assert steps["3_gnn"].retries == 2


def test_pipeline_config_steps_falls_back_to_registry(tmp_path: Path) -> None:
    # A non-dict "steps" value triggers the canonical-registry fallback.
    path = tmp_path / "config.yaml"
    path.write_text("steps: [3_gnn, 5_export]\n")
    cfg = PipelineConfig(path)
    steps = cfg.steps
    assert len(steps) > 0
    assert all(isinstance(v, StepConfig) for v in steps.values())


def test_get_step_config_returns_defaults(tmp_path: Path) -> None:
    cfg = PipelineConfig(tmp_path / "empty.yaml")
    sc = cfg.get_step_config("5_type_checker.py")
    assert sc.step_name == "5_type_checker.py"
    assert sc.timeout == 3600


def test_save_config_roundtrips_json(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    cfg = PipelineConfig(path)
    cfg.config = {"steps": {"3_gnn": {"timeout": 42}}}
    cfg.save_config()
    assert path.exists()
    reloaded = json.loads(path.read_text())
    assert reloaded["steps"]["3_gnn"]["timeout"] == 42


def test_get_pipeline_config_dict_returns_mapping() -> None:
    data = get_pipeline_config_dict()
    assert isinstance(data, dict)


def test_get_output_dir_for_script_accepts_py_suffix(tmp_path: Path) -> None:
    """Stem handling must cover ``.py`` names after dead-branch removal.

    ``Path(name).stem`` already strips the suffix, so the registry lookup
    resolves identically with or without it.
    """
    from gnn.pipeline.config import get_output_dir_for_script

    assert get_output_dir_for_script("7_export.py", tmp_path) == (
        tmp_path / "7_export_output"
    )
    assert get_output_dir_for_script("7_export", tmp_path) == (
        tmp_path / "7_export_output"
    )


def test_get_output_dir_for_script_warns_on_unregistered_stem(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Unregistered script names warn loudly, then keep the fallback path.

    A typo like ``5_typecheck.py`` must not silently make producer and
    consumer disagree about the output directory.
    """
    from gnn.pipeline.config import get_output_dir_for_script

    with caplog.at_level(logging.WARNING, logger="gnn.pipeline.config"):
        result = get_output_dir_for_script("5_typecheck.py", tmp_path)
    assert result == tmp_path / "5_typecheck_output"
    assert any(
        "5_typecheck.py" in rec.message and "Unregistered" in rec.message
        for rec in caplog.records
    )


def test_malformed_yaml_config_logs_error_and_degrades(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed YAML config must log at error level, not debug, and fall
    back to an empty settings dict."""
    path = tmp_path / "config.yaml"
    path.write_text("pipeline: [unclosed\n  bad: : yaml\n")
    with caplog.at_level(logging.ERROR, logger="gnn.pipeline.config"):
        cfg = PipelineConfig(path)
    assert cfg.config == {}
    assert any(
        "Could not parse config file" in rec.message and str(path) in rec.message
        for rec in caplog.records
        if rec.levelno == logging.ERROR
    )


def test_resolve_step_output_dir_from_base(tmp_path: Path) -> None:
    """Resolving from the pipeline base yields ``output/<stem>_output``."""
    from gnn.pipeline.config import resolve_step_output_dir

    assert resolve_step_output_dir("3_gnn", tmp_path) == tmp_path / "3_gnn_output"
    assert resolve_step_output_dir("12_execute", tmp_path) == (
        tmp_path / "12_execute_output"
    )


def test_resolve_step_output_dir_walks_up_from_sibling_step(tmp_path: Path) -> None:
    """A nested caller view inside another step's output resolves against the
    shared pipeline root, not the sibling step's directory."""
    from gnn.pipeline.config import resolve_step_output_dir

    nested = tmp_path / "16_analysis_output" / "results"
    nested.mkdir(parents=True)
    # Step 3 must resolve from the pipeline base (tmp_path), not from inside
    # the 16_analysis_output branch.
    assert resolve_step_output_dir("3_gnn", nested) == tmp_path / "3_gnn_output"
    # Step 12 likewise resolves from the base.
    assert resolve_step_output_dir("12_execute", nested) == (
        tmp_path / "12_execute_output"
    )


def test_resolve_step_output_dir_handles_deep_nesting(tmp_path: Path) -> None:
    """Nested ``*_output`` subdirectories walk up to the pipeline base."""
    from gnn.pipeline.config import resolve_step_output_dir

    nested = tmp_path / "16_analysis_output" / "summaries"
    nested.mkdir(parents=True)
    # Walks up through 16_analysis_output (ends with _output, parent is
    # tmp_path = the pipeline root) -> resolves from tmp_path.
    assert resolve_step_output_dir("3_gnn", nested) == tmp_path / "3_gnn_output"


def test_config_suffix_dispatch_yaml_yml_json(tmp_path: Path) -> None:
    """``.yaml``/``.yml`` route through PyYAML; ``.json`` through json.

    Identical payloads must load identically regardless of suffix.
    """
    payload = {"steps": {"3_gnn": {"timeout": 99}}}
    paths = {
        "yaml": tmp_path / "config.yaml",
        "yml": tmp_path / "config.yml",
        "json": tmp_path / "config.json",
    }
    paths["yaml"].write_text(yaml.safe_dump(payload))
    paths["yml"].write_text(yaml.safe_dump(payload))
    paths["json"].write_text(json.dumps(payload))

    assert PipelineConfig(paths["yaml"]).config == payload
    assert PipelineConfig(paths["yml"]).config == payload
    assert PipelineConfig(paths["json"]).config == payload


def test_broken_yaml_environment_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyYAML is a core dependency, so a broken/absent-yaml environment must
    surface a clear ImportError at import time — never an empty-config
    silent degradation of a user's YAML settings."""
    import gnn.pipeline.config as config_module

    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(ImportError):
        importlib.reload(config_module)
    # Restore the real yaml module, then reload to restore config_module.
    monkeypatch.undo()
    importlib.reload(config_module)


class TestResolveStepOutputDirCallSiteParity:
    """All call sites route through ``resolve_step_output_dir`` (N-1 seam).

    The analysis and gui wrappers are thin delegates: for the same caller
    view they must resolve identically to the shared helper — same shared
    pipeline root, no per-wrapper fallback logic.
    """

    @pytest.mark.parametrize(
        ("view_parts", "shared_root_parts"),
        [
            (("output",), ("output",)),
            (("output", "7_export_output"), ("output",)),
            (("output", "arbitrary_subdir"), ("output", "arbitrary_subdir")),
        ],
    )
    def test_both_call_sites_match_helper(
        self,
        tmp_path: Path,
        view_parts: tuple[str, ...],
        shared_root_parts: tuple[str, ...],
    ) -> None:
        from gnn.analysis.framework_common import resolve_execution_dir
        from gnn.gui.runner import resolve_output_root
        from gnn.pipeline.config import resolve_step_output_dir

        view = tmp_path.joinpath(*view_parts)
        shared_root = tmp_path.joinpath(*shared_root_parts)

        exec_dir = resolve_execution_dir(view)
        gui_root = resolve_output_root(view)

        # Each wrapper is exactly the shared helper with its own step stem.
        assert exec_dir == Path(resolve_step_output_dir("12_execute", view))
        assert gui_root == Path(resolve_step_output_dir("22_gui", view))
        # Same shared root through both call sites (identical resolution).
        assert exec_dir.parent == gui_root.parent == shared_root
