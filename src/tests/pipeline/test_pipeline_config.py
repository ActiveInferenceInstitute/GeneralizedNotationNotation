"""Real behavioral tests for the pipeline configuration contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import PipelineConfig, StepConfig, get_pipeline_config_dict


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
