#!/usr/bin/env python3
"""
Tests for pipeline/pipeline_container_plan.py — hardened container plan for
running the GNN pipeline.

Real objects only: all tests use real Pydantic models, real YAML files (real
input/config.yaml plus real tmp_path YAML), real serialization, and the real
static security review. Includes a positive control (the generated plan reviews
clean), command-shape assertions (src/main.py + dirs + skip-steps propagation),
rollback semantics, and a negative control (a manually-degraded privileged spec
forces a CRITICAL — proving the review is wired and has teeth).
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REAL_CONFIG = (
    Path(__file__).parent.parent.parent.parent / "input" / "config.yaml"
)


def _write_config(tmp_path: Path, skip_steps: list[int]) -> Path:
    """Write a minimal real YAML config with the given skip_steps."""
    import yaml

    cfg = {"pipeline": {"enabled": True, "steps": [], "skip_steps": skip_steps}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


class TestPlanIsClean:
    def test_real_config_plan_reviews_empty(self) -> Any:
        """POSITIVE CONTROL: generated plan from the real config is hardened."""
        from pipeline.pipeline_container_plan import (
            plan_for_pipeline,
            review_pipeline_plan,
        )

        assert REAL_CONFIG.exists(), f"missing real config: {REAL_CONFIG}"
        plan = plan_for_pipeline(REAL_CONFIG)
        findings = review_pipeline_plan(plan)
        assert findings == [], f"expected clean plan, got: {findings}"

    def test_plan_with_skip_steps_still_clean(self, tmp_path: Path) -> Any:
        from pipeline.pipeline_container_plan import (
            plan_for_pipeline,
            review_pipeline_plan,
        )

        cfg = _write_config(tmp_path, [13, 15])
        plan = plan_for_pipeline(cfg)
        assert review_pipeline_plan(plan) == []

    def test_hardened_defaults_present(self, tmp_path: Path) -> Any:
        from pipeline.pipeline_container_plan import plan_for_pipeline

        cfg = _write_config(tmp_path, [])
        plan = plan_for_pipeline(cfg)
        spec = plan.specs[0]
        assert spec.name == "gnn-pipeline"
        assert spec.user == "nonroot"
        assert spec.read_only_rootfs is True
        assert spec.privileged is False
        assert spec.cap_drop == ["ALL"]
        assert spec.cap_add == []
        assert spec.network == "" and spec.pid == "" and spec.ipc == ""
        # The output mount must be a named volume, never a host path.
        assert spec.mounts and not spec.mounts[0].startswith("/")


class TestCommandShape:
    def test_command_references_main_and_dirs(self, tmp_path: Path) -> Any:
        from pipeline.pipeline_container_plan import plan_for_pipeline

        cfg = _write_config(tmp_path, [])
        plan = plan_for_pipeline(
            cfg, target_dir="input/gnn_files", output_dir="output"
        )
        command = plan.specs[0].command
        assert "src/main.py" in command
        i = command.index("--target-dir")
        assert command[i + 1] == "input/gnn_files"
        j = command.index("--output-dir")
        assert command[j + 1] == "output"
        # No skip_steps in config -> no --skip-steps flag.
        assert "--skip-steps" not in command

    def test_custom_dirs_propagate(self, tmp_path: Path) -> Any:
        from pipeline.pipeline_container_plan import plan_for_pipeline

        cfg = _write_config(tmp_path, [])
        plan = plan_for_pipeline(cfg, target_dir="my/in", output_dir="my/out")
        command = plan.specs[0].command
        assert command[command.index("--target-dir") + 1] == "my/in"
        assert command[command.index("--output-dir") + 1] == "my/out"

    def test_skip_steps_propagate_into_command(self, tmp_path: Path) -> Any:
        from pipeline.pipeline_container_plan import plan_for_pipeline

        cfg = _write_config(tmp_path, [15, 16])
        plan = plan_for_pipeline(cfg)
        command = plan.specs[0].command
        assert "--skip-steps" in command
        assert command[command.index("--skip-steps") + 1] == "15,16"


class TestRollback:
    def test_rollback_bumps_version_and_attaches_descriptor(
        self, tmp_path: Path
    ) -> Any:
        from pipeline.pipeline_container_plan import plan_for_pipeline

        cfg = _write_config(tmp_path, [])
        v1 = plan_for_pipeline(cfg)
        assert v1.version == 1
        assert v1.rollback is None

        v2 = plan_for_pipeline(cfg, previous=v1)
        assert v2.version == 2
        assert v2.rollback is not None
        assert v2.rollback.previous_version == 1
        assert v2.rollback.previous_plan_hash == v1.plan_hash


class TestNegativeControl:
    def test_privileged_spec_yields_critical(self, tmp_path: Path) -> Any:
        """NEGATIVE CONTROL: a degraded (privileged) spec must FIRE a CRITICAL."""
        from pipeline.pipeline_container_plan import (
            plan_for_pipeline,
            review_pipeline_plan,
        )

        cfg = _write_config(tmp_path, [])
        plan = plan_for_pipeline(cfg)
        # Manually degrade the otherwise-clean plan.
        plan.specs[0].privileged = True

        findings = review_pipeline_plan(plan)
        crit = [f for f in findings if f.severity == "CRITICAL"]
        assert crit, f"expected a CRITICAL finding, got: {findings}"
        assert any(f.code == "PRIVILEGED_CONTAINER" for f in crit)

    def test_host_mount_spec_yields_critical(self, tmp_path: Path) -> Any:
        """A degraded host-path mount must also FIRE a CRITICAL."""
        from pipeline.pipeline_container_plan import (
            plan_for_pipeline,
            review_pipeline_plan,
        )

        cfg = _write_config(tmp_path, [])
        plan = plan_for_pipeline(cfg)
        plan.specs[0].mounts = ["/var/run/docker.sock:/var/run/docker.sock"]

        findings = review_pipeline_plan(plan)
        assert any(
            f.severity == "CRITICAL" and f.code == "SENSITIVE_HOST_MOUNT"
            for f in findings
        ), f"expected a SENSITIVE_HOST_MOUNT CRITICAL, got: {findings}"


class TestSkipStepsValidation:
    def test_valid_skip_steps_dedup_and_sorted(self, tmp_path: Path) -> Any:
        from pipeline.pipeline_container_plan import read_skip_steps

        cfg = _write_config(tmp_path, [13, 2, 13])
        assert read_skip_steps(cfg) == [2, 13]

    def test_rejects_float_negative_and_out_of_range(self, tmp_path: Path) -> Any:
        """NEGATIVE: a float (15.9), a negative (-3), or an out-of-range (99) step
        must raise rather than silently truncate/accept and mis-target the skip set."""
        import pytest

        from pipeline.pipeline_container_plan import read_skip_steps

        for bad in ([15.9], [-3], [99], ["abc"]):
            cfg = _write_config(tmp_path, bad)
            with pytest.raises(ValueError):
                read_skip_steps(cfg)
