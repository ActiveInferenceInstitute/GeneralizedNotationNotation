"""Container review and export preserve equivalent identities and plan semantics."""

from pathlib import Path

import pytest

from pipeline.container_plan import (
    ContainerPlan,
    generate_container_plan,
    plan_to_compose,
    security_review,
)
from pipeline.pipeline_container_plan import plan_for_pipeline


def plan(**overrides: object) -> ContainerPlan:
    return generate_container_plan(
        "p", [{"name": "worker", "image": "image@sha256:" + "a" * 64, **overrides}]
    )


@pytest.mark.parametrize("user", ["0", "0:1000", "root:1000", "0000", " root "])
def test_root_identity_forms_are_reported(user: str) -> None:
    assert "ROOT_USER" in {f.code for f in security_review(plan(user=user))}


@pytest.mark.parametrize(
    "source",
    ["/etc/", "/etc/passwd", "/safe/../etc", "/proc/1", "//var/run/docker.sock"],
)
def test_sensitive_mount_aliases_and_children_are_reported(source: str) -> None:
    assert "SENSITIVE_HOST_MOUNT" in {
        f.code for f in security_review(plan(mounts=[f"{source}:/data:ro"]))
    }


def test_benign_mounts_and_nonroot_uid_remain_clean() -> None:
    assert (
        security_review(
            plan(
                user="1000:0",
                mounts=["data:/data", "/workspace:/workspace", "/etcetera:/data"],
            )
        )
        == []
    )


def test_compose_retains_reviewed_fields() -> None:
    value = plan(cap_add=["NET_ADMIN"], network="host", pid="host", ipc="host")
    service = plan_to_compose(value)["services"]["worker"]
    assert service["cap_add"] == ["NET_ADMIN"]
    assert service["network_mode"] == "host"
    assert service["pid"] == "host"
    assert service["ipc"] == "host"


@pytest.mark.parametrize("output", ["output", "nested/output", "/results"])
def test_pipeline_output_mount_matches_command(tmp_path: Path, output: str) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("pipeline: {}\n")
    value = plan_for_pipeline(config, output_dir=output)
    spec = value.specs[0]
    command_output = spec.command[spec.command.index("--output-dir") + 1]
    expected = (
        command_output if command_output.startswith("/") else f"/app/{command_output}"
    )
    assert spec.mounts == [f"gnn-output:{expected}"]
    assert "gnn-output" in plan_to_compose(value)["volumes"]
