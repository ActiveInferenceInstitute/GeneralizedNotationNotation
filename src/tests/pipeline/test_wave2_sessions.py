"""Durable acceptance reuse must be bound to current inputs and artifacts."""

import json
from collections.abc import Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import NoReturn

import pytest

from pipeline.model_family_acceptance import Runner
from pipeline.run_session import (
    UnitStatus,
    WorkUnit,
    cancel_safe_cleanup,
    checkpoint,
    load_session,
    remaining_units,
    start_session,
)
from pipeline.session_acceptance import run_session_acceptance

from .test_session_acceptance import _passing_runner

AcceptanceFixture = tuple[Path, Path, Path, Runner, list[list[str]]]


@pytest.fixture
def acceptance(tmp_path: Path) -> AcceptanceFixture:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.md").write_text("# Model\n")
    manifest = tmp_path / "families.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gnn_model_family_manifest_v1",
                "families": [
                    {
                        "name": "family",
                        "description": "test",
                        "target_dir": str(source),
                        "representative_files": ["model.md"],
                        "frameworks": "pymdp",
                    }
                ],
            }
        )
    )
    calls: list[list[str]] = []

    def runner(command: Sequence[str]) -> CompletedProcess[str]:
        calls.append(list(command))
        return _passing_runner(command)

    return manifest, tmp_path / "output", tmp_path / "session.json", runner, calls


@pytest.mark.parametrize(
    "change", ["source", "framework", "profile", "output", "artifact", "unbound"]
)
def test_done_reuse_requires_current_evidence(
    acceptance: AcceptanceFixture, change: str
) -> None:
    manifest, output, session, runner, calls = acceptance
    first = run_session_acceptance(manifest, output, session, runner=runner)
    assert first["status"]["done"]
    assert run_session_acceptance(
        manifest, output, session, runner=runner, resume=True
    )["status"]["done"]
    assert len(calls) == 1
    data = json.loads(manifest.read_text())
    if change == "source":
        (Path(data["families"][0]["target_dir"]) / "model.md").write_text("# Changed\n")
    elif change in ("framework", "profile"):
        if change == "framework":
            data["families"][0]["frameworks"] = "jax"
        else:
            data["families"][0]["acceptance_profile"] = {"required_steps": [3, 5, 6]}
        manifest.write_text(json.dumps(data))
    elif change == "output":
        output = output.parent / "other-output"
    elif change == "artifact":
        next(output.rglob("gnn_processing_summary.json")).unlink()
    else:
        data = json.loads(session.read_text())
        for unit in data["units"]:
            unit.pop("input_identity", None)
            unit.pop("artifact_hashes", None)
        session.write_text(json.dumps(data))
    result = run_session_acceptance(
        manifest, output, session, runner=runner, resume=True
    )
    assert result["status"]["done"]
    assert len(calls) == 2


def test_running_checkpoint_survives_interrupt_and_resumes(
    acceptance: AcceptanceFixture,
) -> None:
    manifest, output, session, runner, calls = acceptance

    def interrupt(command: Sequence[str]) -> NoReturn:
        persisted = load_session(session)
        assert persisted.units[0].status == UnitStatus.RUNNING
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_session_acceptance(manifest, output, session, runner=interrupt)
    assert remaining_units(load_session(session)) == ["family"]
    assert run_session_acceptance(
        manifest, output, session, runner=runner, resume=True
    )["status"]["done"]
    assert len(calls) == 1


def test_running_unit_is_recoverable(tmp_path: Path) -> None:
    session = start_session("s", [WorkUnit(unit_id="a", status=UnitStatus.RUNNING)])
    checkpoint(session, tmp_path / "s.json")
    assert remaining_units(load_session(tmp_path / "s.json")) == ["a"]


@pytest.mark.parametrize("done_ref", ["shared.txt", ".", "done-directory"])
def test_cleanup_protects_done_artifacts_and_descendants(
    tmp_path: Path, done_ref: str
) -> None:
    shared = tmp_path / "shared.txt"
    if done_ref == "done-directory":
        shared = tmp_path / done_ref / "shared.txt"
        shared.parent.mkdir()
    shared.write_text("completed evidence")
    disposable = tmp_path / "partial.txt"
    disposable.write_text("partial")
    session = start_session(
        "s",
        [
            WorkUnit(unit_id="done", status=UnitStatus.DONE, artifact_refs=[done_ref]),
            WorkUnit(
                unit_id="failed",
                status=UnitStatus.FAILED,
                artifact_refs=[str(shared), str(disposable)],
            ),
        ],
    )
    cancel_safe_cleanup(session, tmp_path)
    assert shared.exists()
    assert disposable.exists() == (done_ref == ".")
    assert cancel_safe_cleanup(session, tmp_path) == []


@pytest.mark.parametrize("change", ["source", "profile"])
def test_changed_inputs_during_execution_cannot_be_marked_done(
    acceptance: AcceptanceFixture, change: str
) -> None:
    manifest, output, session, runner, _calls = acceptance

    def changing_runner(command: Sequence[str]) -> CompletedProcess[str]:
        result = runner(command)
        data = json.loads(manifest.read_text())
        if change == "source":
            (Path(data["families"][0]["target_dir"]) / "model.md").write_text(
                "changed during run"
            )
        else:
            data["families"][0]["frameworks"] = "jax"
            manifest.write_text(json.dumps(data))
        return result

    with pytest.raises(ValueError, match="changed during acceptance"):
        run_session_acceptance(manifest, output, session, runner=changing_runner)
    assert load_session(session).units[0].status == UnitStatus.FAILED


def test_done_reuse_checks_child_runtime_config(
    acceptance: AcceptanceFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline import hasher

    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("llm: {model: first}\n")
    monkeypatch.setattr(hasher, "RUNTIME_CONFIG_PATH", config_path, raising=False)
    manifest, output, session, runner, calls = acceptance
    run_session_acceptance(manifest, output, session, runner=runner)
    config_path.write_text("llm: {model: second}\n")
    run_session_acceptance(manifest, output, session, runner=runner, resume=True)
    assert len(calls) == 2
