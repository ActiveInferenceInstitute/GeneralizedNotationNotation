"""Stable content identity and public reproduction guards, without pipeline runs."""

import json
from pathlib import Path
from typing import NoReturn

import pytest

import gnn.cli as cli
import gnn.main as orchestrator
from gnn.pipeline.hasher import compute_run_hash, index_run
from gnn.utils.arguments.pipeline_arguments import PipelineArguments


def test_hash_binds_relative_path_and_is_relocation_stable(tmp_path: Path) -> None:
    source = tmp_path / "first"
    (source / "a").mkdir(parents=True)
    model = source / "a" / "model.md"
    model.write_text("model")
    before = compute_run_hash(source)
    model.parent.rename(source / "b")
    assert compute_run_hash(source) != before
    after = compute_run_hash(source)
    source.rename(tmp_path / "second")
    assert compute_run_hash(tmp_path / "second") == after


def test_summary_hash_binds_effective_args_and_steps(tmp_path: Path) -> None:
    (tmp_path / "model.md").write_text("model")
    args = PipelineArguments(
        target_dir=tmp_path, output_dir=tmp_path / "out", frameworks="pymdp"
    )
    first = orchestrator._initialize_pipeline_summary(
        args, [("11_render.py", "Render")], {}
    )
    args.frameworks = "jax"
    second = orchestrator._initialize_pipeline_summary(
        args, [("11_render.py", "Render")], {}
    )
    assert second["run_hash"] != first["run_hash"]
    third = orchestrator._initialize_pipeline_summary(
        args, [("12_execute.py", "Execute")], {}
    )
    assert third["run_hash"] != second["run_hash"]
    args.output_dir = tmp_path / "different-output"
    args.verbose = True
    fourth = orchestrator._initialize_pipeline_summary(
        args, [("12_execute.py", "Execute")], {}
    )
    assert fourth["run_hash"] == third["run_hash"]


@pytest.mark.parametrize(
    "change",
    ["content", "missing", "extra", "unbound", "config", "selection", "unchanged"],
)
def test_reproduce_checks_saved_identity_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    model = source / "model.md"
    model.write_text("model")
    args = PipelineArguments(
        target_dir=source, output_dir=tmp_path / "output", only_steps="0"
    )
    summary = orchestrator._initialize_pipeline_summary(
        args, [("0_template.py", "Template")], {}
    )
    history = tmp_path / "history"
    index_run(
        summary["run_hash"],
        tmp_path / "summary.json",
        history_dir=history,
        config={
            "args": args.to_dict(),
            "pipeline": {},
            "identity_config": summary.get("identity_config"),
            "run_hash_schema": summary.get("run_hash_schema"),
        },
        file_hashes=summary["file_hashes"],
    )
    if change == "content":
        model.write_text("changed")
    elif change == "missing":
        model.unlink()
    elif change == "extra":
        (source / "other.md").write_text("extra")
    elif change in ("unbound", "config", "selection"):
        path = history / "index.json"
        data = json.loads(path.read_text())
        if change == "unbound":
            data[summary["run_hash"]].pop("file_hashes")
        elif change == "selection":
            data[summary["run_hash"]]["config"]["args"]["only_steps"] = "1"
        else:
            data[summary["run_hash"]]["config"]["args"]["frameworks"] = "jax"
        path.write_text(json.dumps(data))
    dispatch_calls: list[int] = []

    def dispatch(*args: object, **kwargs: object) -> int:
        dispatch_calls.append(1)
        return 0

    monkeypatch.setattr(orchestrator, "main", dispatch)
    result = cli.main(["reproduce", summary["run_hash"], "--history-dir", str(history)])
    assert result == (0 if change == "unchanged" else 1)
    assert len(dispatch_calls) == int(change == "unchanged")


def test_step_uses_recorded_config_instead_of_rereading_live_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    def unexpected_read(_config_path: Path) -> NoReturn:
        raise AssertionError("Live configuration must not replace the run snapshot")

    monkeypatch.setattr(orchestrator, "_read_input_config", unexpected_read)
    args = PipelineArguments(target_dir=tmp_path, output_dir=tmp_path / "output")
    result = orchestrator.execute_pipeline_step(
        "0_template.py",
        args,
        logging.getLogger(__name__),
        pipeline_config={
            "testing_matrix": {"enabled": True, "global_steps": {"0_template": False}}
        },
    )
    assert result["status"] == "SKIPPED"
    assert "disabled" in result["stdout"].lower() or "global_steps" in result["stdout"]


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml", ".txt"])
def test_all_discovered_model_formats_participate_in_hash(
    tmp_path: Path, suffix: str
) -> None:
    model = tmp_path / f"model{suffix}"
    model.write_text("original")
    before = compute_run_hash(tmp_path)
    model.write_text("changed")
    assert compute_run_hash(tmp_path) != before


def test_reproduce_rejects_live_child_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.pipeline import hasher

    config_path = tmp_path / "runtime.yaml"
    config_path.write_text("llm: {model: first}\n")
    monkeypatch.setattr(hasher, "RUNTIME_CONFIG_PATH", config_path, raising=False)
    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_text("model")
    args = PipelineArguments(
        target_dir=source, output_dir=tmp_path / "output", only_steps="0"
    )
    summary = orchestrator._initialize_pipeline_summary(
        args, [("0_template.py", "Template")], {}
    )
    history = tmp_path / "history"
    index_run(
        summary["run_hash"],
        tmp_path / "summary.json",
        history_dir=history,
        config={
            "args": args.to_dict(),
            "pipeline": {},
            "identity_config": summary["identity_config"],
            "run_hash_schema": summary["run_hash_schema"],
        },
        file_hashes=summary["file_hashes"],
    )
    config_path.write_text("llm: {model: second}\n")
    dispatch_calls: list[int] = []

    def dispatch(*args: object, **kwargs: object) -> int:
        dispatch_calls.append(1)
        return 0

    monkeypatch.setattr(orchestrator, "main", dispatch)
    assert (
        cli.main(["reproduce", summary["run_hash"], "--history-dir", str(history)]) == 1
    )
    assert not dispatch_calls


def test_hash_uses_registered_formats_and_excludes_discovery_metadata(
    tmp_path: Path,
) -> None:
    from gnn.parsers.common import get_supported_gnn_extensions

    for index, suffix in enumerate(get_supported_gnn_extensions()):
        model = tmp_path / f"model{index}{suffix}"
        before = compute_run_hash(tmp_path)
        model.write_bytes(b"source bytes")
        assert compute_run_hash(tmp_path) != before, suffix
    before = compute_run_hash(tmp_path)
    (tmp_path / "README.md").write_text("repository documentation")
    assert compute_run_hash(tmp_path) == before


def test_changed_source_before_publication_gets_failed_receipt(tmp_path: Path) -> None:
    import logging

    from gnn.api.pipeline_runner import PIPELINE_SUMMARY

    source = tmp_path / "input"
    source.mkdir()
    model = source / "model.md"
    model.write_text("original")
    args = PipelineArguments(target_dir=source, output_dir=tmp_path / "output")
    summary = orchestrator._initialize_pipeline_summary(args, [], {})
    model.write_text("changed")
    orchestrator._write_pipeline_summary_outputs(
        args, {}, summary, logging.getLogger(__name__)
    )
    receipt = json.loads((args.output_dir / PIPELINE_SUMMARY).read_text())
    assert receipt["overall_status"] == "FAILED"
    assert "identity" in receipt["error"].lower()
