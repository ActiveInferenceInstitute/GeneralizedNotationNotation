"""Execution receipt freshness, idempotence, and malformed-root regressions."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gnn.execute.metadata import (
    _load_render_summary_contract,
    _merge_prior_execution_summary,
)

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize("root", [None, [], "invalid", 1])
def test_render_contract_rejects_invalid_root(tmp_path: Path, root: object) -> None:
    (tmp_path / "render_processing_summary.json").write_text(json.dumps(root))
    assert _load_render_summary_contract(tmp_path, ["pymdp"], LOGGER) == (None, [])


def receipt(source: Path, script: Path, run_id: str, *, success: bool = True) -> dict:
    return {
        "run_id": run_id,
        "configuration": {"frameworks": ["pymdp"]},
        "target_directory": str(source),
        "status": "success" if success else "failed",
        "success": success,
        "exit_code": 0 if success else 1,
        "execution_details": [
            {"script_path": str(script), "framework": "pymdp", "success": success}
        ],
        "successful_executions": int(success),
        "failed_executions": int(not success),
    }


def test_execution_retry_replaces_stale_verdict_and_new_run_is_isolated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "model.md").write_text("model")
    first_script = tmp_path / "first.py"
    first_script.write_text("print(1)")
    second_script = tmp_path / "second.py"
    second_script.write_text("print(2)")
    path = tmp_path / "execution_summary.json"
    prior = receipt(source, first_script, "run-a", success=False)
    _merge_prior_execution_summary(prior, path, LOGGER)
    path.write_text(json.dumps(prior))
    current = receipt(source, second_script, "run-a")
    _merge_prior_execution_summary(current, path, LOGGER)
    assert current["status"] == "success"
    assert current["failed_executions"] == 0
    assert current["total_scripts_found"] == 1
    assert current["execution_details"][0]["script_identity"]["sha256"]
    path.write_text(json.dumps(current))
    next_run = receipt(source, first_script, "run-b")
    _merge_prior_execution_summary(next_run, path, LOGGER)
    assert len(next_run["execution_details"]) == 1
    assert next_run["receipt_identity"]["run_id"] == "run-b"
    assert list((tmp_path / "history").glob("execution-*.json"))


@pytest.mark.parametrize("changed", ["source", "artifact"])
def test_render_contract_rejects_changed_bytes(tmp_path: Path, changed: str) -> None:
    import hashlib

    source = tmp_path / "model.md"
    script = tmp_path / "model.py"
    source.write_text("original source")
    script.write_text("print(1)")

    def identify(path: Path) -> dict[str, str]:
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    summary = {
        "receipt_identity": {"run_id": "r", "config_sha256": "c"},
        "file_results": {
            str(source): {
                "source_identity": identify(source),
                "framework_results": {
                    "pymdp": {
                        "success": True,
                        "output_files": [str(script)],
                        "artifact_identities": [identify(script)],
                    }
                },
            }
        },
    }
    (tmp_path / "render_processing_summary.json").write_text(json.dumps(summary))
    (source if changed == "source" else script).write_text("changed")
    allowed, failures = _load_render_summary_contract(tmp_path, ["pymdp"], LOGGER)
    assert allowed == set()
    assert failures


@pytest.mark.parametrize("kind", ["render", "execute"])
def test_atomic_receipt_serialization_failure_preserves_previous(
    tmp_path: Path, kind: str
) -> None:
    from gnn.execute.metadata import _atomic_execution_json
    from gnn.render.processor import _atomic_render_json

    path = tmp_path / "receipt.json"
    path.write_text('{"prior": true}')
    payload: dict[str, Any] = {}
    payload["cycle"] = payload
    writer = _atomic_execution_json if kind == "execute" else _atomic_render_json
    with pytest.raises(ValueError, match="Circular"):
        writer(path, payload)
    assert json.loads(path.read_text()) == {"prior": True}
    assert list(tmp_path.iterdir()) == [path]


def test_execution_same_run_combines_scopes_but_rejects_changed_config(
    tmp_path: Path,
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    first_script = a / "first.py"
    first_script.write_text("print(1)")
    second_script = b / "second.py"
    second_script.write_text("print(2)")
    path = tmp_path / "execution_summary.json"
    prior = receipt(a, first_script, "run-a")
    prior["execution_details"][0].update(success=False, skipped=True)
    prior.update(status="skipped", success=False, exit_code=2)
    _merge_prior_execution_summary(prior, path, LOGGER)
    path.write_text(json.dumps(prior))
    current = receipt(b, second_script, "run-a")
    _merge_prior_execution_summary(current, path, LOGGER)
    assert current["total_scripts_found"] == 2
    assert current["status"] == "success_with_skips"
    assert current["success"] is True
    path.write_text(json.dumps(current))
    changed = receipt(b, second_script, "run-a")
    changed["configuration"]["strict"] = True
    _merge_prior_execution_summary(changed, path, LOGGER)
    assert changed["total_scripts_found"] == 1
    assert changed["status"] == "success"


def test_execution_atomic_writer_emits_current_receipt(tmp_path: Path) -> None:
    from gnn.execute.processor import _write_execution_summaries

    source = tmp_path / "input"
    source.mkdir()
    script = source / "demo.py"
    script.write_text("print(1)")
    result = receipt(source, script, "run-a")
    result.update(timestamp="2026-09-04", output_directory=str(tmp_path))
    result["execution_details"][0].update(
        script_name="demo.py", executor="python", stdout="full text"
    )
    _write_execution_summaries(tmp_path, result, True, LOGGER)
    slim = json.loads((tmp_path / "summaries/execution_summary.json").read_text())
    detail = json.loads(
        (tmp_path / "summaries/execution_summary_detail.json").read_text()
    )
    assert slim["receipt_identity"]["run_id"] == "run-a"
    assert slim["execution_details"][0]["stdout_length"] == 9
    assert "stdout" not in slim["execution_details"][0]
    assert detail["execution_details"][0]["stdout"] == "full text"
    assert result["execution_details"][0]["stdout"] == "full text"
