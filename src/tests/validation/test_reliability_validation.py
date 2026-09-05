"""Current-run validation verdicts and replay-safe receipts."""

import json
from pathlib import Path
from typing import Any

import pytest

from validation import process_validation

VALID = {"ModelName": "M", "StateSpaceBlock": "s[2]\no[2]", "Connections": "s>o"}
INVALID = {"ModelName": "M", "Connections": "s>o"}


def write_input(
    base: Path, sections: dict[str, str], *, name: str = "model", parsed: bool = True
) -> Path:
    model = base / f"{name}.json"
    model.write_text(json.dumps({"raw_sections": sections}), encoding="utf-8")
    manifest = base / "3_gnn_output" / "gnn_processing_results.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "processed_files": [
                    {
                        "file_name": model.name,
                        "file_path": str(model),
                        "parsed_model_file": str(model),
                        "parse_success": parsed,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return base / "6_validation_output"


def receipt(output: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(
        (output / "validation_results.json").read_text()
    )
    return payload


def test_semantic_invalidity_controls_outcome(tmp_path: Path) -> None:
    output = write_input(tmp_path, INVALID)
    assert process_validation(tmp_path, output) is False
    result = receipt(output)
    assert result["files_validated"][0]["success"] is False
    assert result["files_validated"][0]["errors"]
    assert result["summary"]["failed_validations"] == 1


def test_prior_success_cannot_mask_current_failure(tmp_path: Path) -> None:
    output = write_input(tmp_path, VALID, name="first")
    assert process_validation(tmp_path, output) is True
    write_input(tmp_path, INVALID, name="second")
    assert process_validation(tmp_path, output) is False
    result = receipt(output)
    assert result["summary"]["successful_validations"] == 1
    assert result["current_summary"]["successful_validations"] == 0
    assert result["current_summary"]["failed_validations"] == 1


def test_repeated_input_is_idempotent_and_changed_input_replaces_receipt(
    tmp_path: Path,
) -> None:
    output = write_input(tmp_path, VALID)
    assert process_validation(tmp_path, output, run_id="run-a") is True
    first = receipt(output)["files_validated"][0]["receipt_key"]
    assert process_validation(tmp_path, output, run_id="run-a") is True
    result = receipt(output)
    assert result["summary"]["total_files"] == 1
    assert result["files_validated"][0]["receipt_key"] == first
    assert len(result["summary"]["validation_scores"]["semantic"]) == 1
    write_input(tmp_path, INVALID)
    assert process_validation(tmp_path, output, run_id="run-a") is False
    result = receipt(output)
    assert result["summary"]["total_files"] == 1
    assert result["summary"]["successful_validations"] == 0
    assert result["files_validated"][0]["receipt_key"] != first


@pytest.mark.parametrize(
    "changed", [{"run_id": "run-b"}, {"validation_level": "strict"}]
)
def test_run_or_configuration_change_drops_prior_scope(
    tmp_path: Path, changed: dict
) -> None:
    output = write_input(tmp_path, VALID, name="first")
    options: dict[str, Any] = {"run_id": "run-a", "validation_level": "standard"}
    assert process_validation(tmp_path, output, **options) is True
    write_input(tmp_path, VALID, name="second")
    options.update(changed)
    assert process_validation(tmp_path, output, **options) is True
    assert receipt(output)["summary"]["total_files"] == 1


def test_current_parse_failure_is_recorded_and_cannot_reuse_success(
    tmp_path: Path,
) -> None:
    output = write_input(tmp_path, VALID)
    assert process_validation(tmp_path, output) is True
    write_input(tmp_path, VALID, parsed=False)
    assert process_validation(tmp_path, output) is False
    result = receipt(output)
    assert result["summary"]["failed_validations"] == 1
    assert result["files_validated"][0]["success"] is False


def test_real_step3_manifest_timestamp_is_default_run_scope(tmp_path: Path) -> None:
    output = write_input(tmp_path, VALID, name="first")
    manifest = tmp_path / "3_gnn_output" / "gnn_processing_results.json"
    data = json.loads(manifest.read_text())
    data["timestamp"] = "2026-09-04T01:00:00"
    manifest.write_text(json.dumps(data))
    assert process_validation(tmp_path, output) is True
    write_input(tmp_path, VALID, name="second")
    data = json.loads(manifest.read_text())
    data["timestamp"] = "2026-09-04T02:00:00"
    manifest.write_text(json.dumps(data))
    assert process_validation(tmp_path, output) is True
    assert receipt(output)["summary"]["total_files"] == 1
    assert receipt(output)["files_validated"][0]["file_name"] == "second.json"


def test_mixed_current_pass_fails_and_empty_pass_cannot_reuse_success(
    tmp_path: Path,
) -> None:
    output = write_input(tmp_path, VALID, name="first")
    manifest = tmp_path / "3_gnn_output" / "gnn_processing_results.json"
    valid_entry = json.loads(manifest.read_text())["processed_files"][0]
    write_input(tmp_path, INVALID, name="second")
    data = json.loads(manifest.read_text())
    data["processed_files"].append(valid_entry)
    manifest.write_text(json.dumps(data))
    assert process_validation(tmp_path, output) is False
    assert receipt(output)["current_summary"]["failed_validations"] == 1
    manifest.write_text(json.dumps({"processed_files": []}))
    assert process_validation(tmp_path, output) is False
    assert receipt(output)["current_summary"]["total_files"] == 0
