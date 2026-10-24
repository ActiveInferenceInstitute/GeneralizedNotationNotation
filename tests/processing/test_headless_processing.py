"""Headless consumer failures, dependency isolation and actual profiling outputs."""

import json
import pstats
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gnn.processing import (
    parse_gnn_file,
    process_gnn_directory,
    process_gnn_directory_lightweight,
)
from gnn.processing.profiling import profile_directory


@pytest.fixture
def models(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    path.mkdir()
    (path / "model.md").write_text(
        "## ModelName\nDemo\n## StateSpaceBlock\nx: float\nx = 1\n", encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("parallel", [False, True])
def test_file_failure_keeps_successful_siblings(
    models: Path, tmp_path: Path, parallel: bool
) -> None:
    bad = models / "bad.md"
    bad.write_bytes(b"\xff\xfe")
    result = process_gnn_directory_lightweight(models, parallel=parallel)
    assert result["success"] is False
    assert (
        result["files_found"],
        result["files_processed"],
        result["files_failed"],
    ) == (2, 1, 1)
    assert result["errors"][0]["error_type"] == "UnicodeDecodeError"
    output = tmp_path / "results"
    public = process_gnn_directory(models, output, parallel=parallel)
    assert public["status"] == "FAILED"
    assert set(public["files"]) == {str(models / "model.md"), str(bad)}
    assert public["processed_files"] == [str(models / "model.md")]
    assert "utf-8" in public["error"]
    assert json.loads((output / "gnn_processing_results.json").read_text()) == public


def test_invalid_structure_is_not_success(models: Path) -> None:
    empty = models / "empty.md"
    empty.write_text("")
    result = process_gnn_directory(models)
    assert result["status"] == "FAILED"
    assert str(empty) in result["files"] and str(empty) not in result["processed_files"]
    assert result["errors"][0]["error_type"] == "ValidationError"
    assert "File is empty" in result["error"]


def test_single_source_empty_selection_and_excluded_file(models: Path) -> None:
    result = process_gnn_directory(models / "model.md")
    assert result["status"] == "SUCCESS"
    assert result["processed_files"] == [str(models / "model.md")]
    (models / "model.md").unlink()
    assert process_gnn_directory(models) == {
        "status": "SUCCESS",
        "files": [],
        "processed_files": [],
    }
    readme = models / "README.md"
    readme.write_text("Only documentation")
    assert process_gnn_directory(readme)["status"] == "FAILED"


def test_missing_input_and_output_failure_are_explicit(
    models: Path, tmp_path: Path
) -> None:
    assert process_gnn_directory_lightweight(tmp_path / "missing")["success"] is False
    output = tmp_path / "not_a_directory"
    output.write_text("preserve this")
    result = process_gnn_directory(models, output)
    assert result["status"] == "ERROR" and result["error_type"] == "FileExistsError"
    assert result["processed_files"] == [str(models / "model.md")]
    assert output.read_text() == "preserve this"


def test_concurrent_reports_publish_complete_json(models: Path, tmp_path: Path) -> None:
    output = tmp_path / "results"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: process_gnn_directory(models, output), range(8))
        )
    assert all(result["status"] == "SUCCESS" for result in results)
    assert (
        json.loads((output / "gnn_processing_results.json").read_text()) == results[0]
    )
    assert [p.name for p in output.iterdir()] == ["gnn_processing_results.json"]


def test_lightweight_public_imports_do_not_load_heavy_owners() -> None:
    code = """
import builtins, json, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'psutil', 'yaml', 'matplotlib', 'jax'}:
        raise AssertionError('unexpected heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from gnn.processing import parse_gnn_file, check_gnn_file_structure
assert parse_gnn_file('inline.md', content='## Demo x = 1')['success']
assert check_gnn_file_structure('inline.md', content='## Demo x = 1')['valid']
assert 'gnn.processing.core_processor' not in sys.modules
print(json.dumps({'status': 'success'}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "success"}


def test_agent_cli_reports_one_json_failure(models: Path) -> None:
    (models / "bad.md").write_bytes(b"\xff")
    result = subprocess.run(
        [sys.executable, "-m", "gnn.processing", "--target-dir", str(models)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["status"] == "FAILED"
    assert report["processed_files"] == [str(models / "model.md")]
    assert report["errors"][0]["error_type"] == "UnicodeDecodeError"


def test_profile_retains_distinct_unprofiled_and_traced_work(
    models: Path, tmp_path: Path
) -> None:
    output = tmp_path / "profile"
    result = profile_directory(models, output, repeats=2)
    assert result["status"] == "success"
    assert len(result["unprofiled_samples"]) == 2
    assert all(
        s["wall_seconds"] >= 0 and s["cpu_seconds"] >= 0
        for s in result["unprofiled_samples"]
    )
    assert result["input_bookends_equal"] and result["backend_source_bookends_equal"]
    assert result["trace"]["included_in_timing_samples"] is False
    assert (output / "trace.pstats").stat().st_size > 0
    assert "function calls" in (output / "trace.txt").read_text()
    statistics = pstats.Stats(str(output / "trace.pstats"))
    assert any(key[2] == "_extract_variables_lightweight" for key in statistics.stats)
    assert json.loads((output / "profile_report.json").read_text()) == result
    with pytest.raises(FileExistsError):
        profile_directory(models, output)
    assert json.loads((output / "profile_report.json").read_text()) == result


@pytest.mark.parametrize("repeats", [0, -1, 26, True, 1.0])
def test_profile_rejects_invalid_repeat_admission(
    models: Path, tmp_path: Path, repeats: int
) -> None:
    with pytest.raises(ValueError, match="integer from 1 to 25"):
        profile_directory(models, tmp_path / "profile", repeats=repeats)


def test_profile_retains_real_processing_failure(models: Path, tmp_path: Path) -> None:
    (models / "bad.md").write_bytes(b"\xff")
    output = tmp_path / "profile"
    with pytest.raises(ValueError, match="did not complete"):
        profile_directory(models, output)
    receipt = json.loads((output / "profile_report.json").read_text())
    assert receipt["status"] == "failed" and receipt["unprofiled_samples"] == []
    assert not (output / "trace.pstats").exists()


def test_profile_refuses_output_within_selected_input(models: Path) -> None:
    with pytest.raises(ValueError, match="outside the selected input"):
        profile_directory(models, models / "profile")
    assert not (models / "profile").exists()


@pytest.mark.parametrize("executor", ["process", "thread"])
def test_profile_worker_results_preserve_models_and_disclose_trace_scope(
    models: Path, tmp_path: Path, executor: str
) -> None:
    (models / "second.md").write_text((models / "model.md").read_text())
    serial = profile_directory(models, tmp_path / "serial", repeats=1)
    parallel = profile_directory(
        models, tmp_path / executor, repeats=2, workers=2, executor=executor
    )
    assert parallel["status"] == "success"
    assert (
        parallel["normalized_model_result_sha256"]
        == serial["normalized_model_result_sha256"]
    )
    assert parallel["controls"]["effective_workers"] == 2
    assert parallel["controls"]["executor"] == executor
    assert parallel["trace"]["coverage"] == "calling thread only"
    assert parallel["trace"]["worker_execution_included"] is False
    assert "not traced" in parallel["trace"]["warning"]
    for sample in parallel["unprofiled_samples"]:
        assert sample["worker_execution"]["worker_pids"]
        assert sample["worker_execution"]["executor"] == executor
        assert (
            sample["children_cpu_seconds"] is None
            or sample["children_cpu_seconds"] >= 0
        )


@pytest.mark.parametrize(
    "workers,executor", [(0, None), (True, None), (257, None), (2, "unknown")]
)
def test_profile_worker_admission_precedes_output_creation(
    models: Path, tmp_path: Path, workers: int, executor: str | None
) -> None:
    output = tmp_path / "profile"
    with pytest.raises(ValueError):
        profile_directory(models, output, workers=workers, executor=executor)
    assert not output.exists()


def test_lightweight_variable_boundaries_preserve_unicode_and_multiline_matches() -> (
    None
):
    content = "z" * 10000 + "\nα: float\nprefix=4;\nx \n =2\nz[3]\nα=7\n变量2 = -1"
    parsed = parse_gnn_file("inline.md", content=content)
    assert parsed["success"]
    assert parsed["variables"] == ["α", "prefix", "x", "变量2", "z"]


@pytest.mark.parametrize(
    "content",
    [
        "word" * 2500 + "\nα: float\nx \n =2\nz[3]\n变量2 = -1",
        "Δ\n [first,\n second]\nΓ \n : \n Real\nπ = 3.14;\nΓ[2]\n",
        "not_a_variable" * 750 + "! name=one; value: int\narray[unterminated",
        "__123\u00a0:\u2003 int\n9_array\u2028[one,[two]]\n__123\u2009=\u3000 4;\nδ_7[[]]\n",
    ],
    ids=[
        "long-word",
        "unicode-multiline",
        "unclosed-array",
        "unicode-space-digits-nested",
    ],
)
def test_lightweight_variable_extraction_matches_legacy_oracle(content: str) -> None:
    patterns = [
        r"(\w+)\s*:\s*(\w+)",
        r"(\w+)\s*=\s*([^;\n]+)",
        r"(\w+)\s*\[([^\]]+)\]",
    ]
    expected = list(
        dict.fromkeys(
            match.group(1)
            for pattern in patterns
            for match in re.finditer(pattern, content)
        )
    )
    assert parse_gnn_file("inline.md", content=content)["variables"] == expected
