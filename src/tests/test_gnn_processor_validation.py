#!/usr/bin/env python3
"""Phase 1.3 regression: process_gnn_directory must validate directory inputs.

Before Phase 1.3, passing a nonexistent path to ``process_gnn_directory``
returned ``status=SUCCESS, files=[]`` — a silent-success that masked pipeline
configuration errors. After the fix it returns ``status=FAILED`` with a
diagnostic error message.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnn.processor import process_gnn_directory  # noqa: E402


def test_process_gnn_directory_fails_for_missing_path(tmp_path):
    missing = tmp_path / "definitely_not_here"
    result = process_gnn_directory(missing)
    assert result["status"] == "FAILED"
    assert "does not exist" in result["error"]
    assert result["files"] == []
    assert result["processed_files"] == []


def test_process_gnn_directory_accepts_single_file_path(tmp_path):
    """process_gnn_directory is intentionally lenient: it accepts either a
    directory or a single .md file path. Callers (tests, ad-hoc scripts) rely
    on both being valid, and ``discover_gnn_files`` handles each case."""
    file_path = tmp_path / "a.md"
    file_path.write_text("## GNNSection\nTest")
    result = process_gnn_directory(file_path)
    # Must not be rejected at the path-existence check.
    assert result.get("error", "") == "" or "does not exist" not in result["error"]


def test_process_gnn_directory_rejects_none():
    result = process_gnn_directory(None)
    assert result["status"] == "FAILED"
    assert "is None" in result["error"]


def test_process_gnn_directory_succeeds_for_empty_dir(tmp_path):
    # An existing but empty directory is not an error (it's just "no files").
    empty = tmp_path / "empty"
    empty.mkdir()
    result = process_gnn_directory(empty)
    # Existing-but-empty still progresses through the lightweight path; status
    # comes from the lightweight result. We assert it did NOT fail at validation.
    assert result["status"] != "FAILED" or "does not exist" not in result.get("error", "")


def test_process_gnn_directory_accepts_valid_dir(tmp_path):
    gnn_file = tmp_path / "sample.md"
    gnn_file.write_text(
        "## GNNSection\nTestModel\n\n## ModelName\nT\n\n## StateSpaceBlock\ns[2,1,type=float]\n"
    )
    result = process_gnn_directory(tmp_path)
    assert result["status"] != "FAILED" or "does not exist" not in result.get("error", "")
    # Should discover the file we just wrote.
    assert len(result["files"]) >= 1
