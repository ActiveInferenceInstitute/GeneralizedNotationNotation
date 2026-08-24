#!/usr/bin/env python3
"""Edge-case tests for utils.io_utils (batch file helpers).

Covers the GNN shared file I/O helpers, one of the honestly-owned utils
paths. These functions previously had effectively zero dedicated coverage;
this file pins the real edge cases: text vs bytes vs serialized writes,
atomic temp-file replacement, missing-input handling, and cleanup of
already-absent temp paths.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.io_utils import (
    batch_read_files,
    batch_write_files,
    cleanup_temp_files,
    create_temp_file_with_content,
    get_file_performance_metrics,
)


class TestBatchWriteFiles:
    """Edge cases for batch_write_files."""

    def test_writes_nested_text_and_bytes(
        self, isolated_temp_dir: Path
    ) -> None:
        files_data: List[Dict[str, Any]] = [
            {"path": "nested/deep/file.txt", "content": "hello"},
            {"path": "raw.bin", "content": b"\x00\x01\x02"},
            {"path": "data.json", "content": {"a": [1, 2], "b": True}},
        ]
        result = batch_write_files(files_data, isolated_temp_dir)

        assert result["total_files"] == 3
        assert result["successful_writes"] == 3
        assert result["failed_writes"] == 0
        assert result["total_size_bytes"] > 0

        # Verify actual on-disk bytes round-trip exactly.
        assert (isolated_temp_dir / "nested/deep/file.txt").read_text(
            encoding="utf-8"
        ) == "hello"
        assert (isolated_temp_dir / "raw.bin").read_bytes() == b"\x00\x01\x02"

    def test_atomic_replace_overwrites_existing(
        self, isolated_temp_dir: Path
    ) -> None:
        """Repeated writes to the same path replace cleanly (no .tmp residue)."""
        existing = isolated_temp_dir / "out.txt"
        existing.write_text("old", encoding="utf-8")

        result = batch_write_files(
            [{"path": "out.txt", "content": "new"}], isolated_temp_dir
        )
        assert result["successful_writes"] == 1
        assert existing.read_text(encoding="utf-8") == "new"
        # No leftover temp files after atomic replace.
        leftovers = [
            p for p in isolated_temp_dir.rglob("*.tmp") if p.is_file()
        ]
        assert leftovers == []

    def test_failed_write_is_reported_and_others_succeed(
        self, isolated_temp_dir: Path
    ) -> None:
        """A failure for one entry (invalid JSON content) must not abort the batch."""
        files_data: List[Dict[str, Any]] = [
            {"path": "good.txt", "content": "fine"},
            {"path": "bad.json", "content": object()},  # not JSON-serializable
        ]
        result = batch_write_files(files_data, isolated_temp_dir)

        assert result["total_files"] == 2
        assert result["successful_writes"] == 1
        assert result["failed_writes"] == 1
        assert result["results"][1]["success"] is False
        assert "error" in result["results"][1]
        assert (isolated_temp_dir / "good.txt").exists()

    def test_writes_to_nonexistent_output_dir_are_created(
        self, isolated_temp_dir: Path
    ) -> None:
        """The helper mkdirs parent paths even when output_dir does not exist."""
        result = batch_write_files(
            [{"path": "a/b.txt", "content": "x"}],
            isolated_temp_dir / "brand_new" / "output",
        )
        assert result["successful_writes"] == 1


class TestBatchRead:
    """Edge cases for batch_read_files."""

    def test_missing_and_existing_mixed(
        self, isolated_temp_dir: Path
    ) -> None:
        (isolated_temp_dir / "present.txt").write_text("data", encoding="utf-8")
        (isolated_temp_dir / "binary.bin").write_bytes(b"\xff\xfe\x00\x01")

        result = batch_read_files(
            [
                isolated_temp_dir / "present.txt",
                isolated_temp_dir / "binary.bin",
                isolated_temp_dir / "missing.txt",
            ]
        )

        assert result["total_files"] == 3
        assert result["successful_reads"] == 2
        assert result["failed_reads"] == 1
        by_path = {r["path"]: r for r in result["results"]}
        assert by_path[str(isolated_temp_dir / "present.txt")]["content_type"] == "text"
        assert by_path[str(isolated_temp_dir / "binary.bin")]["content_type"] == "binary"
        assert by_path[str(isolated_temp_dir / "missing.txt")]["success"] is False
        assert by_path[str(isolated_temp_dir / "missing.txt")]["error"] == "File not found"

    def test_empty_file_list(self) -> None:
        result = batch_read_files([])
        assert result["total_files"] == 0
        assert result["successful_reads"] == 0
        assert result["failed_reads"] == 0
        assert result["throughput_mbps"] == 0

    def test_utf8_text_with_invalid_byte_falls_back_to_binary(
        self, isolated_temp_dir: Path
    ) -> None:
        """A non-UTF-8 blob should read as binary rather than raising."""
        path = isolated_temp_dir / "mixed.bin"
        path.write_bytes(b"start\xff\xfeinvalid")

        result = batch_read_files([path])
        assert result["successful_reads"] == 1
        assert result["results"][0]["content_type"] == "binary"
        assert result["results"][0]["content_length"] == len(
            path.read_bytes()
        )


class TestGetFilePerformanceMetrics:
    """Edge cases for get_file_performance_metrics."""

    def test_missing_file_reports_not_exists(self, isolated_temp_dir: Path) -> None:
        metrics = get_file_performance_metrics(isolated_temp_dir / "nope.txt")
        assert metrics["exists"] is False
        assert "error" in metrics

    def test_existing_file_reports_metrics(self, isolated_temp_dir: Path) -> None:
        path = isolated_temp_dir / "readme.txt"
        path.write_text("some content", encoding="utf-8")
        metrics = get_file_performance_metrics(path)
        assert metrics["exists"] is True
        assert metrics["size_bytes"] > 0
        assert metrics["size_mb"] >= 0
        assert metrics["read_time_seconds"] >= 0
        assert "read_throughput_mbps" in metrics
        assert "modified_time" in metrics


class TestTempFiles:
    """Edge cases for create_temp_file_with_content / cleanup_temp_files."""

    def test_create_text_and_bytes(self, isolated_temp_dir: Path) -> None:
        text_file = create_temp_file_with_content("hello", suffix=".txt")
        bin_file = create_temp_file_with_content(b"\x01\x02", suffix=".bin")
        try:
            assert text_file.suffix == ".txt"
            assert bin_file.suffix == ".bin"
            assert text_file.read_text(encoding="utf-8") == "hello"
            assert bin_file.read_bytes() == b"\x01\x02"
        finally:
            text_file.unlink(missing_ok=True)
            bin_file.unlink(missing_ok=True)

    def test_cleanup_reports_already_absent_as_success(
        self, isolated_temp_dir: Path
    ) -> None:
        present = isolated_temp_dir / "present.tmp"
        present.write_text("x", encoding="utf-8")
        absent = isolated_temp_dir / "already_gone.tmp"

        result = cleanup_temp_files([present, absent])
        assert result["total_files"] == 2
        assert result["successful_cleanups"] == 2
        assert result["failed_cleanups"] == 0
        assert present.exists() is False  # actually removed

    def test_cleanup_empty_list(self) -> None:
        result = cleanup_temp_files([])
        assert result["total_files"] == 0
        assert result["successful_cleanups"] == 0