"""Real-behavior tests for ``src/pipeline/hasher.py``.

The run hash is the pipeline's identity primitive: it decides which stored
run a CLI lookup returns. These tests exercise the real hashing, indexing,
and prefix-lookup path against real files in temporary directories (no mocks,
no stubs).

Forward-implementation-first note: the run hash is substantive input/revision
identity (it decides *which* run you operate on), so it is covered as product
behavior, not administrative bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.hasher import compute_run_hash, compute_run_hash_with_files, index_run, lookup_run


@pytest.fixture()
def gnn_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gnn_files"
    d.mkdir()
    (d / "model_a.md").write_text("# GNN\nStateSpaceBlock: s [3]", encoding="utf-8")
    (d / "nested").mkdir()
    (d / "nested" / "model_b.gnn").write_text("## GNN b", encoding="utf-8")
    return d


def test_run_hash_is_deterministic_and_content_sensitive(gnn_dir: Path):
    h1 = compute_run_hash(gnn_dir, config={"a": 1})
    h2 = compute_run_hash(gnn_dir, config={"a": 1})
    assert h1 == h2
    assert len(h1) == 12

    # Changing a file's content changes the hash
    (gnn_dir / "model_a.md").write_text("# GNN changed\n", encoding="utf-8")
    assert compute_run_hash(gnn_dir, config={"a": 1}) != h1

    # Changing the config changes the hash even with identical files
    (gnn_dir / "model_a.md").write_text("# GNN\nStateSpaceBlock: s [3]", encoding="utf-8")
    assert compute_run_hash(gnn_dir, config={"a": 2}) != h1


def test_file_hashes_uses_relative_paths_and_hex_digests(gnn_dir: Path):
    _, file_hashes = compute_run_hash_with_files(gnn_dir)
    assert set(file_hashes) == {"model_a.md", str(Path("nested") / "model_b.gnn")}
    for digest in file_hashes.values():
        assert len(digest) == 64
        int(digest, 16)  # valid hex


def test_index_then_lookup_roundtrip_exact_and_prefix(tmp_path: Path):
    summary = tmp_path / "pipeline_execution_summary.json"
    summary.write_text("{}", encoding="utf-8")
    history = tmp_path / ".history"

    index_run("deadbeef1234", summary, history_dir=history,
              config={"target": "input/gnn_files"})
    index_path = history / "index.json"
    assert index_path.exists()
    stored = json.loads(index_path.read_text(encoding="utf-8"))
    assert "deadbeef1234" in stored

    assert lookup_run("deadbeef1234", history)["config"] == {"target": "input/gnn_files"}
    # Prefix lookup returns the single match
    assert lookup_run("deadbeef", history)["config"] == {"target": "input/gnn_files"}


def test_lookup_prefix_ambiguity_returns_none_with_warning(tmp_path: Path):
    summary = tmp_path / "pipeline_execution_summary.json"
    summary.write_text("{}", encoding="utf-8")
    history = tmp_path / ".history"
    index_run("aaaa1111ffff", summary, history_dir=history)
    index_run("aaaa2222ffff", summary, history_dir=history)

    assert lookup_run("aaaa", history) is None  # ambiguous prefix
    assert lookup_run("aaaa1111ffff", history) is not None  # exact still resolves


def test_lookup_missing_history_returns_none(tmp_path: Path):
    assert lookup_run("whatever", tmp_path / "no_such_history") is None
