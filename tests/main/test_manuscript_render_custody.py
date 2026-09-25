"""Fail-closed reader contracts for the render-custody manifest.

``load_render_manifest`` is the shared reader behind the custody gates
(``scripts/z_record_manuscript_render_manifest.py``,
``scripts/z_verify_fresh_render.py``, and the strict token gate); every
failure must name the regen command instead of returning a partial dict.
``strip_volatile_tokens`` is the exact commit-varying token exclusion that
defines ``variables_sha256`` (mirrored by
``scripts/check_manuscript_tokens._strip_volatile_tokens``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.manuscript.render_custody import (
    load_render_manifest,
    manifest_path,
    strip_volatile_tokens,
)

pytestmark = pytest.mark.unit

_MANIFEST_REL_PARTS = ("output", "data", "manuscript_render_manifest.json")


def _manifest_path(project_root: Path) -> Path:
    path = project_root.joinpath(*_MANIFEST_REL_PARTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_load_render_manifest_missing_names_regen_command(tmp_path: Path) -> None:
    """A missing manifest raises with the recording command, not a bare error."""
    with pytest.raises(RuntimeError, match="z_record_manuscript_render_manifest"):
        load_render_manifest(tmp_path)


def test_load_render_manifest_rejects_corrupt_json(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        load_render_manifest(tmp_path)


def test_load_render_manifest_rejects_non_object(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        load_render_manifest(tmp_path)


def test_load_render_manifest_roundtrip_at_documented_path(tmp_path: Path) -> None:
    manifest = {
        "manifest_version": "gnn_render_manifest_v1",
        "counts_describe_commit": "abc1234",
    }
    path = _manifest_path(tmp_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest_path(tmp_path) == path
    assert load_render_manifest(tmp_path) == manifest


def test_strip_volatile_tokens_drops_only_the_commit_token() -> None:
    variables = {
        "GNN_VERSION": "3.5.0",
        "GNN_GIT_COMMIT": "abc1234",
        "GNN_RELEASE_DATE": "2026-09-22",
    }
    assert strip_volatile_tokens(variables) == {
        "GNN_VERSION": "3.5.0",
        "GNN_RELEASE_DATE": "2026-09-22",
    }
    assert strip_volatile_tokens({}) == {}
