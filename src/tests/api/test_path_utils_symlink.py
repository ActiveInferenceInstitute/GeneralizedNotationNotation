#!/usr/bin/env python3
"""Tests for symlink-safe API path boundary enforcement (api.path_utils)."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.path_utils import PathValidationError, resolve_repo_path


class TestSymlinkPathBoundary:
    """resolve_repo_path must reject symlink traversal (RED_TEAM V-05)."""

    def test_accepts_normal_relative_path(self, tmp_path: Path) -> None:
        (tmp_path / "input").mkdir()
        resolved = resolve_repo_path(
            "input", purpose="Target directory", must_exist=True
        )
        assert resolved.exists()

    def test_rejects_escape_outside_repo(self) -> None:
        with pytest.raises(PathValidationError):
            resolve_repo_path("../../etc", purpose="Target directory", must_exist=False)

    def test_rejects_symlink_component(self, tmp_path: Path) -> None:
        # A symlink planted inside the repo pointing outside must be refused,
        # even though resolve() would flatten it to a location that a naive
        # relative_to() check might otherwise accept.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "victim.txt").write_text("secret")

        repo_root = Path(__file__).parent.parent.parent.parent  # repo root
        link_target = repo_root / "output" / "escaping_link"
        link_target.parent.mkdir(parents=True, exist_ok=True)
        if link_target.exists() or link_target.is_symlink():
            link_target.unlink()
        os.symlink(str(outside), str(link_target))

        try:
            with pytest.raises(PathValidationError):
                resolve_repo_path(
                    "output/escaping_link",
                    purpose="Output directory",
                    must_exist=False,
                )
        finally:
            if link_target.is_symlink():
                link_target.unlink()
