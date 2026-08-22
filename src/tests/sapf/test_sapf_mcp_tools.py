#!/usr/bin/env python3
"""
No-mock tests for the SAPF module MCP tool handlers.

Exercises the handler functions in ``src/sapf/mcp.py`` that expose audio
metadata, artifact inventory, and backend probing. The heavy audio-rendering
path is only exercised through its graceful failure mode (missing target).
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sapf import mcp as sapf_mcp


class TestSAPFMCPTools:
    """Tests for the SAPF MCP tool handlers."""

    @pytest.mark.unit
    def test_get_module_info(self) -> None:
        """Module info should describe the SAPF surface."""
        result = sapf_mcp.get_sapf_module_info_mcp()
        assert result["success"] is True
        assert "description" in result
        assert "features" in result

    @pytest.mark.unit
    def test_list_audio_artifacts_directory(self, tmp_path: Any) -> None:
        """Listing a directory should inventory audio/SAPF files by extension."""
        out = tmp_path / "audio"
        out.mkdir(exist_ok=True)
        (out / "song.wav").write_bytes(b"\x00")
        (out / "patch.sc").write_text("// supercollider patch")
        (out / "readme.txt").write_text("irrelevant")
        result = sapf_mcp.list_audio_artifacts_mcp(str(out))
        assert result["success"] is True
        assert result["total_artifacts"] == 2
        assert result["by_type"]["wav"] == 1
        assert result["by_type"]["sc"] == 1
        types = {a["type"] for a in result["artifacts"]}
        assert types == {"wav", "sc"}

    @pytest.mark.unit
    def test_list_audio_artifacts_missing_directory(self, tmp_path: Any) -> None:
        """A missing directory should yield a clear error."""
        missing = tmp_path / "no_audio"
        result = sapf_mcp.list_audio_artifacts_mcp(str(missing))
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.unit
    def test_check_audio_backends_shape(self) -> None:
        """Backend probing should report availability per backend."""
        result = sapf_mcp.check_audio_backends_mcp()
        assert result["success"] is True
        assert "backends" in result
        assert result["backends"], "Expected at least one probed backend"
        for metadata in result["backends"].values():
            assert "available" in metadata

    @pytest.mark.unit
    def test_process_sapf_nonexistent_target(self, tmp_path: Any) -> None:
        """A nonexistent target directory returns a graceful failure dict."""
        missing = tmp_path / "no_input"
        out = tmp_path / "audio_out"
        result = sapf_mcp.process_sapf_mcp(str(missing), str(out))
        assert isinstance(result, dict)
        # The handler creates the output dir first, then fails resolving globs.
        assert out.exists() or result["success"] is False