#!/usr/bin/env python3
"""
Exercises the audio module MCP tool handlers.

Exercises the metadata and probing handler functions in ``src/audio/mcp.py``:
backend availability, generation options, module info, and content validation.
The heavy audio-rendering path (``process_audio_mcp``) is only exercised via
its graceful failure mode on a missing target directory.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audio import mcp as audio_mcp


class TestAudioMCPTools:
    """Tests for the audio MCP tool handlers."""

    @pytest.mark.unit
    def test_check_audio_backends(self) -> None:
        """Backend probing should report availability per backend."""
        result = audio_mcp.check_audio_backends_mcp()
        assert result["success"] is True
        # The probed backends are the top-level keys alongside 'success'.
        backend_names = {k for k in result if k != "success"}
        assert backend_names, "Expected at least one probed backend"
        for name in backend_names:
            assert isinstance(result[name], dict)
            assert "available" in result[name]

    @pytest.mark.unit
    def test_get_audio_generation_options(self) -> None:
        """Generation options should surface the documented knobs."""
        result = audio_mcp.get_audio_generation_options_mcp()
        assert result["success"] is True
        assert "options" in result
        assert result["options"], "Expected at least one generation option"

    @pytest.mark.unit
    def test_get_audio_module_info(self) -> None:
        """Module info should describe the audio surface with features."""
        result = audio_mcp.get_audio_module_info_mcp()
        assert result["success"] is True
        assert "description" in result or "name" in result
        assert "features" in result

    @pytest.mark.unit
    def test_validate_missing_audio_file(self) -> None:
        """Validating a nonexistent file should return a graceful result dict."""
        result = audio_mcp.validate_audio_content_mcp("/no/such/audio.wav")
        assert isinstance(result, dict)
        assert "success" in result

    @pytest.mark.unit
    def test_process_audio_nonexistent_target(self, tmp_path: Any) -> None:
        """A missing target directory should not raise; returns a graceful dict."""
        missing = tmp_path / "no_input"
        out = tmp_path / "audio_out"
        result = audio_mcp.process_audio_mcp(str(missing), str(out))
        assert isinstance(result, dict)
        assert "success" in result
        assert "message" in result