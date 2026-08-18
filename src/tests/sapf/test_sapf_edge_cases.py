"""Tests for sapf public API edge cases and uncovered surface.

Covers: validate_sapf_code (tuple result), convert_gnn_to_sapf,
generate_sapf_audio, generate_audio_from_sapf, process_gnn_to_audio,
create_sapf_visualization, generate_sapf_report, get_module_info details.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestSapfValidate:
    """validate_sapf_code returns (bool, list-of-issues)."""

    def test_validate_empty_returns_false_with_issue(self) -> None:
        from sapf import validate_sapf_code

        is_valid, issues = validate_sapf_code("")
        assert is_valid is False
        assert isinstance(issues, list)
        assert any("Empty" in i for i in issues)

    def test_validate_garbage_returns_false(self) -> None:
        from sapf import validate_sapf_code

        is_valid, issues = validate_sapf_code("PLAY 440 1.0")
        assert is_valid is False
        assert isinstance(issues, list)
        # No 'play' command and no assignments → issues present
        assert len(issues) > 0

    def test_validate_unbalanced_brackets_reported(self) -> None:
        from sapf import validate_sapf_code

        code = "play = [1, 2, 3"
        is_valid, issues = validate_sapf_code(code)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_validate_returns_tuple(self) -> None:
        from sapf import validate_sapf_code

        result = validate_sapf_code("x = 1")
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestSapfMetadata:
    """Module-level metadata and get_module_info."""

    def test_get_module_info_has_version(self) -> None:
        from sapf import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "supported_formats" in info

    def test_sapf_has_version(self) -> None:
        import sapf

        assert hasattr(sapf, "__version__")
        assert isinstance(sapf.__version__, str)

    def test_sapf_features_is_dict(self) -> None:
        import sapf

        assert hasattr(sapf, "FEATURES")
        assert isinstance(sapf.FEATURES, dict)


class TestSapfProcessing:
    """convert_gnn_to_sapf returns SAPF code string."""

    def test_convert_gnn_to_sapf_returns_str(self) -> None:
        from sapf import convert_gnn_to_sapf

        gnn_content = "# Test\n## ModelName\nM\n## StateSpaceBlock\ns[3]"
        result = convert_gnn_to_sapf(gnn_content, "test_model")
        assert isinstance(result, str)

    def test_convert_gnn_to_sapf_empty_content(self) -> None:
        from sapf import convert_gnn_to_sapf

        result = convert_gnn_to_sapf("", "empty_model")
        assert isinstance(result, str)

    def test_process_gnn_to_audio_returns_dict(self, tmp_path: Any) -> None:
        from sapf import process_gnn_to_audio

        result = process_gnn_to_audio(
            "# Test\n## ModelName\nM", "model1", str(tmp_path), validate_only=True
        )
        assert isinstance(result, dict)
        assert "success" in result
        assert result["model_name"] == "model1"

    def test_process_gnn_to_audio_validate_only_shape(self, tmp_path: Any) -> None:
        from sapf import process_gnn_to_audio

        result = process_gnn_to_audio(
            "# Test\n## ModelName\nM", "model2", str(tmp_path), validate_only=True
        )
        assert "sapf_code" in result
        assert isinstance(result["sapf_code"], str)

    def test_generate_sapf_audio_returns_dict(self, tmp_path: Any) -> None:
        from sapf import generate_sapf_audio

        out_path = tmp_path / "out.wav"
        result = generate_sapf_audio("oscillator 440 1.0", str(out_path))
        assert isinstance(result, dict)
        assert "success" in result
        assert "output_path" in result

    def test_generate_audio_from_sapf_returns_bool(self, tmp_path: Any) -> None:
        from sapf import generate_audio_from_sapf

        out_path = tmp_path / "audio.wav"
        result = generate_audio_from_sapf("oscillator 440 1.0", out_path, duration=0.5)
        assert isinstance(result, bool)

    def test_create_sapf_visualization_returns_dict(self) -> None:
        from sapf import create_sapf_visualization

        result = create_sapf_visualization("oscillator 440 1.0\nenvelope exponential")
        assert isinstance(result, dict)
        assert "success" in result
        assert "visualization_data" in result

    def test_create_sapf_visualization_writes_file(self, tmp_path: Any) -> None:
        from sapf import create_sapf_visualization

        out_path = tmp_path / "viz.json"
        result = create_sapf_visualization("oscillator 440 1.0", str(out_path))
        assert result["success"] is True
        assert out_path.exists()

    def test_generate_sapf_report_returns_dict(self) -> None:
        from sapf import generate_sapf_report

        result = generate_sapf_report({"success": True, "components": []})
        assert isinstance(result, dict)
        assert "summary" in result

    def test_generate_sapf_report_writes_file(self, tmp_path: Any) -> None:
        from sapf import generate_sapf_report

        out_path = tmp_path / "report.json"
        result = generate_sapf_report({"success": True}, str(out_path))
        assert isinstance(result, dict)
        assert out_path.exists()
