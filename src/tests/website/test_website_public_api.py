"""Tests for the website module's public API surface not covered by test_website_overall.

Covers: embed_* functions, generate_html_report, process_website, FEATURES,
SUPPORTED_FILE_TYPES, get_supported_file_types (extended), generate_website.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestWebsiteFeaturesAndConstants:
    """Test module-level constants and feature flags."""

    def test_features_dict_exists(self) -> None:
        import website

        assert hasattr(website, "FEATURES")
        assert isinstance(website.FEATURES, dict)
        assert len(website.FEATURES) > 0

    def test_features_contains_expected_keys(self) -> None:
        import website

        for key in (
            "html",
            "embedding",
            "basic_processing",
            "mcp_integration",
            "multi_page",
            "dark_mode",
        ):
            assert key in website.FEATURES, f"Missing FEATURES key: {key}"

    def test_supported_file_types_dict(self) -> None:
        import website

        assert hasattr(website, "SUPPORTED_FILE_TYPES")
        assert isinstance(website.SUPPORTED_FILE_TYPES, dict)
        assert "html" in website.SUPPORTED_FILE_TYPES
        assert "text" in website.SUPPORTED_FILE_TYPES
        assert "images" in website.SUPPORTED_FILE_TYPES

    def test_get_supported_file_types_returns_list(self) -> None:
        from website import get_supported_file_types

        types = get_supported_file_types()
        assert isinstance(types, list)
        # Should contain common file extensions
        assert "html" in types
        assert "css" in types
        assert "js" in types
        assert "json" in types


class TestWebsiteEmbedFunctions:
    """Test embed_image, embed_markdown_file, embed_text_file, embed_json_file, embed_html_file."""

    def test_embed_image_missing_file_returns_false(self, tmp_path: Any) -> None:
        from website import embed_image

        missing = tmp_path / "nonexistent.png"
        out = tmp_path / "out.html"
        result = embed_image(missing, out)
        # When source file doesn't exist, should return False (graceful)
        assert result is False

    def test_embed_image_creates_output(self, tmp_path: Any) -> None:
        from website import embed_image

        img = tmp_path / "test.png"
        img.write_text("sample-png-data")
        out = tmp_path / "out.html"
        result = embed_image(img, out)
        assert result is True
        assert out.exists()
        content = out.read_text()
        assert "test.png" in content

    def test_embed_markdown_file_missing_returns_false(self, tmp_path: Any) -> None:
        from website import embed_markdown_file

        missing = tmp_path / "nonexistent.md"
        out = tmp_path / "out.html"
        result = embed_markdown_file(missing, out)
        assert result is False

    def test_embed_markdown_file_creates_html(self, tmp_path: Any) -> None:
        from website import embed_markdown_file

        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nWorld")
        out = tmp_path / "out.html"
        result = embed_markdown_file(md, out)
        assert result is True
        assert out.exists()
        content = out.read_text()
        assert "Hello" in content or "html" in content.lower()

    def test_embed_text_file_creates_output(self, tmp_path: Any) -> None:
        from website import embed_text_file

        txt = tmp_path / "test.txt"
        txt.write_text("plain text content")
        out = tmp_path / "out.html"
        result = embed_text_file(txt, out)
        assert result is True
        assert out.exists()

    def test_embed_text_file_missing_returns_false(self, tmp_path: Any) -> None:
        from website import embed_text_file

        result = embed_text_file(tmp_path / "missing.txt", tmp_path / "out.html")
        assert result is False

    def test_embed_json_file_creates_output(self, tmp_path: Any) -> None:
        from website import embed_json_file

        js = tmp_path / "data.json"
        js.write_text('{"key": "value"}')
        out = tmp_path / "out.html"
        result = embed_json_file(js, out)
        assert result is True
        assert out.exists()

    def test_embed_html_file_appends_content(self, tmp_path: Any) -> None:
        from website import embed_html_file

        src = tmp_path / "src.html"
        src.write_text("<p>hello</p>")
        out = tmp_path / "dest.html"
        result = embed_html_file(src, out)
        assert result is True
        assert out.exists()

    def test_embed_html_file_missing_returns_false(self, tmp_path: Any) -> None:
        from website import embed_html_file

        result = embed_html_file(tmp_path / "missing.html", tmp_path / "out.html")
        assert result is False


class TestGenerateHtmlReport:
    """Test generate_html_report function."""

    def test_generate_with_string_content(self, tmp_path: Any) -> None:
        from website import generate_html_report

        out = tmp_path / "report.html"
        result = generate_html_report("Hello **world**", out)
        assert result is True
        assert out.exists()
        content = out.read_text()
        assert len(content) > 0

    def test_generate_with_dict_content(self, tmp_path: Any) -> None:
        from website import generate_html_report

        out = tmp_path / "report.html"
        data: dict[str, Any] = {"name": "test", "value": 42}
        result = generate_html_report(str(data), out)
        assert result is True
        assert out.exists()

    def test_generate_with_empty_string(self, tmp_path: Any) -> None:
        from website import generate_html_report

        out = tmp_path / "report.html"
        result = generate_html_report("", out)
        assert result is True
        assert out.exists()


class TestProcessWebsite:
    """Test process_website function."""

    def test_process_empty_target_dir(self, tmp_path: Any) -> None:
        from website import process_website

        target = tmp_path / "empty_target"
        target.mkdir()
        out = tmp_path / "output"
        result = process_website(target_dir=target, output_dir=out)
        # Should not crash; returns bool
        assert isinstance(result, bool)

    def test_process_nonexistent_target(self, tmp_path: Any) -> None:
        from website import process_website

        out = tmp_path / "output"
        result = process_website(
            target_dir=tmp_path / "nonexistent",
            output_dir=out,
        )
        assert isinstance(result, bool)

    def test_process_with_gnn_content(self, safe_filesystem: Any) -> None:
        from website import process_website

        safe_filesystem.create_file("model.gnn", "# Test\n## ModelName\nM")
        out = safe_filesystem.create_dir("output")
        result = process_website(target_dir=safe_filesystem.temp_dir, output_dir=out)
        assert isinstance(result, bool)


class TestGenerateWebsite:
    """Test generate_website function from generator module."""

    def test_generate_website_returns_result(self, tmp_path: Any) -> None:
        import logging

        from website import generate_website

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.md").write_text("# Test")
        out_dir = tmp_path / "out"
        logger = logging.getLogger("test_website_gen")
        result = generate_website(logger, input_dir, out_dir)
        assert isinstance(result, dict)

    def test_generate_website_with_empty_dir(self, tmp_path: Any) -> None:
        import logging

        from website import generate_website

        input_dir = tmp_path / "input_empty"
        input_dir.mkdir()
        out_dir = tmp_path / "out"
        logger = logging.getLogger("test_website_gen_empty")
        result = generate_website(logger, input_dir, out_dir)
        assert isinstance(result, dict)
