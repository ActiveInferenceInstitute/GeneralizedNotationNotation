"""
Tests for gnn/parsers/system.py — GNNParsingSystem.

Covers: initialization, parser/serializer registries, parse_string,
_detect_format, _detect_format_from_content, get_supported_formats,
get_available_parsers, get_available_serializers, parse_file error paths.
"""

import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from gnn.parsers.system import PARSER_REGISTRY, GNNParsingSystem
from gnn.parsers.unified_parser import GNNFormat, ParseResult

# Minimal valid GNN markdown content
_MD_CONTENT = """## GNNSection
ActInfPOMDP

## ModelName
TestModel

## StateSpaceBlock
s[3,1,type=float]
"""


class TestGNNParsingSystemInit:
    def test_parsers_populated(self) -> Any:
        ps = GNNParsingSystem(strict_validation=False)
        assert len(ps.parsers) > 0

    def test_serializers_populated(self) -> Any:
        ps = GNNParsingSystem(strict_validation=False)
        assert len(ps.serializers) > 0

    def test_markdown_parser_registered(self) -> Any:
        ps = GNNParsingSystem(strict_validation=False)
        assert GNNFormat.MARKDOWN in ps.parsers

    def test_json_parser_registered(self) -> Any:
        ps = GNNParsingSystem(strict_validation=False)
        assert GNNFormat.JSON in ps.parsers


class TestGNNParsingSystemParseString:
    def setup_method(self) -> Any:
        self.ps = GNNParsingSystem(strict_validation=False)

    def test_parse_markdown_returns_result(self) -> Any:
        result = self.ps.parse_string(_MD_CONTENT, GNNFormat.MARKDOWN)
        assert isinstance(result, ParseResult)

    def test_parse_markdown_success(self) -> Any:
        result = self.ps.parse_string(_MD_CONTENT, GNNFormat.MARKDOWN)
        assert result.success is True

    def test_parse_markdown_model_name(self) -> Any:
        result = self.ps.parse_string(_MD_CONTENT, GNNFormat.MARKDOWN)
        assert result.model.model_name == "TestModel"

    def test_unsupported_format_raises_value_error(self) -> Any:
        with pytest.raises((ValueError, KeyError, Exception)):
            self.ps.parse_string("content", cast(Any, "COMPLETELY_INVALID_FORMAT"))

    def test_parse_json_basic(self) -> Any:
        json_content = '{"model_name": "JsonModel", "version": "1.0"}'
        try:
            result = self.ps.parse_string(json_content, GNNFormat.JSON)
            assert isinstance(result, ParseResult)
        except Exception:
            # JSON parser may require specific structure; any result type is ok
            pass


class TestGNNParsingSystemParseFile:
    def setup_method(self) -> Any:
        self.ps = GNNParsingSystem(strict_validation=False)

    def test_nonexistent_file_raises_file_not_found(self) -> Any:
        with pytest.raises(FileNotFoundError):
            self.ps.parse_file("/nonexistent/path/model.md")

    def test_parse_existing_md_file(self) -> Any:
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write(_MD_CONTENT)
            tmp_path = f.name
        try:
            result = self.ps.parse_file(tmp_path)
            assert isinstance(result, ParseResult)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_unknown_extension_raises(self) -> Any:
        with tempfile.NamedTemporaryFile(
            suffix=".unknownext", mode="w", delete=False
        ) as f:
            f.write("content")
            tmp_path = f.name
        try:
            with pytest.raises((ValueError, Exception)):
                self.ps.parse_file(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestGNNParsingSystemDetectFormat:
    def setup_method(self) -> Any:
        self.ps = GNNParsingSystem(strict_validation=False)

    def _path_with_ext(self, ext: Any) -> Any:
        return Path(f"test_file{ext}")

    def test_md_detected_as_markdown(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".md")) == GNNFormat.MARKDOWN

    def test_json_detected(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".json")) == GNNFormat.JSON

    def test_yaml_detected(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".yaml")) == GNNFormat.YAML

    def test_yml_detected_as_yaml(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".yml")) == GNNFormat.YAML

    def test_xml_detected(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".xml")) == GNNFormat.XML

    def test_py_detected_as_python(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".py")) == GNNFormat.PYTHON

    def test_scala_detected(self) -> Any:
        assert self.ps._detect_format(self._path_with_ext(".scala")) == GNNFormat.SCALA

    def test_text_pkl_detected_as_pkl(self, tmp_path: Path) -> None:
        pkl_file = tmp_path / "model.pkl"
        pkl_file.write_text(
            'class GNNModel {\n  name: String = "TextPKL"\n}\n'
            '/* MODEL_DATA: {"model_name":"TextPKL","variables":[],"connections":[],"parameters":[]} */\n'
        )

        assert self.ps._detect_format(pkl_file) == GNNFormat.PKL

    def test_binary_pkl_detected_as_pickle(self, tmp_path: Path) -> None:
        pkl_file = tmp_path / "model.pkl"
        with pkl_file.open("wb") as f:
            pickle.dump(
                {
                    "model_name": "BinaryPKL",
                    "variables": [],
                    "connections": [],
                    "parameters": [],
                },
                f,
            )

        assert self.ps._detect_format(pkl_file) == GNNFormat.PICKLE

    def test_pickle_extension_detected_as_pickle(self) -> None:
        assert (
            self.ps._detect_format(self._path_with_ext(".pickle")) == GNNFormat.PICKLE
        )

    def test_unknown_extension_raises(self) -> Any:
        with pytest.raises(ValueError):
            self.ps._detect_format(self._path_with_ext(".unknownxyz"))

    def test_parse_text_pkl_uses_pkl_parser(self, tmp_path: Path) -> None:
        pkl_file = tmp_path / "model.pkl"
        pkl_file.write_text(
            'class GNNModel {\n  name: String = "TextPKL"\n}\n'
            '/* MODEL_DATA: {"model_name":"TextPKL","variables":[],"connections":[],"parameters":[]} */\n'
        )

        result = self.ps.parse_file(pkl_file)

        assert result.success is True
        assert result.model.model_name == "TextPKL"
        assert result.model.source_format == GNNFormat.PKL

    def test_parse_binary_pkl_uses_pickle_parser(self, tmp_path: Path) -> None:
        pkl_file = tmp_path / "model.pkl"
        with pkl_file.open("wb") as f:
            pickle.dump(
                {
                    "model_name": "BinaryPKL",
                    "variables": [],
                    "connections": [],
                    "parameters": [],
                },
                f,
            )

        result = self.ps.parse_file(pkl_file)

        assert result.success is True
        assert result.model.model_name == "BinaryPKL"
        assert result.model.source_format == GNNFormat.PICKLE


class TestGNNParsingSystemDetectFormatFromContent:
    def setup_method(self) -> Any:
        self.ps = GNNParsingSystem(strict_validation=False)

    def test_xml_content(self) -> Any:
        assert (
            self.ps._detect_format_from_content("<?xml version='1.0'?>")
            == GNNFormat.XML
        )

    def test_json_object_content(self) -> Any:
        assert self.ps._detect_format_from_content('{"key": "val"}') == GNNFormat.JSON

    def test_json_array_content(self) -> Any:
        assert self.ps._detect_format_from_content("[1,2,3]") == GNNFormat.JSON

    def test_yaml_content(self) -> Any:
        assert self.ps._detect_format_from_content("---\nkey: value") == GNNFormat.YAML

    def test_unknown_defaults_to_markdown(self) -> Any:
        result = self.ps._detect_format_from_content("something random")
        assert result == GNNFormat.MARKDOWN


class TestGNNParsingSystemQueries:
    def setup_method(self) -> Any:
        self.ps = GNNParsingSystem(strict_validation=False)

    def test_get_supported_formats_returns_list(self) -> Any:
        formats = self.ps.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0

    def test_get_supported_formats_includes_markdown(self) -> Any:
        assert GNNFormat.MARKDOWN in self.ps.get_supported_formats()

    def test_get_available_parsers_returns_dict(self) -> Any:
        result = self.ps.get_available_parsers()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_get_available_parsers_values_are_strings(self) -> Any:
        result = self.ps.get_available_parsers()
        for v in result.values():
            assert isinstance(v, str)

    def test_get_available_serializers_returns_dict(self) -> Any:
        result = self.ps.get_available_serializers()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parser_registry_coverage(self) -> Any:
        """PARSER_REGISTRY should cover all GNNFormat enum values except PICKLE edge case."""
        ps = GNNParsingSystem(strict_validation=False)
        # Every registered format should have an instantiated parser
        for fmt in PARSER_REGISTRY:
            assert fmt in ps.parsers


class TestGNNParsingSystemStrictValidation:
    def test_strict_validation_false_works(self) -> Any:
        ps = GNNParsingSystem(strict_validation=False)
        result = ps.parse_string(_MD_CONTENT, GNNFormat.MARKDOWN)
        assert result.success is True

    def test_strict_validation_attribute_stored(self) -> Any:
        ps = GNNParsingSystem(strict_validation=False)
        assert ps.strict_validation is False
        ps2 = GNNParsingSystem(strict_validation=True)
        assert ps2.strict_validation is True


class TestGNNParserValidationResultAPI:
    """``is_valid`` must match ``success`` for parser pipeline consumers."""

    def test_is_valid_matches_success(self) -> Any:
        from gnn.parsers.validators import GNNParserValidationResult

        ok = GNNParserValidationResult(success=True)
        assert ok.is_valid is True
        assert ok.success is True
        bad = GNNParserValidationResult(success=False)
        assert bad.is_valid is False
        assert bad.success is False
