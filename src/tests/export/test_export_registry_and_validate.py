#!/usr/bin/env python3
"""Tests for the export format registry and validate_export_outputs.

Covers:
  - Registry completeness (8 canonical formats)
  - resolve_format_writer returns correct callables
  - get_format_categories grouping
  - is_supported_format
  - get_export_registry keyed by name
  - DEFAULT_PIPELINE_FORMATS is the 5 pipeline subset
  - validate_export_outputs: manifest-driven validation
  - validate_export_outputs: missing manifest
  - validate_export_outputs: incomplete model (expected_formats)
  - core.export_gnn_files now propagates writer failures (silent-failure fix)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# -- Registry -----------------------------------------------------------------


class TestRegistry:
    """Canonical format registry invariants."""

    def test_registry_has_eight_formats(self) -> None:
        from export.registry import get_export_registry

        reg = get_export_registry()
        assert len(reg) == 8
        assert set(reg) == {
            "json",
            "xml",
            "graphml",
            "gexf",
            "pickle",
            "txt",
            "dsl",
            "geo_infer",
        }

    def test_pipeline_defaults_are_five(self) -> None:
        from export.registry import DEFAULT_PIPELINE_FORMATS

        assert DEFAULT_PIPELINE_FORMATS == ("json", "xml", "graphml", "gexf", "pickle")

    def test_resolve_writer_known(self) -> None:
        from export.formatters import export_to_json
        from export.registry import resolve_format_writer

        assert resolve_format_writer("json") is export_to_json

    def test_resolve_writer_unknown(self) -> None:
        from export.registry import resolve_format_writer

        assert resolve_format_writer("yaml") is None

    def test_categories(self) -> None:
        from export.registry import get_format_categories

        cats = get_format_categories()
        assert isinstance(cats, dict)
        assert cats["data"] == ["geo_infer", "json", "xml", "pickle"]
        assert cats["graph"] == ["graphml", "gexf"]
        assert cats["text"] == ["txt", "dsl"]

    def test_is_supported(self) -> None:
        from export.registry import is_supported_format

        assert is_supported_format("json")
        assert not is_supported_format("yaml")

    def test_spec_has_writer_and_extension(self) -> None:
        from export.registry import get_export_registry

        for name, spec in get_export_registry().items():
            assert callable(spec["writer"]), f"{name} writer not callable"
            assert spec["extension"].startswith("."), f"{name} extension missing dot"

    def test_format_spec_lookup(self) -> None:
        from export.registry import get_format_spec

        spec = get_format_spec("json")
        assert spec is not None
        assert spec["category"] == "data"
        assert get_format_spec("yaml") is None


# -- validate_export_outputs ---------------------------------------------------


class TestValidateExportOutputs:
    """Manifest-driven export artifact validation."""

    @pytest.fixture()
    def _manifest_dir(self, tmp_path: Path) -> Path:
        """Create a minimal process_export output tree with a manifest."""
        out = tmp_path / "7_export_output"
        model_dir = out / "demo"
        model_dir.mkdir(parents=True)
        # Write real export files
        (model_dir / "demo_json.json").write_text('{"ok": true}')
        (model_dir / "demo_xml.xml").write_text('<?xml version="1.0"?>\n<gnn_model/>\n')
        # Write a valid pickle file (use pickle.dumps for portability)
        import pickle as _pkl

        (model_dir / "demo_pickle.pkl").write_bytes(_pkl.dumps({"ok": True}))
        # Manifest
        manifest = {
            "files_exported": [
                {
                    "file_name": "demo.md",
                    "file_path": "/fake/demo.md",
                    "success": True,
                    "exports": {
                        "json": {
                            "success": True,
                            "export_file": str(model_dir / "demo_json.json"),
                        },
                        "xml": {
                            "success": True,
                            "export_file": str(model_dir / "demo_xml.xml"),
                        },
                        "pickle": {
                            "success": True,
                            "export_file": str(model_dir / "demo_pickle.pkl"),
                        },
                    },
                }
            ],
            "summary": {
                "total_files": 1,
                "successful_exports": 1,
                "failed_exports": 0,
                "formats_generated": {"json": 1, "xml": 1, "pickle": 1},
            },
        }
        (out / "export_results.json").write_text(json.dumps(manifest))
        return out

    def test_valid_manifest(self, _manifest_dir: Path) -> None:
        from export.processor import validate_export_outputs

        result = validate_export_outputs(_manifest_dir)
        assert result["success"] is True
        assert result["checked"] == 3
        assert result["missing"] == []
        assert result["invalid"] == []

    def test_missing_manifest(self, tmp_path: Path) -> None:
        from export.processor import validate_export_outputs

        result = validate_export_outputs(tmp_path)
        assert result["success"] is False
        assert result["missing"]

    def test_missing_export_file(self, _manifest_dir: Path) -> None:
        from export.processor import validate_export_outputs

        # Delete one export file
        json_file = list(_manifest_dir.rglob("*_json.json"))[0]
        json_file.unlink()
        result = validate_export_outputs(_manifest_dir)
        assert result["success"] is False
        assert result["missing"]

    def test_corrupt_json_export(self, _manifest_dir: Path) -> None:
        from export.processor import validate_export_outputs

        json_file = list(_manifest_dir.rglob("*_json.json"))[0]
        json_file.write_text("NOT JSON")
        result = validate_export_outputs(_manifest_dir)
        assert result["success"] is False
        assert any("json" in e.get("format", "") for e in result["invalid"])

    def test_expected_formats_incomplete(self, _manifest_dir: Path) -> None:
        from export.processor import validate_export_outputs

        result = validate_export_outputs(
            _manifest_dir, expected_formats=["json", "xml", "graphml"]
        )
        assert result["success"] is False
        assert result["incomplete"]
        assert "graphml" in result["incomplete"][0]["missing_formats"]

    def test_empty_export_file(self, _manifest_dir: Path) -> None:
        from export.processor import validate_export_outputs

        json_file = list(_manifest_dir.rglob("*_json.json"))[0]
        json_file.write_text("")
        result = validate_export_outputs(_manifest_dir)
        assert result["success"] is False


# -- core.py silent-failure fix -----------------------------------------------


class TestCoreFailurePropagation:
    """export_gnn_files must record writer failures instead of silently
    counting them as successes (the bug fixed by _writer_success)."""

    def test_format_exporters_failure_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import logging

        from export import core as core_mod
        from export import format_exporters as fe
        from export.core import export_gnn_files

        # Make JSON writer return (False, "boom") -- the tuple-return path.

        def _failing_json(model: Any, path: Any) -> tuple:
            return False, "boom"

        # Patch both the module attribute and the local name already bound
        # in core.py's namespace (``from .format_exporters import ...``).
        monkeypatch.setattr(fe, "export_to_json_gnn", _failing_json)
        monkeypatch.setattr(core_mod, "export_to_json_gnn", _failing_json)
        # Ensure the other writers still work so only JSON fails.
        target = tmp_path / "input"
        target.mkdir()
        (target / "test.md").write_text("## ModelName\nTest\n## StateSpaceBlock\ns[2]")
        out = tmp_path / "output"
        ok = export_gnn_files(target, out, logging.getLogger("test"), recursive=False)
        # The file had a failing JSON export -> overall False (partial failure).
        assert ok is False
