"""Unit tests for gnn.website.generator internals.

Pins: the typed step catalogue, pure ``collect_website_data`` sourcing the
MCP page from the step-21 artifacts (``21_mcp_output/mcp_processing_summary.json``
and ``registered_tools.json``) with truthful empty states, HTML escaping on
every page, resilient per-page writes, and the ``website_results.json``
manifest contract. Deterministic and filesystem-only — no network.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _build_site(tmp_path: Any, **extra: Any) -> tuple[Path, dict[str, Any]]:
    """Build a site from an empty input dir."""
    from gnn.website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    out = tmp_path / "site"
    result = WebsiteGenerator().generate_website(
        {
            "input_dir": str(input_dir),
            "output_dir": str(out),
            "pipeline_output_root": str(tmp_path),
            **extra,
        }
    )
    return out, result


_SUCCESS_SUMMARY: dict[str, Any] = {
    "timestamp": "2026-09-19T10:00:00",
    "target_dir": "/tmp/pipeline/input",
    "output_dir": "/tmp/pipeline/21_mcp_output",
    "processing_status": "completed",
    "mcp_version": "1.0.0",
    "tools_registered": 3,
    "registered_modules_count": 2,
    "registered_modules": ["gnn.website", "gnn.export"],
    "resources_count": 4,
    "message": "MCP processing completed - 3 tools registered from 2 modules",
}

_REGISTERED_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_website_status",
        "description": "Report website generation status",
        "module": "gnn.website",
        "category": "status",
        "version": "1.0.0",
    },
    {
        "name": "export_model",
        "description": "Export a GNN model",
        "module": "gnn.export",
        "category": "export",
        "version": "1.0.0",
    },
    {
        "name": "list_files",
        "description": "List discovered GNN files",
        "module": "gnn.website",
        "category": "query",
        "version": "1.0.0",
    },
]


def _write_mcp_artifacts(
    tmp_path: Any,
    *,
    summary: dict[str, Any] | None = None,
    tools: list[Any] | None = None,
) -> Path:
    """Write step-21 MCP artifacts (summary and/or registered tools)."""
    mcp_dir = tmp_path / "21_mcp_output"
    mcp_dir.mkdir(exist_ok=True)
    if summary is not None:
        (mcp_dir / "mcp_processing_summary.json").write_text(json.dumps(summary))
    if tools is not None:
        (mcp_dir / "registered_tools.json").write_text(json.dumps(tools))
    return mcp_dir


def _mcp_tools_stat(index_html: str) -> str:
    """Extract the dashboard MCP Tools stat value from index.html."""
    m = re.search(r'MCP Tools</div>\s*<div class="value">(\d+)</div>', index_html)
    assert m is not None, "MCP Tools stat card missing from index.html"
    return m.group(1)


class TestPipelineStepCatalogue:
    @pytest.mark.unit
    def test_catalogue_covers_steps_0_to_24(self) -> None:
        from gnn.website import get_pipeline_steps

        steps = get_pipeline_steps()
        assert [s.number for s in steps] == list(range(25))
        assert all(s.name and s.description for s in steps)

    @pytest.mark.unit
    def test_step_script_names_match_display_convention(self) -> None:
        from gnn.website import get_pipeline_steps

        steps = {s.number: s for s in get_pipeline_steps()}
        assert steps[20].script_name == "20_website.py"
        assert steps[3].script_name == "3_gnn_processing.py"

    @pytest.mark.unit
    def test_step_info_is_frozen(self) -> None:
        from gnn.website import StepInfo

        step = StepInfo(99, "Frozen", "cannot mutate")
        with pytest.raises((AttributeError, TypeError)):
            step.name = "mutated"  # type: ignore[misc]


class TestCollectWebsiteData:
    @pytest.mark.unit
    def test_discovers_gnn_files_and_step_statuses(self, tmp_path: Any) -> None:
        from gnn.website import collect_website_data

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "model.md").write_text("# model\n")
        (tmp_path / "03_gnn_output").mkdir()
        assets = tmp_path / "assets"
        assets.mkdir()

        data = collect_website_data(tmp_path, input_dir, assets)

        assert [f.name for f in data["gnn_files"]] == ["model.md"]
        assert data["processed_files"] == 1
        assert data["step_statuses"][3] == "ok"
        assert data["step_statuses"][20] == "pending"

    @pytest.mark.unit
    def test_reports_capped_per_dir_and_malformed_analysis_skipped(
        self, tmp_path: Any
    ) -> None:
        from gnn.website import collect_website_data

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        out16 = tmp_path / "16_analysis_output"
        results = out16 / "analysis_results"
        results.mkdir(parents=True)
        (results / "good.json").write_text('{"file_name": "a", "mean": "<b>&</b>"}')
        (out16 / "bad.json").write_text("{not json")
        step_dir = tmp_path / "07_export_output"
        step_dir.mkdir()
        for i in range(7):
            (step_dir / f"r{i}.json").write_text('{"k": %d}' % i)
        assets = tmp_path / "assets"
        assets.mkdir()

        data = collect_website_data(tmp_path, input_dir, assets)
        per_dir: dict[str, int] = {}
        for rep in data["reports"]:
            per_dir[rep["dir"]] = per_dir.get(rep["dir"], 0) + 1
        assert per_dir == {"07_export_output": 5, "16_analysis_output": 2}
        assert len(data["analysis"]) == 1  # malformed sibling skipped
        assert data["analysis"][0]["mean"] == "<b>&</b>"  # raw data preserved


class TestMcpArtifactSourcing:
    """The MCP page reads real data from step-21 artifacts, never a live registry."""

    @pytest.mark.unit
    def test_mcp_page_renders_recorded_summary_and_tool_cards(
        self, tmp_path: Any
    ) -> None:
        _write_mcp_artifacts(
            tmp_path, summary=_SUCCESS_SUMMARY, tools=_REGISTERED_TOOLS
        )

        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "mcp.html").read_text(encoding="utf-8")
        assert "No MCP tools registered. Run step 21" not in page
        assert "3 tools registered across all modules" in page
        assert "MCP processing completed - 3 tools registered from 2 modules" in page
        assert "completed" in page
        assert "gnn.website" in page
        assert "gnn.export" in page
        for tool_name in ("get_website_status", "export_model", "list_files"):
            assert tool_name in page

        index = (site / "index.html").read_text(encoding="utf-8")
        assert _mcp_tools_stat(index) == "3"

    @pytest.mark.unit
    def test_mcp_page_truthful_when_no_artifacts_were_written(
        self, tmp_path: Any
    ) -> None:
        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "mcp.html").read_text(encoding="utf-8")
        assert "No MCP data recorded" in page
        assert "did not run or wrote no MCP output" in page
        assert "No MCP tools registered" not in page
        assert 'class="tool-name"' not in page

        index = (site / "index.html").read_text(encoding="utf-8")
        assert _mcp_tools_stat(index) == "0"

    @pytest.mark.unit
    def test_failed_summary_is_shown_truthfully(self, tmp_path: Any) -> None:
        failed = dict(_SUCCESS_SUMMARY)
        failed.update(
            {
                "processing_status": "failed",
                "tools_registered": 0,
                "registered_modules": [],
                "registered_modules_count": 0,
                "message": "MCP processing failed: registry import boom",
                "error": "registry import boom",
            }
        )
        _write_mcp_artifacts(tmp_path, summary=failed)

        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "mcp.html").read_text(encoding="utf-8")
        assert "MCP processing failed: registry import boom" in page
        assert "registry import boom" in page
        assert "failed" in page
        assert "No MCP tools registered" not in page
        assert 'class="tool-name"' not in page  # no tool cards fabricated

    @pytest.mark.unit
    def test_malformed_summary_degrades_to_truthful_empty_state(
        self, tmp_path: Any
    ) -> None:
        mcp_dir = tmp_path / "21_mcp_output"
        mcp_dir.mkdir()
        (mcp_dir / "mcp_processing_summary.json").write_text("{definitely not json")

        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "mcp.html").read_text(encoding="utf-8")
        assert "No MCP data recorded" in page
        assert "No MCP tools registered" not in page

        index = (site / "index.html").read_text(encoding="utf-8")
        assert _mcp_tools_stat(index) == "0"

    @pytest.mark.unit
    def test_collect_website_data_maps_registered_tools_to_cards(
        self, tmp_path: Any
    ) -> None:
        from gnn.website import collect_website_data

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        assets = tmp_path / "assets"
        assets.mkdir()
        raw_tools: list[dict[str, Any] | str] = [
            {
                "name": "get_website_status",
                "description": "Report website generation status",
                "module": "gnn.website",
                "category": "status",
                "version": "1.0.0",  # tolerated, dropped from the card shape
            },
            {
                "name": "legacy_tool",
                "desc": "legacy desc field",
                "module": "gnn.export",
                "category": "export",
            },
            {"name": "minimal_tool", "module": "gnn.core"},
            "junk-not-a-dict",  # skipped
        ]
        _write_mcp_artifacts(tmp_path, summary=_SUCCESS_SUMMARY, tools=raw_tools)

        data = collect_website_data(tmp_path, input_dir, assets)
        assert data["mcp_summary"] == _SUCCESS_SUMMARY
        assert data["mcp_tools"] == [
            {
                "name": "get_website_status",
                "module": "gnn.website",
                "desc": "Report website generation status",
                "category": "status",
            },
            {
                "name": "legacy_tool",
                "module": "gnn.export",
                "desc": "legacy desc field",
                "category": "export",
            },
            {
                "name": "minimal_tool",
                "module": "gnn.core",
                "desc": "",
                "category": "",
            },
        ]

    @pytest.mark.unit
    def test_collect_website_data_empty_without_mcp_artifacts(
        self, tmp_path: Any
    ) -> None:
        from gnn.website import collect_website_data

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        assets = tmp_path / "assets"
        assets.mkdir()

        data = collect_website_data(tmp_path, input_dir, assets)
        assert data["mcp_summary"] == {}
        assert data["mcp_tools"] == []


class TestPageEscaping:
    @pytest.mark.unit
    def test_gnn_file_content_is_escaped(self, tmp_path: Any) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "evil.md").write_text("<script>alert(1)</script> & <b>x</b>\n")

        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "gnn_files.html").read_text(encoding="utf-8")
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
        assert "<script>alert(1)</script>" not in page

    @pytest.mark.unit
    def test_analysis_values_are_escaped(self, tmp_path: Any) -> None:
        out16 = tmp_path / "16_analysis_output"
        out16.mkdir()
        (out16 / "a.json").write_text('{"file_name": "a", "v": "<img src=x>"}')

        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "analysis.html").read_text(encoding="utf-8")
        assert "<img src=x>" not in page
        assert "&lt;img src=x&gt;" in page

    @pytest.mark.unit
    def test_mcp_tool_fields_are_escaped(self, tmp_path: Any) -> None:
        _write_mcp_artifacts(
            tmp_path,
            summary=_SUCCESS_SUMMARY,
            tools=[{"name": "<t>", "module": "m&", "category": "c", "desc": "<d>"}],
        )
        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "mcp.html").read_text(encoding="utf-8")
        assert "&lt;t&gt;" in page
        assert "<t>" not in page

    @pytest.mark.unit
    def test_report_content_is_escaped(self, tmp_path: Any) -> None:
        step_dir = tmp_path / "07_export_output"
        step_dir.mkdir()
        (step_dir / "r.json").write_text('{"k": "<u>"}')

        site, result = _build_site(tmp_path)
        assert result["success"] is True
        page = (site / "reports.html").read_text(encoding="utf-8")
        assert "&lt;u&gt;" in page


class TestResilientPageWrites:
    @pytest.mark.unit
    def test_one_bad_page_does_not_destroy_the_site(self, tmp_path: Any) -> None:
        out16 = tmp_path / "16_analysis_output"
        out16.mkdir()
        # Loads fine at collection time, crashes the analysis page renderer.
        (out16 / "bad.json").write_text('["not", "a", "dict"]')

        site, result = _build_site(tmp_path)

        assert result["success"] is False  # errors present
        assert result["pages_created"] == 6
        assert "analysis.html" not in result["pages"]
        assert any("Failed to render analysis.html" in e for e in result["errors"])
        assert (site / "index.html").exists()  # rest of the site survived
        assert not (site / "analysis.html").exists()

    @pytest.mark.unit
    def test_clean_run_reports_all_pages(self, tmp_path: Any) -> None:
        _, result = _build_site(tmp_path)

        assert result["success"] is True
        assert result["pages_created"] == 7
        assert len(result["pages"]) == 7
        assert result["errors"] == []


class TestProcessWebsiteManifest:
    @pytest.mark.unit
    def test_manifest_records_the_run(self, tmp_path: Any) -> None:
        from gnn.website import process_website

        target = tmp_path / "input"
        target.mkdir()
        out = tmp_path / "out"

        result = process_website(
            target_dir=target,
            output_dir=out,
            verbose=False,
            logger=logging.getLogger("t"),
            recursive=False,
            website_html_filename="ignored.html",
        )
        assert result is True

        manifest = json.loads((out / "website_results.json").read_text())
        assert manifest["success"] is True
        assert manifest["pages_created"] == 7
        assert set(manifest["pages"]) == {
            "index.html",
            "pipeline.html",
            "gnn_files.html",
            "analysis.html",
            "visualization.html",
            "reports.html",
            "mcp.html",
        }
        assert manifest["errors"] == []
        assert "generated_at" in manifest

    @pytest.mark.unit
    def test_missing_target_returns_false(self, tmp_path: Any) -> None:
        from gnn.website import process_website

        assert process_website(tmp_path / "nope", tmp_path / "out") is False


class TestEmbedEscaping:
    @pytest.mark.unit
    def test_markdown_content_is_escaped(self, tmp_path: Any) -> None:
        from gnn.website import embed_markdown_file

        md = tmp_path / "evil.md"
        md.write_text("<script>x</script>")
        out = tmp_path / "out.html"

        assert embed_markdown_file(md, out) is True
        content = out.read_text(encoding="utf-8")
        assert "<script>x</script>" not in content
        assert "&lt;script&gt;" in content

    @pytest.mark.unit
    def test_text_content_is_escaped(self, tmp_path: Any) -> None:
        from gnn.website import embed_text_file

        txt = tmp_path / "a.txt"
        txt.write_text("1 < 2 & 3 > 2")
        out = tmp_path / "out.html"

        assert embed_text_file(txt, out) is True
        content = out.read_text(encoding="utf-8")
        assert "1 &lt; 2 &amp; 3 &gt; 2" in content

    @pytest.mark.unit
    def test_image_src_attribute_is_escaped(self, tmp_path: Any) -> None:
        from gnn.website import embed_image

        img = tmp_path / "a&b.png"
        img.write_text("png")
        out = tmp_path / "out.html"

        assert embed_image(img, out) is True
        content = out.read_text(encoding="utf-8")
        assert 'src="' in content
        assert "a&amp;b.png" in content

    @pytest.mark.unit
    def test_html_embedding_stays_verbatim(self, tmp_path: Any) -> None:
        from gnn.website import embed_html_file

        src = tmp_path / "src.html"
        src.write_text("<p>keep &me</p>")
        out = tmp_path / "out.html"

        assert embed_html_file(src, out) is True
        assert "<p>keep &me</p>" in out.read_text(encoding="utf-8")

    @pytest.mark.unit
    def test_missing_sources_return_false(self, tmp_path: Any) -> None:

        from gnn.website import (
            embed_html_file,
            embed_image,
            embed_json_file,
            embed_markdown_file,
            embed_text_file,
        )

        missing = tmp_path / "missing.bin"
        out = tmp_path / "out.html"
        assert embed_image(missing, out) is False
        assert embed_markdown_file(missing, out) is False
        assert embed_text_file(missing, out) is False
        assert embed_json_file(missing, out) is False
        assert embed_html_file(missing, out) is False


class TestModuleInfoAlignment:
    @pytest.mark.unit
    def test_reported_version_matches_package_version(self) -> None:
        import gnn.website as website
        from gnn.website import get_module_info

        assert get_module_info()["version"] == website.__version__

    @pytest.mark.unit
    def test_manifest_keys_preserved(self) -> None:
        from gnn.website import get_module_info

        info = get_module_info()
        for key in (
            "version",
            "description",
            "features",
            "supported_file_types",
            "embedding_capabilities",
        ):
            assert key in info
