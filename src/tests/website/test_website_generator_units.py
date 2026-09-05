"""Unit tests for website.generator internals.

Pins: the typed step catalogue, pure ``collect_website_data`` with an
injected MCP-tools provider, HTML escaping on every page, resilient
per-page writes, and the ``website_results.json`` manifest contract.
Deterministic and filesystem-only — no network, no live MCP registry.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _build_site(tmp_path: Any, **extra: Any) -> tuple[Path, dict[str, Any]]:
    """Build a site from an empty input dir with a deterministic MCP provider."""
    from website import WebsiteGenerator

    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    out = tmp_path / "site"
    result = WebsiteGenerator(mcp_tools_provider=list).generate_website(
        {
            "input_dir": str(input_dir),
            "output_dir": str(out),
            "pipeline_output_root": str(tmp_path),
            **extra,
        }
    )
    return out, result


class TestPipelineStepCatalogue:
    @pytest.mark.unit
    def test_catalogue_covers_steps_0_to_24(self) -> None:
        from website import get_pipeline_steps

        steps = get_pipeline_steps()
        assert [s.number for s in steps] == list(range(25))
        assert all(s.name and s.description for s in steps)

    @pytest.mark.unit
    def test_step_script_names_match_display_convention(self) -> None:
        from website import get_pipeline_steps

        steps = {s.number: s for s in get_pipeline_steps()}
        assert steps[20].script_name == "20_website.py"
        assert steps[3].script_name == "3_gnn_processing.py"

    @pytest.mark.unit
    def test_step_info_is_frozen(self) -> None:
        from website import StepInfo

        step = StepInfo(99, "Frozen", "cannot mutate")
        with pytest.raises((AttributeError, TypeError)):
            step.name = "mutated"  # type: ignore[misc]


class TestCollectWebsiteData:
    @pytest.mark.unit
    def test_provider_injection_controls_mcp_inventory(self, tmp_path: Any) -> None:
        from website import collect_website_data

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        assets = tmp_path / "assets"
        assets.mkdir()
        tools = [
            {
                "name": "fake_tool",
                "module": "synthetic",
                "category": "test",
                "desc": "injected",
            }
        ]
        data = collect_website_data(
            tmp_path, input_dir, assets, mcp_tools_provider=lambda: tools
        )
        assert data["mcp_tools"] == tools

    @pytest.mark.unit
    def test_discovers_gnn_files_and_step_statuses(self, tmp_path: Any) -> None:
        from website import collect_website_data

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "model.md").write_text("# model\n")
        (tmp_path / "03_gnn_output").mkdir()
        assets = tmp_path / "assets"
        assets.mkdir()

        data = collect_website_data(
            tmp_path, input_dir, assets, mcp_tools_provider=list
        )

        assert [f.name for f in data["gnn_files"]] == ["model.md"]
        assert data["processed_files"] == 1
        assert data["step_statuses"][3] == "ok"
        assert data["step_statuses"][20] == "pending"

    @pytest.mark.unit
    def test_reports_capped_per_dir_and_malformed_analysis_skipped(
        self, tmp_path: Any
    ) -> None:
        from website import collect_website_data

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

        data = collect_website_data(
            tmp_path, input_dir, assets, mcp_tools_provider=list
        )
        per_dir: dict[str, int] = {}
        for rep in data["reports"]:
            per_dir[rep["dir"]] = per_dir.get(rep["dir"], 0) + 1
        assert per_dir == {"07_export_output": 5, "16_analysis_output": 2}
        assert len(data["analysis"]) == 1  # malformed sibling skipped
        assert data["analysis"][0]["mean"] == "<b>&</b>"  # raw data preserved


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
        site, result = _build_site(
            tmp_path,
            mcp_tools=[{"name": "<t>", "module": "m&", "category": "c", "desc": "<d>"}],
        )
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
        site, result = _build_site(tmp_path, mcp_tools=["not-a-dict"])

        assert result["success"] is False  # errors present
        assert result["pages_created"] == 6
        assert "mcp.html" not in result["pages"]
        assert any("Failed to render mcp.html" in e for e in result["errors"])
        assert (site / "index.html").exists()  # rest of the site survived
        assert not (site / "mcp.html").exists()

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
        from website import process_website

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
        from website import process_website

        assert process_website(tmp_path / "nope", tmp_path / "out") is False


class TestEmbedEscaping:
    @pytest.mark.unit
    def test_markdown_content_is_escaped(self, tmp_path: Any) -> None:
        from website import embed_markdown_file

        md = tmp_path / "evil.md"
        md.write_text("<script>x</script>")
        out = tmp_path / "out.html"

        assert embed_markdown_file(md, out) is True
        content = out.read_text(encoding="utf-8")
        assert "<script>x</script>" not in content
        assert "&lt;script&gt;" in content

    @pytest.mark.unit
    def test_text_content_is_escaped(self, tmp_path: Any) -> None:
        from website import embed_text_file

        txt = tmp_path / "a.txt"
        txt.write_text("1 < 2 & 3 > 2")
        out = tmp_path / "out.html"

        assert embed_text_file(txt, out) is True
        content = out.read_text(encoding="utf-8")
        assert "1 &lt; 2 &amp; 3 &gt; 2" in content

    @pytest.mark.unit
    def test_image_src_attribute_is_escaped(self, tmp_path: Any) -> None:
        from website import embed_image

        img = tmp_path / "a&b.png"
        img.write_text("png")
        out = tmp_path / "out.html"

        assert embed_image(img, out) is True
        content = out.read_text(encoding="utf-8")
        assert 'src="' in content
        assert "a&amp;b.png" in content

    @pytest.mark.unit
    def test_html_embedding_stays_verbatim(self, tmp_path: Any) -> None:
        from website import embed_html_file

        src = tmp_path / "src.html"
        src.write_text("<p>keep &me</p>")
        out = tmp_path / "out.html"

        assert embed_html_file(src, out) is True
        assert "<p>keep &me</p>" in out.read_text(encoding="utf-8")

    @pytest.mark.unit
    def test_missing_sources_return_false(self, tmp_path: Any) -> None:

        from website import (
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
        import website
        from website import get_module_info

        assert get_module_info()["version"] == website.__version__

    @pytest.mark.unit
    def test_manifest_keys_preserved(self) -> None:
        from website import get_module_info

        info = get_module_info()
        for key in (
            "version",
            "description",
            "features",
            "supported_file_types",
            "embedding_capabilities",
        ):
            assert key in info
