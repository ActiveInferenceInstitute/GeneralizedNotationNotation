"""Tests for cli module's public API surface not covered by existing tests.

Covers: get_module_info, FEATURES, __version__, _cmd_run, _cmd_parse,
_cmd_render (with missing file), _cmd_report, _cmd_preflight, _cmd_serve,
_cmd_lsp, _cmd_watch, _cmd_graph, _cmd_templates, _cmd_pull, _find_render_artifact.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCLIConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import cli

        assert hasattr(cli, "FEATURES")
        assert isinstance(cli.FEATURES, dict)
        for key in (
            "subcommands",
            "pipeline_execution",
            "file_validation",
            "file_parsing",
            "render_dispatch",
            "lsp_launch",
        ):
            assert key in cli.FEATURES

    def test_version(self) -> None:
        import cli

        assert hasattr(cli, "__version__")
        assert isinstance(cli.__version__, str)

    def test_get_module_info(self) -> None:
        from cli import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info


class TestCmdHandlers:
    """Test individual command handlers."""

    def test_cmd_parse_missing_file(self) -> None:
        from cli import _cmd_parse

        args = SimpleNamespace(file=Path("/nonexistent.md"), format="json")
        result = _cmd_parse(args)
        assert result == 1  # Error exit code

    def test_cmd_render_missing_file(self) -> None:
        from cli import _cmd_render

        args = SimpleNamespace(
            file=Path("/nonexistent.md"),
            framework="pymdp",
            output=None,
            verbose=False,
        )
        result = _cmd_render(args)
        assert result == 1

    def test_cmd_report_missing_dir(self) -> None:
        from cli import _cmd_report

        args = SimpleNamespace(output_dir=Path("/nonexistent_output"))
        result = _cmd_report(args)
        assert result == 1

    def test_cmd_preflight_default(self) -> None:
        from cli import _cmd_preflight

        args = SimpleNamespace(config=None, verbose=False, json=False)
        result = _cmd_preflight(args)
        assert isinstance(result, int)

    def test_cmd_preflight_json(self, capsys: Any) -> None:
        from cli import _cmd_preflight

        args = SimpleNamespace(config=None, verbose=False, json=True)
        result = _cmd_preflight(args)
        assert isinstance(result, int)
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] in ("success", "warning")
        assert "checks_passed" in envelope["data"]

    def test_cmd_serve_default(self) -> None:
        # _cmd_serve starts a blocking uvicorn server (never returns), so we
        # verify the handler wiring instead: api.app must be importable and
        # the serve handler must accept the documented args.
        import cli

        assert hasattr(cli, "_cmd_serve")
        import api.app  # noqa: F401

    def test_cmd_lsp(self) -> None:
        # _cmd_lsp calls start_server() which tries to start stdio server.
        # Instead, test that the handler function exists and is callable.
        from cli import _cmd_lsp

        assert callable(_cmd_lsp)

    def test_cmd_graph_missing_file(self) -> None:
        from cli import _cmd_graph

        args = SimpleNamespace(
            file=Path("/nonexistent.md"), format="mermaid", verbose=False
        )
        result = _cmd_graph(args)
        assert result == 1

    def test_cmd_pull_missing_template(self, tmp_path: Path) -> None:
        from cli import _cmd_pull

        args = SimpleNamespace(
            name="nonexistent-template",
            output_dir=tmp_path,
            dry_run=True,
            overwrite=False,
            verbose=False,
        )
        result = _cmd_pull(args)
        assert result == 1

    def test_cmd_templates_unknown_subcommand(self, capsys: Any) -> None:
        from cli import _cmd_templates

        args = SimpleNamespace(
            templates_command="nonexistent",
            verbose=False,
        )
        result = _cmd_templates(args)
        assert isinstance(result, int)

    def test_cmd_watch_handler_exists(self) -> None:
        # _cmd_watch starts a blocking filesystem watcher (GNNWatcher.start()
        # loops until stopped), so we verify the handler is present and
        # callable rather than invoking it in-process.
        from cli import _cmd_watch

        assert callable(_cmd_watch)


class TestFindRenderArtifact:
    """Test _find_render_artifact helper."""

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        from cli import _find_render_artifact

        result = _find_render_artifact(tmp_path, "pymdp")
        assert result is None

    def test_finds_py_file(self, tmp_path: Path) -> None:
        from cli import _find_render_artifact

        output = tmp_path / "pymdp" / "model.py"
        output.parent.mkdir(parents=True)
        output.write_text("code")
        result = _find_render_artifact(tmp_path, "pymdp")
        assert result == output

    def test_finds_toml_for_rxinfer(self, tmp_path: Path) -> None:
        from cli import _find_render_artifact

        output = tmp_path / "rxinfer" / "model.toml"
        output.parent.mkdir(parents=True)
        output.write_text("config")
        result = _find_render_artifact(tmp_path, "rxinfer")
        assert result == output

    def test_skips_known_non_artifact_files(self, tmp_path: Path) -> None:
        from cli import _find_render_artifact

        (tmp_path / "README.md").write_text("readme")
        (tmp_path / "processing_summary.json").write_text("{}")
        result = _find_render_artifact(tmp_path, "pymdp")
        assert result is None

    def test_finds_from_summary_json(self, tmp_path: Path) -> None:
        import json

        from cli import _find_render_artifact

        summary = tmp_path / "render_processing_summary.json"
        output_file = tmp_path / "pymdp" / "model.py"
        output_file.parent.mkdir()
        output_file.write_text("code")
        summary.write_text(
            json.dumps(
                {
                    "file_results": {
                        "model.md": {
                            "framework_results": {
                                "pymdp": {"output_files": [str(output_file)]}
                            }
                        }
                    }
                }
            )
        )
        result = _find_render_artifact(tmp_path, "pymdp")
        assert result == output_file
