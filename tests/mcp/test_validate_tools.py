"""Unit tests for ``gnn.mcp.validate_tools`` (the audit script's ``main()``).

``main()`` previously had zero test coverage: the MCP init-failure exit, the
per-tool ``NOT_CALLABLE`` / ``UNDOCUMENTED`` classification, the callability
spot-check branches (SKIP / ``success=False`` / exception), and the
``logging_miss`` ``register_tools`` scan were all unpinned. These tests
isolate ``main()`` from the live MCP registry by stubbing
``gnn.mcp.initialize`` / ``gnn.mcp.mcp_instance`` and pointing ``SRC_ROOT``
at a tmp tree so the audit report is written to tmp instead of the repo.

Deliberately unmarked: the audit is offline logic exercised by the default
suite (the ``mcp`` marker would deselect it locally).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gnn.mcp import validate_tools  # noqa: E402


def _install_fake_mcp(
    monkeypatch: pytest.MonkeyPatch,
    tools: dict[str, Any],
    modules: dict[str, Any] | None = None,
    execute_tool_fn: Callable[[str], Any] | None = None,
) -> SimpleNamespace:
    """Stub ``gnn.mcp.initialize``/``mcp_instance`` for one ``main()`` call."""

    def _default_execute_tool(name: str, args: dict[str, Any]) -> Any:
        return {"success": False, "error": "unconfigured"}

    fake = SimpleNamespace(
        tools=tools,
        modules=modules or {},
        execute_tool=execute_tool_fn or _default_execute_tool,
    )

    def initialize(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gnn.mcp.initialize", initialize)
    monkeypatch.setattr("gnn.mcp.mcp_instance", fake)
    return fake


def _make_tool(
    *,
    func: Any = None,
    description: str = "",
    module: str = "tests",
    category: str = "test",
) -> Any:
    return SimpleNamespace(
        func=func,
        function=None,
        description=description,
        module=module,
        category=category,
    )


def _real_func() -> dict[str, Any]:
    return {"success": True}


def _seed_src_tree(
    tmp_path: Path, good_modules: list[str], bad_modules: list[str]
) -> Path:
    """Create the ``<SRC_ROOT>/gnn/<mod>/mcp.py`` tree the logging scan reads."""
    src_root = tmp_path / "src"
    (src_root / "gnn" / "mcp").mkdir(parents=True)
    for name in good_modules:
        d = src_root / "gnn" / name
        d.mkdir()
        (d / "mcp.py").write_text(
            "def register_tools(mcp):\n    logger.info('registered')\n",
            encoding="utf-8",
        )
    for name in bad_modules:
        d = src_root / "gnn" / name
        d.mkdir()
        (d / "mcp.py").write_text(
            "def register_tools(mcp):\n    pass\n", encoding="utf-8"
        )
    return src_root


def _read_report(src_root: Path) -> dict[str, Any]:
    report_path = src_root / "gnn" / "mcp" / "audit_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.unit
class TestValidateToolsMain:
    def test_clean_audit_returns_zero_and_writes_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=["alpha_mod"], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "tool_a": _make_tool(func=_real_func, description="Does A"),
            "list_analysis_tools": _make_tool(
                func=_real_func, description="Lists analysis tools"
            ),
            "list_render_frameworks": _make_tool(
                func=_real_func, description="Lists render frameworks"
            ),
        }
        _install_fake_mcp(
            monkeypatch,
            tools,
            modules={"alpha_mod": SimpleNamespace(status="loaded", tools_count=3)},
            execute_tool_fn=lambda name, args: {"success": True},
        )

        rc = validate_tools.main()

        assert rc == 0
        report = _read_report(src_root)
        assert report["tools_total"] == 3
        assert report["modules_loaded"] == 1
        assert report["modules_errored"] == 0
        assert report["issues"] == []
        # Two of the 14 spot-check names are registered and succeed; the
        # other 12 hit the SKIP (not registered) branch.
        assert report["spot_checks_ok"] == 2
        assert report["spot_checks_err"] == 0
        assert report["logging_ok"] == 1
        assert report["logging_miss"] == 0
        out = capsys.readouterr().out
        assert "PASS - all tools real, documented, logged" in out
        assert "SKIP (not registered)" in out

    def test_mcp_init_failure_returns_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)

        def failing_initialize(**kwargs: Any) -> None:
            raise RuntimeError("sdk missing")

        monkeypatch.setattr("gnn.mcp.initialize", failing_initialize)

        rc = validate_tools.main()

        assert rc == 1
        out = capsys.readouterr().out
        assert "FATAL: Could not initialize MCP: sdk missing" in out

    def test_not_callable_and_undocumented_classified_as_issues(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "lambda_tool": _make_tool(func=lambda: {}, description="lambda tool"),
            "none_tool": _make_tool(func=None, description="no func"),
            "bare_tool": _make_tool(func=_real_func, description="   "),
            "ok_tool": _make_tool(func=_real_func, description="Real tool"),
        }
        _install_fake_mcp(monkeypatch, tools)

        rc = validate_tools.main()

        assert rc == 1
        issues = {
            (issue["tool"], issue["issue"])
            for issue in _read_report(src_root)["issues"]
        }
        assert ("lambda_tool", "not a real callable") in issues
        assert ("none_tool", "not a real callable") in issues
        assert ("bare_tool", "missing description") in issues

    def test_spot_check_failure_and_exception_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "list_analysis_tools": _make_tool(
                func=_real_func, description="returns success=False"
            ),
            "list_render_frameworks": _make_tool(func=_real_func, description="raises"),
        }

        def execute_tool(name: str, args: dict[str, Any]) -> Any:
            if name == "list_render_frameworks":
                raise RuntimeError("spot crash")
            return {"success": False, "error": "returned-false"}

        _install_fake_mcp(monkeypatch, tools, execute_tool_fn=execute_tool)

        rc = validate_tools.main()

        assert rc == 1
        report = _read_report(src_root)
        issues = {(issue["tool"], issue["issue"]) for issue in report["issues"]}
        assert ("list_analysis_tools", "execute returned success=False") in issues
        assert ("list_render_frameworks", "exception: spot crash") in issues
        assert report["spot_checks_ok"] == 0
        assert report["spot_checks_err"] == 2

    def test_logging_miss_issue_recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        src_root = _seed_src_tree(
            tmp_path, good_modules=["alpha_mod"], bad_modules=["beta_mod"]
        )
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {"ok_tool": _make_tool(func=_real_func, description="Real tool")}
        _install_fake_mcp(monkeypatch, tools)

        rc = validate_tools.main()

        assert rc == 1
        report = _read_report(src_root)
        issues = {(issue["tool"], issue["issue"]) for issue in report["issues"]}
        assert ("beta_mod/mcp.py", "register_tools() has no logger.info") in issues
        assert report["logging_ok"] == 1
        assert report["logging_miss"] == 1
