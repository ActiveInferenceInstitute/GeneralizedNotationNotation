#!/usr/bin/env python3
"""Pin execute.planning.plan_execute dry-run behavior.

``plan_execute`` composes the same discovery/contract primitives as
``process_execute`` but runs no scripts. These tests build minimal render
output trees under ``tmp_path`` using the real Step 11
``render_processing_summary.json`` contract shape (``file_results`` →
``framework_results`` → per-framework ``success``/``output_files``) and
assert the plan's status, script dispositions, and contract fields —
deterministically, no network.
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from execute.planning import plan_execute  # noqa: E402


def _write_render_summary(
    render_dir: Path,
    file_results: dict,
    *,
    render_failures_block: dict | None = None,
) -> None:
    """Write a render_processing_summary.json with the real contract shape."""
    render_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "render_summary_version": "render_summary_v1",
        "file_results": file_results,
    }
    if render_failures_block is not None:
        payload["render_failures"] = render_failures_block
    (render_dir / "render_processing_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _make_script(render_dir: Path, model: str, framework: str) -> Path:
    """Create a fake rendered .py script under <render>/<model>/<framework>/."""
    d = render_dir / model / framework
    d.mkdir(parents=True, exist_ok=True)
    s = d / f"{model}_{framework}.py"
    s.write_text("print('hi')\n", encoding="utf-8")
    return s


def _file_result(
    target_dir: Path,
    source_name: str,
    framework: str,
    output_files: list[Path],
    success: bool = True,
) -> dict:
    """One file_results entry for the contract, with an absolute source path."""
    source_path = target_dir / source_name
    return {
        str(source_path): {
            "framework_results": {
                framework: {
                    "success": success,
                    "output_files": [str(p) for p in output_files],
                }
            }
        }
    }


def test_plan_execute_invalid_frameworks_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plan_execute(tmp_path, tmp_path, frameworks="not_a_real_framework")


def test_plan_execute_no_render_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate from the repo's real output/ so priority-4 cwd globs find nothing.
    monkeypatch.chdir(tmp_path)
    plan = plan_execute(tmp_path, tmp_path, frameworks="all")
    assert plan["status"] == "no_render_output"
    assert plan["total_scripts"] == 0
    assert plan["would_execute"] == []


def test_plan_execute_no_executable_scripts(tmp_path: Path) -> None:
    render_dir = tmp_path / "11_render_output"
    _write_render_summary(render_dir, {})
    plan = plan_execute(
        tmp_path, tmp_path, frameworks="pymdp", render_output_dir=render_dir
    )
    assert plan["status"] == "no_executable_scripts"
    assert plan["render_contract_found"] is True
    assert plan["total_scripts"] == 0


def test_plan_execute_classifies_pymdp_script(tmp_path: Path) -> None:
    render_dir = tmp_path / "11_render_output"
    script = _make_script(render_dir, "model_a", "pymdp")
    _write_render_summary(
        render_dir, _file_result(tmp_path, "model_a.md", "pymdp", [script])
    )
    plan = plan_execute(
        tmp_path, tmp_path, frameworks="pymdp", render_output_dir=render_dir
    )
    assert plan["status"] == "ready"
    assert plan["total_scripts"] == 1
    assert plan["render_contract_found"] is True
    assert len(plan["would_execute"]) + len(plan["would_skip_dependency"]) == 1
    # pymdp may or may not be importable in the test env; classify accordingly.
    bucket = (
        plan["would_execute"]
        if plan["would_execute"]
        else plan["would_skip_dependency"]
    )
    entry = bucket[0]
    assert entry["script_name"] == "model_a_pymdp.py"
    assert entry["framework"] == "pymdp"
    assert entry["model_name"] == "model_a"
    assert entry["script_path"] == str(script)


def test_plan_execute_reports_missing_render_scripts(tmp_path: Path) -> None:
    render_dir = tmp_path / "11_render_output"
    # Contract references a script path that does not exist on disk.
    ghost = render_dir / "model_c" / "pymdp" / "model_c_pymdp.py"
    _write_render_summary(
        render_dir, _file_result(tmp_path, "model_c.md", "pymdp", [ghost])
    )
    plan = plan_execute(
        tmp_path, tmp_path, frameworks="pymdp", render_output_dir=render_dir
    )
    assert plan["status"] == "no_executable_scripts"
    assert plan["render_contract_found"] is True
    assert (
        str(ghost.resolve())
        in [str(Path(p).resolve()) for p in plan["missing_render_scripts"]]
        or str(ghost) in plan["missing_render_scripts"]
    )


def test_plan_execute_render_failures_forwarded(tmp_path: Path) -> None:
    render_dir = tmp_path / "11_render_output"
    _write_render_summary(
        render_dir,
        {
            str(tmp_path / "broken.md"): {
                "framework_results": {
                    "pymdp": {"success": False, "message": "bad matrix"},
                }
            }
        },
    )
    plan = plan_execute(
        tmp_path, tmp_path, frameworks="pymdp", render_output_dir=render_dir
    )
    assert plan["render_failures"]
    assert plan["render_failures"][0]["file"] == "broken.md"
    assert plan["render_failures"][0]["framework"] == "pymdp"
    assert plan["render_failures"][0]["message"] == "bad matrix"


def test_plan_execute_unsupported_framework_omitted_from_contract(
    tmp_path: Path,
) -> None:
    render_dir = tmp_path / "11_render_output"
    # A framework the renderer declared unsupported → not an execution candidate.
    _write_render_summary(
        render_dir,
        {
            str(tmp_path / "cont.md"): {
                "framework_results": {
                    "pymdp": {"unsupported": True, "success": False},
                }
            }
        },
    )
    plan = plan_execute(
        tmp_path, tmp_path, frameworks="pymdp", render_output_dir=render_dir
    )
    assert plan["render_contract_found"] is True
    assert plan["total_scripts"] == 0
    # Unsupported is not a failure either.
    assert plan["render_failures"] == []
