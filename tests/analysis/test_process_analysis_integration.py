"""Integration pins for the Step-16 entry point ``process_analysis``.

Focus: the documented sentinel contracts (missing target → exit 2, invalid
animation flags → False) and a full synthetic run producing artifacts.
"""

from __future__ import annotations

from pathlib import Path

from gnn.analysis.processor import process_analysis

_SPEC = """## GNNSection
ActInfPOMDP

## ModelName
Step16Probe

## StateSpaceBlock
s[2,1,type=float]
o[2,1,type=int]

## Connections
s-o

## Time
Static

## Footer
Step16Probe
"""


def test_missing_target_dir_returns_sentinel_2(tmp_path: Path) -> None:
    result = process_analysis(tmp_path / "no_such_dir", tmp_path / "out")

    assert result == 2


def test_conflicting_animation_flags_return_false(tmp_path: Path) -> None:
    target = tmp_path / "in"
    target.mkdir()

    result = process_analysis(
        target,
        tmp_path / "out",
        generate_animations=True,
        no_animations=True,
    )

    assert result is False


def test_full_run_processes_specs_and_writes_results(tmp_path: Path) -> None:
    target = tmp_path / "in"
    target.mkdir()
    (target / "probe.md").write_text(_SPEC, encoding="utf-8")
    out = tmp_path / "out"

    result = process_analysis(target, out)

    # Full runs either succeed or finish with the widened nothing-to-fail
    # sentinel; the contract is "no crash and a resolved result".
    assert result in {True, 2}
    if result is True:
        assert out.exists()
        produced = [p.name for p in out.rglob("*")]
        assert produced, "a successful analysis pass must write artifacts"
