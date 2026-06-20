#!/usr/bin/env python3
"""Tests for pipeline/run_session.py — resumable run sessions (no mocks)."""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.run_session import (  # noqa: E402
    RunSession,
    UnitStatus,
    WorkUnit,
    cancel_safe_cleanup,
    checkpoint,
    load_session,
    mark,
    remaining_units,
    resume_plan,
    start_session,
    status_report,
)


def test_start_session_deterministic_hash() -> Any:
    s1 = start_session("sess", ["b", "a", "c"])
    s2 = start_session("sess", ["c", "a", "b"])
    # Hash is order-independent (sorted unit ids).
    assert s1.run_hash == s2.run_hash
    assert s1.run_hash != ""
    assert len(s1.run_hash) == 12
    assert [u.unit_id for u in s1.units] == ["b", "a", "c"]
    assert all(u.status == UnitStatus.PENDING for u in s1.units)


def test_start_session_accepts_workunits() -> Any:
    wu = WorkUnit(unit_id="x", steps=[1, 2], status=UnitStatus.RUNNING)
    s = start_session("sess", [wu, "y"], created_by="tester")
    assert s.created_by == "tester"
    assert s.units[0].steps == [1, 2]
    # The session holds a copy — mutating the source does not bleed in.
    wu.steps.append(99)
    assert s.units[0].steps == [1, 2]


def test_checkpoint_roundtrip_and_remaining(tmp_path: Path) -> Any:
    s = start_session("sess", ["a", "b", "c"], created_by="me")
    s = mark(s, "a", UnitStatus.DONE)
    s = mark(s, "b", UnitStatus.DONE)

    path = tmp_path / "manifest.json"
    checkpoint(s, path)
    loaded = load_session(path)

    assert loaded == s
    assert isinstance(loaded, RunSession)
    # Only the not-done unit remains.
    assert remaining_units(loaded) == ["c"]
    assert resume_plan(loaded) == ["c"]


def test_mark_is_non_destructive() -> Any:
    s = start_session("sess", ["a", "b"])
    s2 = mark(s, "a", UnitStatus.DONE, artifact_refs=["out/a.json"])
    # Original untouched.
    assert s.units[0].status == UnitStatus.PENDING
    assert s.units[0].artifact_refs == []
    # Copy updated.
    assert s2.units[0].status == UnitStatus.DONE
    assert s2.units[0].artifact_refs == ["out/a.json"]


def test_mark_unknown_unit_raises() -> Any:
    s = start_session("sess", ["a"])
    with pytest.raises(KeyError):
        mark(s, "nope", UnitStatus.DONE)


def test_failed_units_are_resume_targets() -> Any:
    s = start_session("sess", ["a", "b", "c"])
    s = mark(s, "a", UnitStatus.DONE)
    s = mark(s, "b", UnitStatus.FAILED, error="boom")
    s = mark(s, "c", UnitStatus.SKIPPED)
    # FAILED is a resume target; SKIPPED and DONE are not.
    assert remaining_units(s) == ["b"]


def test_status_report_math() -> Any:
    s = start_session("sess", ["a", "b", "c"])
    s = mark(s, "a", UnitStatus.DONE)
    s = mark(s, "b", UnitStatus.DONE)
    report = status_report(s)
    assert report["total"] == 3
    assert report["completed"] == 2
    assert report["percent_complete"] == pytest.approx(66.66, abs=0.1)
    assert report["done"] is False
    assert report["by_status"][UnitStatus.DONE.value] == 2
    assert report["by_status"][UnitStatus.PENDING.value] == 1

    s = mark(s, "c", UnitStatus.DONE)
    report = status_report(s)
    assert report["percent_complete"] == pytest.approx(100.0, abs=0.001)
    assert report["done"] is True


def test_status_report_empty_session() -> Any:
    s = start_session("sess", [])
    report = status_report(s)
    assert report["total"] == 0
    assert report["percent_complete"] == 0.0
    assert report["done"] is False


def test_atomicity_no_temp_left_and_always_loadable(tmp_path: Path) -> Any:
    path = tmp_path / "manifest.json"
    s = start_session("sess", ["a", "b"])
    checkpoint(s, path)

    # No stray temp file remains after a checkpoint.
    leftovers = list(tmp_path.glob("manifest.json.tmp.*"))
    assert leftovers == []
    assert load_session(path) == s

    # A second checkpoint replaces atomically; the manifest stays valid.
    s = mark(s, "a", UnitStatus.DONE)
    checkpoint(s, path)
    assert list(tmp_path.glob("manifest.json.tmp.*")) == []
    reloaded = load_session(path)
    assert reloaded == s
    assert reloaded.units[0].status == UnitStatus.DONE


def test_cancel_safe_cleanup_preserves_done(tmp_path: Path) -> Any:
    # Create real artifact files for three units.
    refs = {}
    for uid in ("a", "b", "c"):
        f = tmp_path / f"{uid}.out"
        f.write_text(f"artifact-{uid}", encoding="utf-8")
        refs[uid] = f"{uid}.out"

    s = start_session(
        "sess",
        [
            WorkUnit(unit_id="a", artifact_refs=[refs["a"]]),
            WorkUnit(unit_id="b", artifact_refs=[refs["b"]]),
            WorkUnit(unit_id="c", artifact_refs=[refs["c"]]),
        ],
    )
    # Mark one DONE; its artifact must be preserved.
    s = mark(s, "a", UnitStatus.DONE, artifact_refs=[refs["a"]])

    removed = cancel_safe_cleanup(s, tmp_path)
    assert len(removed) == 2
    assert (tmp_path / "a.out").exists()  # DONE preserved
    assert not (tmp_path / "b.out").exists()
    assert not (tmp_path / "c.out").exists()

    # Session on disk still loads after cleanup.
    manifest = tmp_path / "manifest.json"
    checkpoint(s, manifest)
    assert load_session(manifest) == s

    # Idempotent: a second call removes nothing and does not raise.
    removed_again = cancel_safe_cleanup(s, tmp_path)
    assert removed_again == []


def test_cancel_safe_cleanup_path_escape_negative_control(tmp_path: Path) -> Any:
    # A file OUTSIDE the workdir that must never be touched.
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    victim = outside_root / "evil.txt"
    victim.write_text("do-not-delete", encoding="utf-8")

    workdir = tmp_path / "work"
    workdir.mkdir()

    # A legitimate in-workdir artifact, to confirm cleanup still runs.
    inside = workdir / "ok.out"
    inside.write_text("inside", encoding="utf-8")

    s = start_session(
        "sess",
        [
            WorkUnit(
                unit_id="u",
                status=UnitStatus.FAILED,
                artifact_refs=["../outside/evil.txt", "ok.out"],
            ),
        ],
    )

    removed = cancel_safe_cleanup(s, workdir)

    # The path-escape ref was ignored; the outside victim survives.
    assert victim.exists()
    assert victim.read_text(encoding="utf-8") == "do-not-delete"
    # The in-workdir artifact was removed.
    assert not inside.exists()
    assert str(victim.resolve()) not in removed
    assert any(r.endswith("ok.out") for r in removed)


def test_duplicate_unit_ids_rejected() -> None:
    """NEGATIVE: a duplicate unit_id is rejected — it would otherwise create an
    unreachable unit that never completes and makes resume loop forever."""
    import pytest

    from pipeline.run_session import start_session

    with pytest.raises(ValueError, match="duplicate unit_id"):
        start_session("dup", ["a", "a", "b"])
    # The non-duplicate case still works.
    assert len(start_session("ok", ["a", "b", "c"]).units) == 3


def test_checkpoint_temp_is_unique_not_pid_only(tmp_path) -> None:
    """Repeated checkpoints to the same path leave a loadable manifest and no temp residue."""
    from pipeline.run_session import (
        UnitStatus,
        checkpoint,
        load_session,
        mark,
        start_session,
    )

    session = start_session("s", ["a", "b"])
    dest = tmp_path / "session.json"
    for unit in ("a", "b"):
        session = mark(session, unit, UnitStatus.DONE)
        checkpoint(session, dest)
        assert load_session(dest).session_id == "s"
    assert not list(tmp_path.glob("*.tmp"))
