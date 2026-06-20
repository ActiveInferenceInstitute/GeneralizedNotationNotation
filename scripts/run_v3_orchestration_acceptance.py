#!/usr/bin/env python3
"""v3.0.0 Long-Running Orchestration acceptance gate.

Exercises the three v3.0.0 contracts end to end and fails closed:

  1. Durable observation streams  (pipeline.durable_streams)
  2. Resumable long-running sessions (pipeline.run_session)
  3. Auditable container plans     (pipeline.container_plan)

For each contract the gate runs POSITIVE checks (the happy path must succeed) and
NEGATIVE CONTROLS (a deliberately corrupted input MUST be caught). If any positive
check fails, or any negative control fails to fire, the gate exits non-zero — so a
silently weakened validator cannot pass. Safe by design: no container is executed,
no cluster or device is touched; only local manifest/trace/plan data is written under
a temporary working directory.

Usage:
    python scripts/run_v3_orchestration_acceptance.py [--strict] [--inject-defect]

``--inject-defect`` breaks one positive assertion to demonstrate the non-zero exit
path (used to evidence fail-closed behavior in the TODO acceptance).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from pipeline import container_plan as cp  # noqa: E402
from pipeline import durable_streams as ds  # noqa: E402
from pipeline import run_session as rs  # noqa: E402


class _Gate:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def expect(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))

    def report(self) -> int:
        passed = sum(1 for _, ok, _ in self.checks if ok)
        for name, ok, detail in self.checks:
            mark = "✓" if ok else "✗"
            line = f"  {mark} {name}"
            if detail and not ok:
                line += f"  — {detail}"
            print(line)
        print(f"\n{passed}/{len(self.checks)} checks passed")
        return 0 if passed == len(self.checks) else 1


def _check_streams(g: _Gate, workdir: Path, inject_defect: bool) -> None:
    arr = np.arange(24, dtype=np.float64).reshape(4, 6)
    manifest = ds.StreamManifest.from_array("obs_stream", arr, source="obs.npy")
    mpath = ds.write_stream_manifest(manifest, workdir / "obs.manifest.json")
    reloaded = ds.read_stream_manifest(mpath)
    g.expect("streams: manifest round-trips", reloaded == manifest)

    # Array manifest validates clean.
    g.expect("streams: array manifest valid", ds.validate_stream_manifest(manifest, workdir) == [])

    # File-backed manifest + negative control (tamper the file).
    data = b"durable-observation-bytes" * 8
    fpath = workdir / "obs.bin"
    fpath.write_bytes(data)
    fmanifest = ds.StreamManifest.from_file("file_stream", fpath)
    g.expect("streams: file manifest valid", ds.validate_stream_manifest(fmanifest, workdir) == [])
    fpath.write_bytes(data + b"TAMPER")
    tamper_problems = ds.validate_stream_manifest(fmanifest, workdir)
    g.expect("streams: NEGATIVE control fires on tamper", len(tamper_problems) > 0,
             "checksum mismatch was not detected")

    # Trace integrity + deterministic replay.
    trace = ds.ExecutionTrace(trace_id="run1")
    for step, action, payload in (("3", "parse", b"a"), ("5", "typecheck", b"b"), ("11", "render", b"c")):
        trace = trace.append_event(step, action, payload)
    g.expect("streams: trace integrity clean", ds.trace_integrity(trace) == [])
    digest = ds.replay_trace(trace)
    g.expect("streams: replay deterministic", ds.replay_trace(trace) == digest)
    g.expect("streams: verify_replay accepts true digest", ds.verify_replay(trace, digest))
    g.expect("streams: verify_replay rejects wrong digest", not ds.verify_replay(trace, "0" * 64))

    # NEGATIVE control: a corrupted trace (gap) must be reported.
    bad = ds.ExecutionTrace(
        trace_id="bad",
        events=[
            ds.TraceEvent(seq=0, step="3", action="parse", payload_ref="", payload_checksum="x"),
            ds.TraceEvent(seq=2, step="5", action="typecheck", payload_ref="", payload_checksum="y"),
        ],
    )
    g.expect("streams: NEGATIVE control fires on seq gap", len(ds.trace_integrity(bad)) > 0)

    if inject_defect:
        g.expect("streams: INJECTED DEFECT (must fail)", False, "deliberate failure")


def _check_session(g: _Gate, workdir: Path) -> None:
    session = rs.start_session("accept-1", ["basics", "discrete", "gridworld"], created_by="acceptance")
    session = rs.mark(session, "basics", rs.UnitStatus.DONE, artifact_refs=["basics/out.json"])
    session = rs.mark(session, "discrete", rs.UnitStatus.DONE, artifact_refs=["discrete/out.json"])
    spath = rs.checkpoint(session, workdir / "session.json")
    loaded = rs.load_session(spath)
    g.expect("session: checkpoint round-trips", loaded == session)
    g.expect("session: resume targets the unfinished unit", rs.remaining_units(loaded) == ["gridworld"])
    report = rs.status_report(loaded)
    g.expect("session: status math (2/3 done)", report["completed"] == 2 and not report["done"])

    # Cancellation-safe cleanup: create artifacts, only non-DONE ones are removed.
    for unit in ("basics", "discrete", "gridworld"):
        p = workdir / unit
        p.mkdir(parents=True, exist_ok=True)
        (p / "out.json").write_text("{}")
    session = rs.mark(session, "gridworld", rs.UnitStatus.RUNNING, artifact_refs=["gridworld/out.json"])
    removed = rs.cancel_safe_cleanup(session, workdir)
    g.expect("session: cleanup removed the partial artifact", any("gridworld" in r for r in removed))
    g.expect("session: cleanup preserved DONE artifact", (workdir / "basics" / "out.json").exists())
    g.expect("session: cleanup idempotent", rs.cancel_safe_cleanup(session, workdir) == [])


def _check_container(g: _Gate) -> None:
    plan = cp.generate_container_plan(
        "gnn-pipeline",
        [{"name": "runner", "image": "ghcr.io/gnn/runner@sha256:" + "a" * 64, "command": ["python", "src/main.py"]}],
    )
    g.expect("container: hardened default passes security review", cp.security_review(plan) == [])
    g.expect("container: plan hash is deterministic", cp.compute_plan_hash(plan) == plan.plan_hash)

    # Rollback semantics across versions.
    v2 = cp.generate_container_plan(
        "gnn-pipeline",
        [{"name": "runner", "image": "ghcr.io/gnn/runner@sha256:" + "b" * 64, "command": ["python", "src/main.py"]}],
        previous=plan,
    )
    g.expect("container: v2 carries rollback to v1",
             v2.version == 2 and v2.rollback is not None and v2.rollback.previous_plan_hash == plan.plan_hash)

    # NEGATIVE control: an insecure plan must be flagged.
    bad = cp.ContainerPlan(
        plan_id="bad",
        specs=[cp.ContainerSpec(
            name="bad", image="myimage:latest", privileged=True, user="root",
            env={"DB_PASSWORD": "hunter2"}, read_only_rootfs=False, cap_drop=[],
        )],
    )
    findings = cp.security_review(bad)
    codes = {f.code for f in findings}
    has_critical = any(f.severity == "CRITICAL" for f in findings)
    g.expect("container: NEGATIVE control flags insecure plan (incl. CRITICAL)",
             has_critical and len(codes) >= 4, f"codes={sorted(codes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="v3.0.0 orchestration acceptance gate")
    parser.add_argument("--strict", action="store_true", help="(default behavior) fail closed on any miss")
    parser.add_argument("--inject-defect", action="store_true", help="break a positive check to show non-zero exit")
    args = parser.parse_args()

    g = _Gate()
    print("v3.0.0 Long-Running Orchestration acceptance")
    print("=" * 48)
    with tempfile.TemporaryDirectory(prefix="gnn-v3-accept-") as tmp:
        workdir = Path(tmp)
        _check_streams(g, workdir, args.inject_defect)
        _check_session(g, workdir)
        _check_container(g)
    code = g.report()
    print("\n" + ("✅ v3.0.0 acceptance PASSED" if code == 0 else "❌ v3.0.0 acceptance FAILED"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
