#!/usr/bin/env python3
"""Durable run manifest emission — turn a completed pipeline run into v3 artifacts.

This module is ADDITIVE and PURE: it reads existing on-disk run artifacts (the
``output/`` directory of a COMPLETED pipeline run) and emits durable, replayable
v3 artifacts. It NEVER executes the pipeline, runs containers, or contacts a
cluster. All file IO is local and atomic (delegated to ``durable_streams``).

For a finished run's output directory it produces:
  - one :class:`StreamManifest` per produced JSON artifact under the
    ``N_*_output/`` step directories (content-addressed, replayable);
  - one :class:`ExecutionTrace` reconstructed from
    ``00_pipeline_summary/pipeline_execution_summary.json`` (one event per step
    record, in run order), or — if that summary is absent — from the sorted set
    of existing ``N_*_output/`` step directories;
  - an index JSON listing every emitted artifact.

Public surface:
  - emit_run_manifests(run_output_dir, *, manifest_out=None) -> dict
  - verify_run_manifests(manifest_dir, run_output_dir) -> list[str]
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pipeline.durable_streams import (
    ExecutionTrace,
    StreamManifest,
    read_stream_manifest,
    read_trace,
    replay_trace,
    trace_integrity,
    validate_stream_manifest,
    write_stream_manifest,
    write_trace,
)

# Step output directories are named like ``3_gnn_output`` or ``11_render_output``.
_STEP_DIR_RE = re.compile(r"^(\d+)_.*_output$")

# Default subdirectory (relative to the run output dir) for emitted artifacts.
DEFAULT_MANIFEST_SUBDIR = "v3_run_manifest"

# Canonical location of the per-run execution summary within a run output dir.
_SUMMARY_REL = Path("00_pipeline_summary") / "pipeline_execution_summary.json"


def _stable_stream_id(rel_path: Path) -> str:
    """Derive a stable, filesystem-safe stream id from a relative artifact path.

    The id is a readable slug of the POSIX relative path PLUS a short hash of the
    exact path. The hash makes the id INJECTIVE: distinct artifacts whose slugs
    would otherwise collide (e.g. ``a.b.json`` and ``a-b.json`` both slugify to
    ``a_b_json``) get different ids, so no artifact's manifest can silently
    overwrite another's and leave its bytes unbound.

    Args:
        rel_path: Path of the artifact relative to the run output directory.

    Returns:
        A stable, collision-free stream identifier string.
    """
    posix = rel_path.as_posix()
    slug = re.sub(r"[^0-9A-Za-z]+", "_", posix).strip("_")
    digest = hashlib.sha256(posix.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def _discover_artifacts(run_output_dir: Path) -> List[Path]:
    """Return sorted JSON artifact paths under the run's ``N_*_output/`` dirs.

    Only ``*.json`` files inside step output directories are collected. The
    ``00_pipeline_summary`` directory is a meta directory (it backs the trace,
    not an artifact stream) and is therefore excluded. The result is sorted by
    POSIX relative path for deterministic ordering.

    Args:
        run_output_dir: The completed run's output directory.

    Returns:
        A deterministically sorted list of absolute artifact paths.
    """
    artifacts: List[Path] = []
    for child in run_output_dir.iterdir():
        if not child.is_dir():
            continue
        if not _STEP_DIR_RE.match(child.name):
            continue
        for json_path in child.rglob("*.json"):
            if json_path.is_file():
                artifacts.append(json_path)
    artifacts.sort(key=lambda p: p.relative_to(run_output_dir).as_posix())
    return artifacts


def _step_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the ordered list of per-step records from a summary dict.

    Args:
        summary: The parsed ``pipeline_execution_summary.json`` content.

    Returns:
        The list of step record dicts (possibly empty).
    """
    steps = summary.get("steps", [])
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _step_label(record: Dict[str, Any], fallback_index: int) -> str:
    """Build a human-meaningful step label from a step record.

    Prefers ``script_name``; falls back through ``description`` /
    ``step_number`` / a positional index so the label is always non-empty.

    Args:
        record: A single step record dict.
        fallback_index: Positional index used if the record carries no name.

    Returns:
        A non-empty step label.
    """
    for key in ("script_name", "step_name", "description"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    number = record.get("step_number")
    if number is not None:
        return f"step_{number}"
    return f"step_{fallback_index}"


def _build_trace_from_summary(
    trace_id: str, summary: Dict[str, Any]
) -> ExecutionTrace:
    """Reconstruct an execution trace from a parsed run summary.

    One event is appended per step record, in run order. The event ``step`` is
    the step label, ``action`` is the recorded status, and the payload bytes are
    the canonical JSON encoding of that step record (so the checksum binds the
    full step summary).

    Args:
        trace_id: Identifier for the resulting trace.
        summary: The parsed execution summary.

    Returns:
        A populated :class:`ExecutionTrace`.
    """
    trace = ExecutionTrace(trace_id=trace_id, created_by="run_manifest")
    for index, record in enumerate(_step_records(summary)):
        label = _step_label(record, index)
        status = record.get("status")
        action = status if isinstance(status, str) and status else "UNKNOWN"
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        trace = trace.append_event(
            step=label,
            action=action,
            payload_bytes=payload,
            payload_ref=label,
        )
    return trace


def _build_trace_from_dirs(
    trace_id: str, run_output_dir: Path
) -> ExecutionTrace:
    """Reconstruct a trace from the sorted set of existing step dirs.

    Used when no execution summary is present. Step directories are sorted by
    their numeric prefix, and one event is appended per directory. The payload
    bytes are the directory's relative name, so the trace digest is stable for a
    given set of step directories.

    Args:
        trace_id: Identifier for the resulting trace.
        run_output_dir: The completed run's output directory.

    Returns:
        A populated :class:`ExecutionTrace`.
    """
    step_dirs: List[Path] = []
    for child in run_output_dir.iterdir():
        if child.is_dir() and _STEP_DIR_RE.match(child.name):
            step_dirs.append(child)

    def _numeric_prefix(path: Path) -> tuple[int, str]:
        match = _STEP_DIR_RE.match(path.name)
        number = int(match.group(1)) if match else 0
        return (number, path.name)

    step_dirs.sort(key=_numeric_prefix)

    trace = ExecutionTrace(trace_id=trace_id, created_by="run_manifest")
    for child in step_dirs:
        name = child.name
        trace = trace.append_event(
            step=name,
            action="present",
            payload_bytes=name.encode("utf-8"),
            payload_ref=name,
        )
    return trace


def emit_run_manifests(
    run_output_dir: Union[str, Path],
    *,
    manifest_out: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Emit durable v3 manifests + a trace for a COMPLETED pipeline run.

    Walks ``run_output_dir`` for produced JSON artifacts (under the
    ``N_*_output/`` step dirs), builds a content-addressed
    :class:`StreamManifest` for each, reconstructs an :class:`ExecutionTrace`
    from the run summary (or, if absent, from the existing step directories),
    and writes everything plus an index JSON into ``manifest_out``.

    This function reads on-disk data only. It does not execute anything.

    Args:
        run_output_dir: The output directory of a completed run.
        manifest_out: Destination directory for the emitted artifacts; defaults
            to ``run_output_dir/v3_run_manifest``.

    Returns:
        A summary dict::

            {
                "stream_count": int,
                "trace_event_count": int,
                "trace_integrity_ok": bool,
                "manifest_dir": str,
            }

    Raises:
        FileNotFoundError: If ``run_output_dir`` does not exist.
        NotADirectoryError: If ``run_output_dir`` is not a directory.
    """
    run_dir = Path(run_output_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run output dir does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run output dir is not a directory: {run_dir}")

    manifest_dir = (
        Path(manifest_out)
        if manifest_out is not None
        else run_dir / DEFAULT_MANIFEST_SUBDIR
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build a StreamManifest per produced JSON artifact (deterministic order).
    artifacts = _discover_artifacts(run_dir)
    manifest_entries: List[Dict[str, str]] = []
    written_filenames: set[str] = set()
    for artifact in artifacts:
        rel = artifact.relative_to(run_dir)
        stream_id = _stable_stream_id(rel)
        manifest = StreamManifest.from_file(
            stream_id=stream_id,
            path=artifact,
            source=rel.as_posix(),
            created_by="run_manifest",
        )
        manifest_filename = f"{stream_id}.manifest.json"
        # Defense in depth: a non-injective id would let one artifact's manifest
        # silently overwrite another's, leaving its bytes unbound. Refuse to.
        if manifest_filename in written_filenames:
            raise ValueError(
                f"manifest filename collision for {rel.as_posix()!r}: {manifest_filename} "
                "already written (stream_id is not injective)"
            )
        written_filenames.add(manifest_filename)
        write_stream_manifest(manifest, manifest_dir / manifest_filename)
        manifest_entries.append(
            {
                "stream_id": stream_id,
                "source": rel.as_posix(),
                "manifest_file": manifest_filename,
            }
        )

    # 2. Reconstruct the execution trace from the summary, else from step dirs.
    trace_id = f"run::{run_dir.name}"
    summary_path = run_dir / _SUMMARY_REL
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trace = _build_trace_from_summary(trace_id, summary)
    else:
        trace = _build_trace_from_dirs(trace_id, run_dir)

    trace_filename = "execution_trace.json"
    write_trace(trace, manifest_dir / trace_filename)
    integrity_problems = trace_integrity(trace)
    trace_integrity_ok = not integrity_problems

    # 3. Write an index JSON listing every emitted artifact.
    index = {
        "run_output_dir": str(run_dir),
        "schema_version": "3.0",
        "trace_file": trace_filename,
        "trace_event_count": len(trace.events),
        "trace_integrity_ok": trace_integrity_ok,
        "stream_count": len(manifest_entries),
        "manifests": manifest_entries,
    }
    index_text = json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False)
    _write_index(manifest_dir / "index.json", index_text)

    return {
        "stream_count": len(manifest_entries),
        "trace_event_count": len(trace.events),
        "trace_integrity_ok": trace_integrity_ok,
        "manifest_dir": str(manifest_dir),
    }


def _write_index(path: Path, text: str) -> None:
    """Atomically write the index JSON.

    Reuses the same atomic strategy as ``durable_streams`` (tmp file +
    ``os.replace``) via a tiny local helper to avoid importing a private name.

    Args:
        path: Destination index path.
        text: Serialized index JSON.
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, str(path))
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def verify_run_manifests(
    manifest_dir: Union[str, Path], run_output_dir: Union[str, Path]
) -> List[str]:
    """Re-validate every emitted manifest and the trace.

    Each emitted :class:`StreamManifest` is re-checked against its backing file
    under ``run_output_dir`` (so a tampered artifact surfaces as a checksum
    mismatch). The trace's structural integrity is re-verified AND its meaning is
    re-bound to ground truth: the trace is re-derived from the live execution
    summary (or step dirs) and its replay digest is compared to the stored trace,
    so tampering the summary (e.g. flipping a step status) or relabeling a stored
    event surfaces as a digest mismatch — not just structurally-clean but stale.

    Args:
        manifest_dir: Directory previously written by :func:`emit_run_manifests`.
        run_output_dir: The run output directory the manifests describe.

    Returns:
        A list of human-readable problems. Empty means everything re-validates.
    """
    mdir = Path(manifest_dir)
    run_dir = Path(run_output_dir)
    problems: List[str] = []

    index_path = mdir / "index.json"
    if not index_path.is_file():
        problems.append(f"index.json missing in {mdir}")
        return problems

    index = json.loads(index_path.read_text(encoding="utf-8"))

    for entry in index.get("manifests", []):
        manifest_file = entry.get("manifest_file")
        if not manifest_file:
            problems.append(f"index entry missing manifest_file: {entry!r}")
            continue
        manifest_path = mdir / manifest_file
        if not manifest_path.is_file():
            problems.append(f"manifest file missing: {manifest_path}")
            continue
        manifest = read_stream_manifest(manifest_path)
        for problem in validate_stream_manifest(manifest, run_dir):
            problems.append(f"{manifest.stream_id}: {problem}")

    trace_file = index.get("trace_file", "execution_trace.json")
    trace_path = mdir / trace_file
    if not trace_path.is_file():
        problems.append(f"trace file missing: {trace_path}")
    else:
        trace = read_trace(trace_path)
        for problem in trace_integrity(trace):
            problems.append(f"trace: {problem}")
        # Re-bind the trace's meaning to ground truth: rebuild it from the live
        # summary (or step dirs) exactly as emit did and compare replay digests.
        trace_id = f"run::{run_dir.name}"
        summary_path = run_dir / _SUMMARY_REL
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected = _build_trace_from_summary(trace_id, summary)
            except (json.JSONDecodeError, ValueError):
                problems.append("trace: execution summary is unreadable; cannot re-bind trace")
                expected = None
        else:
            expected = _build_trace_from_dirs(trace_id, run_dir)
        if expected is not None and replay_trace(expected) != replay_trace(trace):
            problems.append(
                "trace: replay digest does not match the trace re-derived from the "
                "live run summary (summary tampered or stored trace relabeled)"
            )

    return problems
