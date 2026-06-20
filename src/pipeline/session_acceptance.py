#!/usr/bin/env python3
"""Resumable, session-wrapped model-family acceptance orchestration.

Wraps the existing manifest-driven model-family acceptance runner
(:mod:`pipeline.model_family_acceptance`) in a resumable :class:`RunSession`
(:mod:`pipeline.run_session`) so an extended multi-family acceptance run is
checkpointed after every family and can be resumed after a crash.

This module is pure orchestration over the existing acceptance runner. It
NEVER executes the pipeline, runs containers, or calls clusters: it delegates
to ``run_model_family_acceptance`` (which itself only stages on-disk fixtures
and invokes an injectable ``runner`` callable), records per-family ledgers and
artifact directories, and checkpoints a session manifest to disk after each
family so an interrupted run is resumable.

The session work units are the family names. Each family is processed in its
own single-family acceptance run; the unit is marked ``DONE`` on a passing
ledger and ``FAILED`` on an exception or a failed ledger. Checkpoint writes are
atomic (handled by :func:`pipeline.run_session.checkpoint`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.model_family_acceptance import (
    Runner,
    load_model_family_manifest,
    run_model_family_acceptance,
)
from pipeline.run_session import (
    RunSession,
    UnitStatus,
    checkpoint,
    load_session,
    mark,
    remaining_units,
    start_session,
    status_report,
)


def _selected_family_names(
    manifest_path: Path, family_names: Optional[Iterable[str]]
) -> List[str]:
    """Return ordered family names to process, filtered by ``family_names``.

    Args:
        manifest_path: Path to the model-family manifest.
        family_names: Optional subset of family names to include; ``None`` or
            empty means every family declared in the manifest.

    Returns:
        Family names in manifest order, restricted to the requested subset.

    Raises:
        KeyError: If any requested family name is absent from the manifest.
    """
    families = load_model_family_manifest(manifest_path)
    all_names = [family.name for family in families]
    requested = [name.strip() for name in (family_names or []) if name.strip()]
    if not requested:
        return all_names
    missing = [name for name in requested if name not in set(all_names)]
    if missing:
        raise KeyError(f"Unknown model families: {', '.join(sorted(missing))}")
    requested_set = set(requested)
    return [name for name in all_names if name in requested_set]


def run_session_acceptance(
    manifest_path: Path,
    output_dir: Path,
    session_path: Path,
    *,
    family_names: Optional[Iterable[str]] = None,
    strict: bool = False,
    runner: Optional[Runner] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run model-family acceptance under a resumable, checkpointed session.

    Loads the requested families, starts (or resumes) a :class:`RunSession`
    over the family names, and runs single-family acceptance for each family
    that is not yet ``DONE``. The session is checkpointed to ``session_path``
    after every family so a crash mid-run is resumable.

    Args:
        manifest_path: Path to the model-family manifest.
        output_dir: Directory for per-family acceptance ledger artifacts.
        session_path: Destination for the resumable session manifest JSON.
        family_names: Optional subset of family names to process; ``None`` means
            every family in the manifest.
        strict: Forwarded to ``run_model_family_acceptance`` (raises on a failed
            ledger). A failed family is recorded as ``FAILED`` regardless.
        runner: Optional injectable runner callable forwarded to the underlying
            acceptance runner (used to avoid executing the real pipeline).
        resume: When ``True`` and ``session_path`` exists, resume that session
            instead of starting a fresh one.

    Returns:
        A dict with keys ``session`` (the final ``RunSession`` dumped to a
        dict), ``status`` (the ``status_report``), and ``ledgers`` (a mapping of
        family name to its acceptance ledger dict, for families processed this
        call).
    """
    output_dir = Path(output_dir)
    session_path = Path(session_path)

    selected_names = _selected_family_names(manifest_path, family_names)

    session: RunSession
    if resume and session_path.exists():
        session = load_session(session_path)
        # A resumed session must cover every requested family. If the caller
        # requests a family the persisted session never knew about, the loop
        # below would silently skip it and report done=True having run nothing —
        # fail loudly on that mismatch instead.
        session_units = {unit.unit_id for unit in session.units}
        unknown = sorted(name for name in selected_names if name not in session_units)
        if unknown:
            raise ValueError(
                "resume request includes families not in the persisted session "
                f"{sorted(session_units)}: {unknown}"
            )
    else:
        session = start_session(
            session_path.stem or "session_acceptance",
            selected_names,
            created_by="session_acceptance",
        )

    pending = set(remaining_units(session))
    ledgers: Dict[str, Any] = {}

    for name in selected_names:
        if name not in pending:
            continue
        family_output = output_dir / name
        try:
            ledger = run_model_family_acceptance(
                manifest_path,
                family_output,
                family_names=[name],
                strict=strict,
                runner=runner,
            )
        except Exception as exc:  # noqa: BLE001 — record failure, keep resumable
            session = mark(
                session,
                name,
                UnitStatus.FAILED,
                artifact_refs=[str(family_output)],
                error=f"{type(exc).__name__}: {exc}",
            )
            checkpoint(session, session_path)
            raise

        ledgers[name] = ledger
        if ledger.get("status") == "passed":
            session = mark(
                session,
                name,
                UnitStatus.DONE,
                artifact_refs=[str(family_output)],
            )
        else:
            failed = ledger.get("failed_families") or [name]
            session = mark(
                session,
                name,
                UnitStatus.FAILED,
                artifact_refs=[str(family_output)],
                error=f"acceptance ledger failed: {', '.join(map(str, failed))}",
            )
        checkpoint(session, session_path)

    return {
        "session": session.model_dump(),
        "status": status_report(session),
        "ledgers": ledgers,
    }
