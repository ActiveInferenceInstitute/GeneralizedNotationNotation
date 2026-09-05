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

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.hasher import runtime_config_identity
from pipeline.model_family_acceptance import (
    DEFAULT_EVIDENCE_STEPS,
    ModelFamily,
    Runner,
    load_model_family_manifest,
    run_model_family_acceptance,
    select_model_families,
)
from pipeline.run_session import (
    RunSession,
    UnitStatus,
    checkpoint,
    load_session,
    mark,
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
    return [family.name for family in select_model_families(families, family_names)]


def _file_digest(path: Path) -> str:
    """Hash artifact bytes without loading large execution outputs into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _family_identity(
    family: ModelFamily, manifest_path: Path, output_dir: Path, strict: bool
) -> Dict[str, Any]:
    """Bind a family to the actual source, policy, and destination of this call."""
    return {
        "schema": "acceptance-family-v2",
        "manifest_path": str(manifest_path.resolve()),
        "source_directory": str(family.target_dir.resolve()),
        "sources": {
            name: _file_digest(family.target_dir / name)
            for name in sorted(family.representative_files)
        },
        "frameworks": family.frameworks,
        "acceptance_profile": family.acceptance_profile,
        "strict": strict,
        "steps": DEFAULT_EVIDENCE_STEPS,
        "runtime_config": runtime_config_identity(),
        "output_directory": str(output_dir.resolve()),
    }


def _artifact_hashes(directory: Path, session_path: Path) -> Dict[str, str]:
    """Snapshot the complete output inventory, excluding the checkpoint itself."""
    return {
        path.relative_to(directory).as_posix(): _file_digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.resolve() != session_path.resolve()
    }


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
    output_dir = Path(output_dir).resolve()
    session_path = Path(session_path)

    families = select_model_families(
        load_model_family_manifest(manifest_path), family_names
    )
    selected_names = [family.name for family in families]
    identities = {
        family.name: _family_identity(
            family, manifest_path, output_dir / family.name, strict
        )
        for family in families
    }

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

    ledgers: Dict[str, Any] = {}

    for family in families:
        name = family.name
        family_output = output_dir / name
        unit = next(unit for unit in session.units if unit.unit_id == name)
        if unit.status == UnitStatus.DONE and unit.input_identity == identities[name]:
            try:
                if unit.artifact_hashes and unit.artifact_hashes == _artifact_hashes(
                    family_output, session_path
                ):
                    continue
            except OSError:
                pass  # Missing/unreadable evidence is not reusable completion.
        elif unit.status == UnitStatus.SKIPPED:
            continue
        unit.input_identity = identities[name]
        unit.artifact_hashes = {}
        session.run_hash = hashlib.sha256(
            json.dumps(
                {u.unit_id: u.input_identity for u in session.units}, sort_keys=True
            ).encode()
        ).hexdigest()[:12]
        session = mark(
            session, name, UnitStatus.RUNNING, artifact_refs=[str(family_output)]
        )
        checkpoint(session, session_path)
        try:
            ledger = run_model_family_acceptance(
                manifest_path,
                family_output,
                family_names=[name],
                strict=strict,
                runner=runner,
            )
            current_families = select_model_families(
                load_model_family_manifest(manifest_path), [name]
            )
            if (
                _family_identity(
                    current_families[0], manifest_path, family_output, strict
                )
                != identities[name]
            ):
                raise ValueError(f"Source changed during acceptance: {name}")
            artifacts = _artifact_hashes(family_output, session_path)
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
        next(
            unit for unit in session.units if unit.unit_id == name
        ).artifact_hashes = artifacts
        checkpoint(session, session_path)

    return {
        "session": session.model_dump(),
        "status": status_report(session),
        "ledgers": ledgers,
    }
