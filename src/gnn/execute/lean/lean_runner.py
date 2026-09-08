"""Lean verification runner: drives ``fep-lean bridge verify-document``.

Resolves the sibling ``fep_lean`` checkout (env ``FEP_LEAN_ROOT`` overrides
the ``../fep_lean`` default), discovers emitted Lean/GNN documents under the
rendered target directory, and verifies each through the fep_lean bridge
``verify-document`` operation (bridge contract v0.6, Direction 2 S7:
well-formedness against the ``FEP.GnnDocument`` typed AST).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Union

from gnn.execute.subprocess_envelope import run_subprocess_envelope

logger = logging.getLogger(__name__)

#: Env override naming the fep_lean checkout.
FEP_LEAN_ROOT_ENV = "FEP_LEAN_ROOT"

#: Default ceiling for one verify-document invocation (Lean + Mathlib).
_VERIFY_TIMEOUT_SECONDS = 1800


def resolve_fep_lean_root() -> Path | None:
    """Resolve and validate the fep_lean checkout root, or ``None``.

    ``FEP_LEAN_ROOT`` wins when set; otherwise the default is the
    ``fep_lean`` sibling of the GNN repository root. A checkout must carry
    ``pyproject.toml`` and the ``src/fep_lean`` package.
    """
    env = os.environ.get(FEP_LEAN_ROOT_ENV)
    root = Path(env) if env else Path(__file__).resolve().parents[4].parent / "fep_lean"
    root = root.resolve()
    if (root / "pyproject.toml").is_file() and (root / "src" / "fep_lean").is_dir():
        return root
    return None


def lean_toolchain_available() -> bool:
    """True when a fep_lean checkout is resolvable for verification."""
    return resolve_fep_lean_root() is not None


def verify_document(
    document: Union[str, Path],
    receipt: Union[str, Path] | None = None,
    *,
    model: str = "finite",
    fail_on_warnings: bool = True,
    gnn_root: Union[str, Path] | None = None,
    timeout: int = _VERIFY_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Verify one emitted document via ``fep-lean bridge verify-document``.

    Returns a record with ``success`` plus either the parsed receipt or an
    ``error`` message captured from the bridge CLI (fail-closed).
    """
    root = resolve_fep_lean_root()
    if root is None:
        logger.info(
            "ℹ️ fep_lean not available - skipping Lean verification (set %s)",
            FEP_LEAN_ROOT_ENV,
        )
        return {"success": False, "error": "fep_lean unavailable"}

    document_path = Path(document).resolve()
    gnn_root_path = (
        Path(gnn_root).resolve() if gnn_root else Path(__file__).resolve().parents[4]
    )
    if receipt is not None:
        receipt_path = Path(receipt).resolve()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        receipt_path = (
            Path(tempfile.mkdtemp(prefix="gnn-lean-verify-"))
            / f"{document_path.stem}-receipt.json"
        )

    command = [
        "uv",
        "run",
        "fep-lean",
        "bridge",
        "verify-document",
        "--gnn-root",
        str(gnn_root_path),
        "--document",
        str(document_path),
        "--model",
        model,
        "--receipt",
        str(receipt_path),
    ]
    if fail_on_warnings:
        command.append("--fail-on-warnings")

    record: dict[str, Any] = {
        "success": False,
        "document": str(document_path),
        "command": command,
    }
    envelope = run_subprocess_envelope(command, timeout=timeout, cwd=str(root))
    record["returncode"] = envelope["return_code"]
    if envelope["success"]:
        record["success"] = True
        if receipt_path.is_file():
            try:
                record["receipt"] = json.loads(receipt_path.read_text())
            except json.JSONDecodeError:
                record["receipt_error"] = "unparseable receipt JSON"
        return record

    if envelope["return_code"] == -1:
        # Timeout or invocation failure (OSError) — fail closed with the cause.
        record["error"] = (
            f"verify-document invocation failed: {envelope.get('error', '')}"
        )
        return record

    record["error"] = (
        envelope["stderr"] or envelope["stdout"] or "verify-document failed"
    ).strip()[-2000:]
    return record


def run_lean_scripts(
    rendered_simulators_dir: Union[str, Path],
    execution_output_dir: Union[str, Path] | None = None,
    recursive_search: bool = True,
    verbose: bool = False,
) -> bool:
    """Verify every emitted Lean/GNN document under the target directory.

    Mirrors the per-framework runner contract: returns ``True`` when every
    discovered document verifies (or when there is nothing to verify),
    ``False`` when the fep_lean checkout is unavailable or any document
    fails. Per-document receipts are written under ``execution_output_dir``.
    """
    if resolve_fep_lean_root() is None:
        logger.info(
            "ℹ️ fep_lean not available - skipping Lean verification (set %s)",
            FEP_LEAN_ROOT_ENV,
        )
        return False

    target = Path(rendered_simulators_dir)
    glob = target.rglob if recursive_search else target.glob
    documents = sorted(set(glob("*.lean")))
    documents += sorted(path for path in glob("*.md") if _is_gnn_document(path))

    output_dir = Path(execution_output_dir) if execution_output_dir else target / "lean"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        logger.info("ℹ️ No Lean/GNN documents found to verify")
        return True

    all_ok = True
    for document in documents:
        record = verify_document(document, output_dir / f"{document.stem}-receipt.json")
        ok = bool(record.get("success"))
        all_ok = all_ok and ok
        status_icon = "✅" if ok else "❌"
        message = record.get("error", "well-formed")
        if ok or verbose:
            logger.info(f"{status_icon} Lean verification {document.name}: {message}")
        else:
            logger.warning(
                f"{status_icon} Lean verification {document.name}: {message}"
            )
    return all_ok


def _is_gnn_document(path: Path) -> bool:
    """A GNN ``.md`` document carries the required ``GNNSection`` header."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    return "## GNNSection" in head
