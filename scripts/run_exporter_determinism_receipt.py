#!/usr/bin/env python3
"""Consumers exporter-determinism receipt.

Runs the GNN export pipeline twice over a corpus of GNN markdown files and
proves that every successfully exported artifact is byte-identical across the
two runs. For each corpus file the source is parsed exactly once and the same
parsed model dict is exported twice into two scratch directories, so any byte
difference between the two runs is an exporter defect, not a parser artifact.

Receipt behavior:

- The receipt JSON is always written, even when the run fails mid-way, so a
  failed gate still leaves an auditable artifact.
- Files that fail to parse are recorded with ``parse_ok: false`` and the error
  string, and are excluded from the determinism claim (they are not counted as
  mismatches).
- Exit codes are fail-closed: 0 means deterministic with at least one artifact
  compared; 1 means nondeterministic, no parseable corpus, zero comparable
  artifacts, or any operational failure.
- ``result`` is one of ``deterministic``, ``nondeterministic``,
  ``no-parseable-corpus``, or ``error`` (parseable corpus that produced zero
  comparable artifacts, or an unexpected operational error, which additionally
  adds an ``error`` field to the receipt).

Paths: default ``--target-dir`` and ``--receipts-dir`` are resolved relative to
the repository root so the receipt is reproducible from a fresh checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gnn.export import export_model  # noqa: E402
from gnn.export.processor import parse_gnn_content  # noqa: E402

RECEIPT_VERSION = "gnn_exporter_determinism_receipt_v1"
GENERATOR = "run_exporter_determinism_receipt.py"
RECEIPT_FILENAME = "exporter_determinism_receipt.json"
DEFAULT_TARGET_DIR = "input/gnn_files"
DEFAULT_RECEIPTS_DIR = "output/7_export_output"
DEFAULT_FORMATS = "json,xml,graphml,gexf,pickle"
_DIRTY_PATH_CAP = 200


def _sha256_file(path: Path) -> str | None:
    """Return the sha256 hex digest of ``path``, or None when unreadable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git_output(args: list[str], cwd: Path) -> str | None:
    """Run a read-only git command, returning trimmed stdout or None."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _describe_commit(root: Path) -> str:
    """Short HEAD commit for provenance, or 'unknown' when git is unavailable."""
    sha = _git_output(["rev-parse", "--short", "HEAD"], root)
    return sha if sha else "unknown"


def _working_tree_dirty(root: Path) -> tuple[bool, list[str]]:
    """Dirty flag plus up to ``_DIRTY_PATH_CAP`` dirty paths (deterministic)."""
    status = _git_output(["status", "--porcelain"], root)
    if status is None:
        return False, []
    paths: list[str] = []
    for line in status.splitlines():
        # Porcelain format: two status chars, a space, then the path (renames
        # are ``ORIG -> NEW``; the destination is the surviving path).
        paths.append(line[3:].split(" -> ")[-1])
    dirty_paths = sorted(set(paths))[:_DIRTY_PATH_CAP]
    return bool(paths), dirty_paths


def _collect_corpus(target_dir: Path) -> list[Path]:
    """Every ``*.md`` file under ``target_dir`` in deterministic POSIX order."""
    return sorted(
        (p for p in target_dir.rglob("*.md") if p.is_file()),
        key=lambda p: p.as_posix(),
    )


def compare_artifacts(dir_a: Path, dir_b: Path) -> list[dict[str, str | None]]:
    """Byte-compare every artifact file under two export directories.

    Files are keyed by their POSIX path relative to each directory. A file
    present in only one directory is a mismatch (the missing side hashes as
    None). Returns one dict per differing path, sorted by relative path:
    ``{"path", "run_a_sha256", "run_b_sha256"}``.
    """
    rel_paths: set[str] = set()
    for directory in (dir_a, dir_b):
        for p in directory.rglob("*"):
            if p.is_file():
                rel_paths.add(p.relative_to(directory).as_posix())
    diffs: list[dict[str, str | None]] = []
    for rel in sorted(rel_paths):
        sha_a = _sha256_file(dir_a / rel)
        sha_b = _sha256_file(dir_b / rel)
        if sha_a != sha_b:
            diffs.append({"path": rel, "run_a_sha256": sha_a, "run_b_sha256": sha_b})
    return diffs


def decide_result(
    artifacts_compared: int, mismatches: list[Any], parseable_files: int
) -> str:
    """Map run counters onto the receipt ``result`` vocabulary (fail-closed)."""
    if parseable_files == 0:
        return "no-parseable-corpus"
    if artifacts_compared == 0:
        # Vacuous determinism guard: a parseable corpus that exported nothing
        # (empty format list, all-unsupported formats, empty exports) must not
        # pass as "deterministic". Fail closed instead.
        return "error"
    if mismatches:
        return "nondeterministic"
    return "deterministic"


def _parse_formats(raw: str) -> list[str]:
    """Split a comma-separated format list, preserving declaration order."""
    seen: dict[str, None] = {}
    for token in raw.split(","):
        fmt = token.strip()
        if fmt:
            seen.setdefault(fmt, None)
    return list(seen)


def _export_twice(
    model_data: dict[str, Any],
    formats: list[str],
) -> tuple[
    dict[str, Any], dict[str, dict[str, str | None]], list[dict[str, Any]], Path
]:
    """Export one parsed model into two scratch dirs and hash the artifacts.

    Returns ``(run_a_result, formats_map, mismatches, scratch_root)``. The
    formats map is the per-format receipt inventory ``fmt -> {run_a_sha256,
    run_b_sha256}`` (None when the artifact was not produced or is unreadable).
    Mismatches are one entry per differing artifact with ``format`` and both
    hashes. Scratch dirs live under a fresh temp root the caller must remove.
    """
    scratch = Path(tempfile.mkdtemp(prefix="gnn_export_det_"))
    dir_a = scratch / "run_a"
    dir_b = scratch / "run_b"
    try:
        result_a = export_model(model_data, dir_a, formats=formats)
        export_model(model_data, dir_b, formats=formats)

        formats_map: dict[str, dict[str, str | None]] = {}
        basename_to_fmt: dict[str, str] = {}
        exports: dict[str, Any] = result_a.get("exports", {})
        for fmt in formats:
            file_info = exports.get(fmt)
            basename: str | None = (
                Path(str(file_info["file"])).name
                if isinstance(file_info, dict) and file_info.get("file")
                else None
            )
            if basename is not None:
                basename_to_fmt[basename] = fmt
                sha_a = _sha256_file(dir_a / basename)
                sha_b = _sha256_file(dir_b / basename)
            else:
                sha_a = None
                sha_b = None
            formats_map[fmt] = {
                "run_a_sha256": sha_a,
                "run_b_sha256": sha_b,
            }

        mismatches: list[dict[str, Any]] = []
        for diff in compare_artifacts(dir_a, dir_b):
            mismatches.append(
                {
                    "format": basename_to_fmt.get(
                        Path(str(diff["path"])).name, str(diff["path"])
                    ),
                    "run_a_sha256": diff["run_a_sha256"],
                    "run_b_sha256": diff["run_b_sha256"],
                }
            )
    except BaseException:
        _rmtree(scratch)
        raise
    return result_a, formats_map, mismatches, scratch


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def build_receipt(
    target_dir: Path,
    formats: list[str],
    corpus: list[Path],
) -> dict[str, Any]:
    """Parse-and-export every corpus file twice; return the receipt payload."""
    corpus_entries: list[dict[str, Any]] = []
    parse_failures: list[dict[str, str]] = []
    all_mismatches: list[dict[str, Any]] = []
    parseable_files = 0
    artifacts_compared = 0

    for md_path in corpus:
        rel = md_path.relative_to(target_dir).as_posix()
        try:
            content = md_path.read_text(encoding="utf-8")
            parsed = parse_gnn_content(content)
        except (OSError, ValueError) as exc:
            parsed = {"error": f"unreadable source: {exc}"}
        if "sections" not in parsed:
            error = str(parsed.get("error", "unknown parse error"))
            parse_failures.append({"file": rel, "error": error})
            corpus_entries.append({"file": rel, "parse_ok": False, "formats": {}})
            continue
        parseable_files += 1

        result_a, formats_map, mismatches, scratch = _export_twice(parsed, formats)
        try:
            for mismatch in mismatches:
                all_mismatches.append({"file": rel, **mismatch})
            artifacts_compared += sum(
                1
                for entry in formats_map.values()
                if entry["run_a_sha256"] is not None
                and entry["run_b_sha256"] is not None
            )
            _ = result_a  # result dicts carry per-run absolute paths; not compared
        finally:
            _rmtree(scratch)
        corpus_entries.append({"file": rel, "parse_ok": True, "formats": formats_map})

    result = decide_result(artifacts_compared, all_mismatches, parseable_files)
    error_note: str | None = None
    if result == "error":
        error_note = (
            "no comparable artifacts exported for a parseable corpus "
            f"(formats={formats!r}); determinism not established"
        )
    dirty, dirty_paths = _working_tree_dirty(_PROJECT_ROOT)
    receipt: dict[str, Any] = {
        "receipt_version": RECEIPT_VERSION,
        "generator": GENERATOR,
        "counts_describe_commit": _describe_commit(_PROJECT_ROOT),
        "working_tree_dirty": dirty,
        "dirty_paths": dirty_paths,
        "corpus": corpus_entries,
        "artifacts_compared": artifacts_compared,
        "mismatches": all_mismatches,
        "parse_failures": parse_failures,
        "result": result,
    }
    if error_note is not None:
        receipt["error"] = error_note
    return receipt


def _write_receipt(receipts_dir: Path, receipt: dict[str, Any]) -> Path:
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / RECEIPT_FILENAME
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    receipt_path.write_text(payload, encoding="utf-8")
    return receipt_path


def _summary_path(path: Path) -> str:
    """Human-facing path: repo-relative when possible, never leaking depth."""
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export every corpus model twice and verify byte-identical artifacts"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(DEFAULT_TARGET_DIR),
        help="directory of GNN markdown files (default: %(default)s)",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=Path(DEFAULT_RECEIPTS_DIR),
        help="directory for the receipt JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="comma-separated export formats (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    formats = _parse_formats(args.formats)
    target_dir = args.target_dir
    if not target_dir.is_absolute():
        target_dir = (_PROJECT_ROOT / target_dir).resolve()
    receipts_dir = args.receipts_dir
    if not receipts_dir.is_absolute():
        receipts_dir = (_PROJECT_ROOT / receipts_dir).resolve()

    receipt: dict[str, Any] = {}
    error: str | None = None
    try:
        if not target_dir.is_dir():
            raise FileNotFoundError(f"target directory does not exist: {target_dir}")
        receipt = build_receipt(target_dir, formats, _collect_corpus(target_dir))
    except Exception as exc:  # noqa: BLE001 - fail-closed receipt on any error
        error = f"{type(exc).__name__}: {exc}"
        receipt = {
            "receipt_version": RECEIPT_VERSION,
            "generator": GENERATOR,
            "counts_describe_commit": _describe_commit(_PROJECT_ROOT),
            "working_tree_dirty": False,
            "dirty_paths": [],
            "corpus": [],
            "artifacts_compared": 0,
            "mismatches": [],
            "parse_failures": [],
            "result": "no-parseable-corpus",
            "error": error,
        }

    try:
        receipt_path = _write_receipt(receipts_dir, receipt)
    except OSError as exc:
        print(f"receipt write failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    result = str(receipt.get("result", "no-parseable-corpus"))
    print(
        "exporter determinism: result=%s artifacts_compared=%s "
        "parse_failures=%s mismatches=%s"
        % (
            result,
            receipt.get("artifacts_compared", 0),
            len(receipt.get("parse_failures", [])),
            len(receipt.get("mismatches", [])),
        )
    )
    if error is not None:
        print(f"operational error: {error}", file=sys.stderr)
    print(f"receipt: {_summary_path(receipt_path)}")
    return 0 if result == "deterministic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
