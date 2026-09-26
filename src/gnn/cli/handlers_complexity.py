#!/usr/bin/env python3
"""
Complexity subcommand handlers: complexity and benchmark.

Each handler receives the parsed argparse namespace and returns a process
exit code per the CLI contract (0 success, 1 error, 2 warnings). Extracted
from ``cli.__init__``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .helpers import (
    EXIT_ERROR,
    EXIT_SUCCESS,
    EXIT_WARNING,
    _print_envelope,
)

logger = logging.getLogger(__name__)

#: Terminal calibration-note column width; full note stays in the receipt.
_NOTE_WIDTH = 60


def _truncate_note(note: Any, width: int = _NOTE_WIDTH) -> str:
    """Shorten a calibration note for one table column (never fabricate)."""
    text = " ".join(str(note or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _print_complexity_table(receipts: list[dict[str, Any]]) -> None:
    """Print the static bounds table (ESTIMATE labels; never measurements)."""
    print("📄 Static complexity bounds — not measurements")
    for receipt in receipts:
        model = receipt.get("model") or {}
        model_name = str(model.get("name") or "<unnamed>")
        kinds = ", ".join(str(k) for k in receipt.get("model_kinds") or [])
        print(f"\n🧮 {model_name} (model_kinds: {kinds})")
        print("  framework          | applicable | class             | asymptotic")
        for row in receipt.get("per_backend") or []:
            applicable = "yes" if row.get("applicable") else "no"
            complexity_class = str(row.get("complexity_class") or "")
            asymptotic = " ".join(str(row.get("asymptotic") or "").split())
            print(
                f"  {str(row.get('framework') or ''):<18} | "
                f"{applicable:<10} | {complexity_class:<17} | {asymptotic}"
            )


def _estimate_path_receipts(path: Path) -> list[dict[str, Any]]:
    """Resolve a model path to static receipts via the harness seam.

    Primary: ``gnn.analysis.complexity.benchmark.estimate_path_complexity``
    (the pinned cross-lane seam). Fallback while that module is in flight
    (sibling-owned): walk the path with the static estimator directly —
    same pinned ``gnn.complexity_estimate/v1`` receipt schema and join
    keys, so the CLI surface behaves identically once the seam lands.
    """
    try:
        from gnn.analysis.complexity.benchmark import estimate_path_complexity
    except ImportError:
        logger.warning(
            "gnn.analysis.complexity.benchmark not landed; using the static"
            " estimator seam directly"
        )
        from gnn.analysis.complexity import estimate_model_complexity

        if path.is_file():
            return [estimate_model_complexity(path)]
        return [estimate_model_complexity(p) for p in sorted(path.glob("*.md"))]
    return estimate_path_complexity(path)


def _cmd_complexity(args: argparse.Namespace) -> int:
    """Estimate static per-backend complexity bounds for a model or directory."""
    is_json = getattr(args, "json", False)
    path = Path(args.path)
    if not path.exists():
        message = f"GNN model path not found: {path}"
        logger.error("%s", message)
        if is_json:
            _print_envelope("error", error=message, command="complexity")
        else:
            print(f"❌ {message}")
        return EXIT_ERROR

    # Heavy imports stay inside the handler (lazy-import convention).
    from gnn.analysis.complexity import to_json_text

    try:
        receipts = _estimate_path_receipts(path)
    except Exception as exc:  # estimator contract is parse-error raising
        message = f"static complexity estimation failed for {path}: {exc}"
        logger.error("%s", message)
        if is_json:
            _print_envelope("error", error=message, command="complexity")
        else:
            print(f"❌ {message}")
        return EXIT_ERROR
    if not receipts:
        message = f"no GNN models found under: {path}"
        logger.error("%s", message)
        if is_json:
            _print_envelope("error", error=message, command="complexity")
        else:
            print(f"❌ {message}")
        return EXIT_ERROR

    is_dir = path.is_dir()
    output_path = Path(args.output) if getattr(args, "output", None) else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = {"receipts": receipts} if is_dir else receipts[0]
        output_path.write_text(to_json_text(payload), encoding="utf-8")

    if is_json:
        data: dict[str, Any] = (
            {"receipts": receipts} if is_dir else {"receipt": receipts[0]}
        )
        if output_path is not None:
            data["output_path"] = str(output_path)
        _print_envelope("success", data=data, command="complexity")
    else:
        _print_complexity_table(receipts)
        if output_path is not None:
            print(f"\n📄 Receipt written to: {output_path}")
    return EXIT_SUCCESS


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the empirical complexity benchmark harness over a corpus directory."""
    is_json = getattr(args, "json", False)
    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        message = f"Benchmark corpus directory not found: {target_dir}"
        logger.error("%s", message)
        if is_json:
            _print_envelope("error", error=message, command="benchmark")
        else:
            print(f"❌ {message}")
        return EXIT_ERROR

    output_dir = Path(args.output_dir)
    try:
        from gnn.analysis.complexity.benchmark import run_complexity_benchmark
    except ImportError as exc:
        message = f"benchmark harness unavailable: {exc}"
        logger.error("%s", message)
        if is_json:
            _print_envelope("error", error=message, command="benchmark")
        else:
            print(f"❌ {message}")
        return EXIT_ERROR

    try:
        result = run_complexity_benchmark(
            target_dir,
            output_dir,
            frameworks=str(args.frameworks),
            repeats=int(args.repeats),
        )
    except Exception as exc:
        message = f"complexity benchmark failed: {exc}"
        logger.error("%s", message)
        if is_json:
            _print_envelope("error", error=message, command="benchmark")
        else:
            print(f"❌ {message}")
        return EXIT_ERROR

    result_dict: dict[str, Any] = result if isinstance(result, dict) else {}
    status = str(result_dict.get("status") or "")
    # Measurement rows carry the availability flag; calibration rows carry
    # the joined static-vs-measured table. Each source feeds its own output.
    rows = list(result_dict.get("rows") or [])
    unavailable_rows = [row for row in rows if not row.get("available", True)]
    calibration_path = result_dict.get("calibration_receipt")
    calibration_rows: list[dict[str, Any]] = []
    if calibration_path:
        try:
            receipt = json.loads(Path(str(calibration_path)).read_text())
            calibration_rows = list(receipt.get("rows") or [])
        except (OSError, ValueError) as exc:
            message = f"calibration receipt unreadable at {calibration_path}: {exc}"
            logger.error("%s", message)
            if is_json:
                _print_envelope("error", error=message, command="benchmark")
            else:
                print(f"❌ {message}")
            return EXIT_ERROR

    if is_json:
        _print_envelope(
            "warning" if unavailable_rows else "success",
            data={
                "benchmark_receipt": result_dict.get("benchmark_receipt"),
                "calibration_receipt": result_dict.get("calibration_receipt"),
                "rows": rows,
                "status": status or None,
            },
            error=(
                f"{len(unavailable_rows)} backend row(s) reported available:false"
                if unavailable_rows
                else None
            ),
            command="benchmark",
        )
    else:
        print("📊 Complexity benchmark — empirical measurements")
        print(
            "  model              | framework          | applicable | class      "
            "| wall med (s) | peak RSS (MB) | reps | note"
        )
        for row in calibration_rows:
            wall = row.get("wall_median_seconds")
            wall_s = "null" if wall is None else f"{float(wall):.3f}"
            rss = row.get("peak_rss_mb")
            rss_s = "null" if rss is None else f"{float(rss):.2f}"
            print(
                f"  {str(row.get('model_name') or ''):<18} | "
                f"{str(row.get('framework') or ''):<18} | "
                f"{'yes' if row.get('applicable') else 'no':<10} | "
                f"{str(row.get('complexity_class') or ''):<10} | "
                f"{wall_s:>12} | {rss_s:>13} | "
                f"{str(row.get('repeats') or ''):>4} | "
                f"{_truncate_note(row.get('calibration_note'))}"
            )
        if unavailable_rows:
            print(
                f"\n⚠️ {len(unavailable_rows)} backend row(s) recorded "
                "available:false (never silently skipped)"
            )

    if status.startswith("success"):
        # Harness success family (success / success_with_skips /
        # success_with_failures); skips surface as the warning below.
        return EXIT_WARNING if unavailable_rows else EXIT_SUCCESS
    logger.error("Benchmark harness reported status: %s", status or "unknown")
    return EXIT_ERROR
