"""Source-bound lightweight processing measurements, independent of native stacks.

Unprofiled wall/CPU samples and deterministic tracing are separate runs. Input
bookends detect endpoint drift, not hostile concurrent replacement or ABA edits.
The profile has finite repeat/input admission limits, not a hard wall timeout.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from gnn.processing.processor import (
    _atomic_write_json,
    discover_gnn_files,
    process_gnn_directory_lightweight,
)
from gnn.utils.file_workers import effective_file_workers, resolve_file_workers

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_INPUT_FILES = 256


def _normalize(value: Any) -> Any:
    """Remove volatile timestamps and worker PIDs; preserve model data and order."""
    if isinstance(value, dict):
        normalized = {
            key: _normalize(item)
            for key, item in value.items()
            if key not in {"timestamp", "parse_timestamp", "validation_timestamp"}
        }
        execution = normalized.get("worker_execution")
        if isinstance(execution, dict):
            execution.pop("worker_pids", None)
        return normalized
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _model_digest(value: dict[str, Any]) -> str:
    """Compare model outputs across worker selections, retaining controls separately."""
    normalized = _normalize(value)
    normalized.pop("worker_execution", None)
    return _digest(normalized)


def _backend_inventory() -> dict[str, str]:
    """Bind the processing owners and shared file-worker policy actually used."""
    directory = Path(__file__).parent
    paths = [*sorted(directory.glob("*.py")), directory / "../utils/file_workers.py"]
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _child_cpu_time() -> float | None:
    """Read terminated-child CPU usage where the host exposes POSIX resource data."""
    try:
        resource = importlib.import_module("resource")
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _inventory(target: Path) -> list[dict[str, Any]]:
    files = discover_gnn_files(target, recursive=True)
    if not files or len(files) > MAX_INPUT_FILES:
        raise ValueError(
            f"Profiling requires 1..{MAX_INPUT_FILES} selected model files"
        )
    result = []
    total = 0
    for path in files:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                total += len(block)
                size += len(block)
                if total > MAX_INPUT_BYTES:
                    raise ValueError(
                        f"Selected profiling input exceeds {MAX_INPUT_BYTES} bytes"
                    )
                digest.update(block)
        result.append(
            {"path": str(path), "size_bytes": size, "sha256": digest.hexdigest()}
        )
    return result


def profile_directory(
    target_dir: Path,
    output_dir: Path,
    *,
    repeats: int = 5,
    workers: int | None = None,
    executor: str | None = None,
) -> dict[str, Any]:
    """Measure complete parse/validation runs and retain a trusted local trace.

    Args:
        target_dir: Authored model directory or one supported model source.
        output_dir: New destination, outside the selected input directory.
        repeats: Number of unprofiled samples, from 1 to 25; one warmup is extra.
        workers: Explicit file-worker limit; omitted preserves serial processing.
        executor: Process or thread executor; explicit workers default to processes.

    Returns:
        JSON-compatible report with input/source/runtime bindings, all samples,
        result digest, distinct trace and unprofiled timing summaries.

    Raises:
        ValueError: Invalid admission, model failure, result or source drift.
        OSError: Input/output failure or an already existing destination.
    """
    if type(repeats) is not int or not 1 <= repeats <= 25:
        raise ValueError("repeats must be an integer from 1 to 25")
    effective_workers, selected_executor = resolve_file_workers(workers, executor)
    target = Path(target_dir).resolve()
    output = Path(output_dir).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if output == target or target in output.parents:
        raise ValueError("Profile output must be outside the selected input")
    before = _inventory(target)
    effective_workers = effective_file_workers(
        min(effective_workers, len(before)), selected_executor
    )
    source_before = _backend_inventory()
    output.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "workload": "lightweight_parse_and_validate",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "target": str(target),
        "inputs": before,
        "backend_source_sha256": source_before,
        "controls": {
            "repeats": repeats,
            "warmups": 1,
            "parallel": effective_workers > 1,
            "requested_workers": workers,
            "effective_workers": effective_workers,
            "requested_executor": executor,
            "executor": selected_executor if effective_workers > 1 else "serial",
            "max_input_files": MAX_INPUT_FILES,
            "max_input_bytes": MAX_INPUT_BYTES,
        },
        "unprofiled_samples": samples,
        "cpu_timing_scope": {
            "cpu_seconds": "calling process only; excludes worker process CPU",
            "children_cpu_seconds": "terminated children of this process; POSIX hosts only",
        },
        "normalization": {
            "removed_volatile_fields": [
                "timestamp",
                "parse_timestamp",
                "validation_timestamp",
                "worker_execution.worker_pids",
            ],
            "model_digest_additionally_excludes": ["worker_execution"],
        },
        "scope": "Lightweight processing only; no framework execution, local LLM generation, whole-pipeline acceleration or full Python-version compatibility claim.",
    }

    def run_once() -> dict[str, Any]:
        result = process_gnn_directory_lightweight(
            target, recursive=True, workers=workers, executor=executor
        )
        if not result["success"] or result["files_processed"] != len(before):
            raise ValueError(
                f"Selected processing did not complete: {result['errors']}"
            )
        return result

    try:
        started = time.perf_counter()
        first = run_once()
        report["warmup_seconds"] = time.perf_counter() - started
        expected = _digest(_normalize(first))
        expected_model = _model_digest(first)
        report["worker_execution"] = first.get("worker_execution")
        for _ in range(repeats):
            children_before = _child_cpu_time()
            wall = time.perf_counter()
            cpu = time.process_time()
            current = run_once()
            elapsed_cpu = time.process_time() - cpu
            elapsed_wall = time.perf_counter() - wall
            children_after = _child_cpu_time()
            samples.append(
                {
                    "wall_seconds": elapsed_wall,
                    "cpu_seconds": elapsed_cpu,
                    "children_cpu_seconds": (
                        children_after - children_before
                        if children_before is not None and children_after is not None
                        else None
                    ),
                    "worker_execution": current.get("worker_execution"),
                }
            )
            if _digest(_normalize(current)) != expected:
                raise ValueError("Repeated processing results changed")
        tracing = importlib.import_module(
            "profiling.tracing" if sys.version_info >= (3, 15) else "cProfile"
        )
        profiler = tracing.Profile()
        traced = profiler.runcall(run_once)
        if _digest(_normalize(traced)) != expected:
            raise ValueError("Traced processing results changed")
        profiler.dump_stats(str(output / "trace.pstats"))
        import pstats

        with (output / "trace.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(30)
        if _inventory(target) != before:
            raise ValueError("Selected input inventory changed during profiling")
        source_after = _backend_inventory()
        if source_after != source_before:
            raise ValueError("Backend source changed during profiling")
        walls = [sample["wall_seconds"] for sample in samples]
        report.update(
            status="success",
            files_processed=first["files_processed"],
            normalized_result_sha256=expected,
            normalized_model_result_sha256=expected_model,
            input_bookends_equal=True,
            backend_source_bookends_equal=True,
            median_wall_seconds=statistics.median(walls),
            min_wall_seconds=min(walls),
            max_wall_seconds=max(walls),
            trace={
                "module": tracing.__name__,
                "additional_runs": 1,
                "stats": "trace.pstats",
                "text": "trace.txt",
                "included_in_timing_samples": False,
                "worker_execution": traced.get("worker_execution"),
                "coverage": "calling thread only",
                "worker_execution_included": effective_workers <= 1,
                "warning": (
                    "Worker threads/processes are not traced; parent wait/dispatch time is not worker CPU time."
                    if effective_workers > 1
                    else ""
                ),
            },
        )
        _atomic_write_json(output / "profile_report.json", report)
        return report
    except (OSError, ValueError, RuntimeError, KeyboardInterrupt) as exc:
        report.update(
            status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _atomic_write_json(output / "profile_report.json", report)
        raise
