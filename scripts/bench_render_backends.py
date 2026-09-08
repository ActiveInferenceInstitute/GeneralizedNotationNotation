#!/usr/bin/env python3
"""Deterministic render-backend conformance benchmark (autoresearch harness).

Workload: render every discovered GNN model under ``input/gnn_files`` for every
framework in ``FRAMEWORK_REGISTRY`` via the canonical Step 11 surface
(``gnn.render.processor.process_render``), then score the resulting render
receipt (``render_processing_summary.json``).

A successful rendering only counts toward the primary metric when its emitted
artifacts are conformant:

- every emitted ``.py`` artifact must ``compile()`` (syntax check);
- every emitted ``.py`` artifact passes a conservative undefined-name scan
  (Load-context names never bound anywhere in the module - runtime NameError
  risk the syntax check cannot see);
- every framework with an entry in ``gnn.render.contracts.CONTRACTS`` must
  satisfy its output contract on the artifact carrying the framework's
  canonical file extension.

Pure codegen: nothing is executed, no network access, fixed run_id. The same
corpus and framework set produce identical counts on every run.

Output: ``METRIC <name>=<value>`` lines on stdout (primary metric first),
followed by ``FAIL``/``CONFORMANCE`` diagnostic lines.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gnn.render.framework_registry import (  # noqa: E402
    FRAMEWORK_REGISTRY,
    get_supported_frameworks,
)
from gnn.render.processor import process_render  # noqa: E402

CORPUS = REPO_ROOT / "input" / "gnn_files"
RUN_ID = "bench"

MAX_DIAGNOSTICS_PER_RENDERING = 4

# Conservative AST undefined-name scan skips very large emitted artifacts
# (the PyMDP scaling-study exemplars expand dense B tensors as O(n^3) text);
# compile() still covers their syntax. Deterministic and logged, not silent.
UNDEFINED_SCAN_MAX_BYTES = 1_000_000

# Module-context names that are never bound by statements but always exist.
MODULE_CONTEXT_NAMES = frozenset(
    {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__"}
)


def _undefined_names(code: str) -> tuple[list[tuple[str, int]], bool]:
    """Conservative undefined-name scan for one Python artifact.

    Returns ``((name, line) findings, star_import_present)``. A name is
    reported when it appears in Load context, is never bound anywhere in the
    module (import/def/class/assignment/arg/except/global), and is not a
    builtin or module-context name. Binding anywhere counts, so the check is
    imprecise about scoping but has near-zero false positives - the right
    trade for a deterministic benchmark gate. When the module uses a star
    import the scan is skipped entirely (names cannot be resolved without
    executing the source module), reported via the second element.
    """
    import ast
    import builtins

    tree = ast.parse(code)
    bound: set[str] = set(MODULE_CONTEXT_NAMES)
    star_import = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    star_import = True
                    continue
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    if star_import:
        return [], True
    findings = sorted(
        {
            (node.id, node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in bound
            and not hasattr(builtins, node.id)
        },
        key=lambda item: (item[1], item[0]),
    )
    return findings, False


def _score_conformance(
    receipt: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """Score emitted artifacts of successful renderings for conformance.

    Returns:
        ``(conformance_failures, syntax_errors, contract_violations,
        undefined_name_findings, undefined_scan_skipped)`` where
        ``conformance_failures`` counts successful renderings with at least
        one nonconformant artifact; ``syntax_errors`` counts artifacts
        failing ``compile()``; ``contract_violations`` counts individual
        contract violations across validated artifacts;
        ``undefined_name_findings`` counts Load-context names never bound in
        emitted Python (runtime NameError risk); ``undefined_scan_skipped``
        counts size-guard skips.
    """
    from gnn.render.contracts import CONTRACTS, validate_rendered_output

    canonical_extensions = {
        name: str(spec.get("file_extension", ""))
        for name, spec in FRAMEWORK_REGISTRY.items()
    }
    conformance_failures = 0
    syntax_errors = 0
    contract_violations = 0
    undefined_name_findings = 0
    undefined_scan_skipped = 0
    for source, record in receipt["file_results"].items():
        if not isinstance(record, dict):
            continue
        source_name = Path(source).name
        for framework, result in record.get("framework_results", {}).items():
            if not isinstance(result, dict) or not result.get("success"):
                continue
            artifacts = [
                str(path)
                for path in result.get("output_files", [])
                if Path(str(path)).is_file()
            ]
            extension = canonical_extensions.get(framework, "")
            failures: list[str] = []
            for artifact in artifacts:
                path = Path(artifact)
                code = path.read_text(errors="replace")
                if path.suffix == ".py":
                    try:
                        compile(code, str(path), "exec")
                    except SyntaxError as exc:
                        syntax_errors += 1
                        failures.append(f"syntax: {exc.msg} (line {exc.lineno})")
                        continue
                    if path.stat().st_size > UNDEFINED_SCAN_MAX_BYTES:
                        undefined_scan_skipped += 1
                        print(
                            f"SCAN-SKIP {framework} {source_name}: "
                            f"artifact over {UNDEFINED_SCAN_MAX_BYTES} bytes"
                        )
                        continue
                    for name, lineno in _undefined_names(code)[0]:
                        undefined_name_findings += 1
                        failures.append(f"undefined-name: {name!r} (line {lineno})")
                if extension and path.suffix == extension and framework in CONTRACTS:
                    for violation in validate_rendered_output(
                        code, framework, file_path=str(path)
                    ):
                        contract_violations += 1
                        failures.append(f"contract: {violation}")
            if failures:
                conformance_failures += 1
                for failure in failures[:MAX_DIAGNOSTICS_PER_RENDERING]:
                    print(f"CONFORMANCE {framework} {source_name}: {failure}")
    return (
        conformance_failures,
        syntax_errors,
        contract_violations,
        undefined_name_findings,
        undefined_scan_skipped,
    )


def main() -> int:
    # Quiet the module's INFO chatter; METRIC lines stay parseable.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gnn").setLevel(logging.WARNING)

    frameworks = get_supported_frameworks()
    output_dir = Path(tempfile.mkdtemp(prefix="gnn-render-bench-"))

    started = time.perf_counter()
    process_render(
        CORPUS,
        output_dir,
        frameworks=frameworks,
        strict_validation=False,
        run_id=RUN_ID,
    )
    workload_ms = (time.perf_counter() - started) * 1000.0

    receipt_path = output_dir / "render_processing_summary.json"
    if not receipt_path.is_file():
        print(
            f"HARNESS ERROR: render receipt missing at {receipt_path}",
            file=sys.stderr,
        )
        return 1
    receipt = json.loads(receipt_path.read_text())

    attempts = int(receipt["total_framework_attempts"])
    rendered = int(receipt["successful_framework_renderings"])
    failed = receipt["failed_framework_renderings"]
    unsupported = receipt["unsupported_framework_renderings"]
    total_files = int(receipt["total_files"])
    successful_files = int(receipt["successful_files"])

    # Files that produced no framework attempts at all: strict-validation
    # skips, per-file crashes, and other silent render-path dropouts.
    no_attempt_files = sum(
        1
        for record in receipt["file_results"].values()
        if not isinstance(record, dict)
        or not any(
            isinstance(result, dict)
            for result in record.get("framework_results", {}).values()
        )
    )

    (
        conformance_failures,
        syntax_errors,
        contract_violation_count,
        undefined_name_findings,
        undefined_scan_skipped,
    ) = _score_conformance(receipt)
    conformance_success = rendered - conformance_failures
    success_rate = (rendered / attempts * 100.0) if attempts else 0.0

    print(f"METRIC render_conformance_success_count={conformance_success}")
    print(f"METRIC render_success_count={rendered}")
    print(f"METRIC render_success_rate={success_rate:.2f}")
    print(f"METRIC render_attempts={attempts}")
    print(f"METRIC render_errors={len(failed)}")
    print(f"METRIC render_unsupported={len(unsupported)}")
    print(f"METRIC render_syntax_errors={syntax_errors}")
    print(f"METRIC render_contract_violations={contract_violation_count}")
    print(f"METRIC render_undefined_name_findings={undefined_name_findings}")
    print(f"METRIC render_undefined_scan_skipped={undefined_scan_skipped}")
    print(f"METRIC render_files_total={total_files}")
    print(f"METRIC render_files_successful={successful_files}")
    print(f"METRIC render_files_no_attempts={no_attempt_files}")
    print(f"METRIC render_workload_ms={workload_ms:.1f}")

    for diag in failed[:25]:
        message = str(diag.get("message", "")).replace("\n", " ")[:220]
        file_name = Path(str(diag.get("file", "?"))).name
        print(f"FAIL {diag.get('framework', '?')} {file_name}: {message}")
    if len(failed) > 25:
        print(f"FAIL ... {len(failed) - 25} more failures omitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
