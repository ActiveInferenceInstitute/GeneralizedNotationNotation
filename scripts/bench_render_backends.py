#!/usr/bin/env python3
"""Deterministic render-backend conformance benchmark (autoresearch harness).

A successful rendering only counts toward the primary metric when its emitted
artifacts are conformant:

- every emitted ``.py`` artifact must ``compile()`` (syntax check);
- every emitted ``.py`` artifact passes the conservative undefined-name scan
  from ``gnn.render.emitted_artifact_checks`` (Load-context names never bound
  anywhere in the module - runtime NameError risk the syntax check cannot
  see);
- every framework with an entry in ``gnn.render.contracts.CONTRACTS`` must
  satisfy its output contract on the artifact carrying the framework's
  canonical file extension.

The receipt itself is also verified (``render_receipt_integrity_findings``):

- every successful rendering records at least one artifact and every
  recorded artifact exists on disk;
- every recorded ``artifact_identities`` sha256 matches the on-disk bytes
  and every ``output_files`` entry has a recorded identity;
- every ``unsupported`` diagnostic names a backend the registry marks
  ``supports_continuous: False``, and every continuous-capable backend
  actually succeeded on the models reported unsupported elsewhere.

Pure codegen: nothing is executed, no network access, fixed run_id. The same
corpus and framework set produce identical counts on every run.

Output: ``METRIC <name>=<value>`` lines on stdout (primary metric first),
followed by ``FAIL``/``CONFORMANCE``/``INTEGRITY`` diagnostic lines.
"""

from __future__ import annotations

import hashlib
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

# The conservative AST undefined-name scan skips very large emitted artifacts
# (the PyMDP scaling-study exemplars expand dense B tensors as O(n^3) text);
# compile() still covers their syntax. Deterministic and logged, not silent.
UNDEFINED_SCAN_MAX_BYTES = 1_000_000


def _score_conformance(
    receipt: dict[str, Any],
) -> tuple[int, int, int, int, int, int]:
    """Score emitted artifacts of successful renderings for conformance.

    Returns:
        ``(conformance_failures, syntax_errors, contract_violations,
        undefined_name_findings, undefined_scan_skipped,
        first_party_import_findings)`` where
        ``conformance_failures`` counts successful renderings with at least
        one nonconformant artifact; ``syntax_errors`` counts artifacts
        failing ``compile()``; ``contract_violations`` counts individual
        contract violations across validated artifacts;
        ``undefined_name_findings`` counts Load-context names never bound in
        emitted Python (runtime NameError risk); ``undefined_scan_skipped``
        counts size-guard skips; ``first_party_import_findings`` counts
        ``gnn.*`` imports in emitted Python that do not resolve from the
        repository (stale-template drift; runtime ImportError risk).
    """
    from gnn.render.contracts import CONTRACTS, validate_rendered_output
    from gnn.render.emitted_artifact_checks import (
        first_party_unresolvable_imports,
        undefined_names,
    )

    canonical_extensions = {
        name: str(spec.get("file_extension", ""))
        for name, spec in FRAMEWORK_REGISTRY.items()
    }
    conformance_failures = 0
    syntax_errors = 0
    contract_violations = 0
    undefined_name_findings = 0
    undefined_scan_skipped = 0
    first_party_import_findings = 0
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
                    findings, _star = undefined_names(code)
                    for name, lineno in findings:
                        undefined_name_findings += 1
                        failures.append(f"undefined-name: {name!r} (line {lineno})")
                    for module, lineno in first_party_unresolvable_imports(code):
                        first_party_import_findings += 1
                        failures.append(f"unresolved-import: {module} (line {lineno})")
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
        first_party_import_findings,
    )


def _verify_receipt_integrity(
    receipt: dict[str, Any],
) -> int:
    """Verify the receipt's own claims; returns the integrity finding count.

    Checks (per successful framework rendering, plus receipt-wide):
      1. success implies at least one recorded artifact, and every recorded
         ``output_files`` path exists on disk;
      2. every ``artifact_identities`` entry's sha256 matches the on-disk
         bytes, and every ``output_files`` entry has a recorded identity
         (the writer only identities existing files - a listed-but-vanished
         artifact shows up as a missing identity);
      3. every ``unsupported`` diagnostic names a backend the registry marks
         ``supports_continuous: False``;
      4. every continuous-capable backend actually succeeded on each model
         reported unsupported elsewhere.
    """
    findings = 0

    def flag(message: str) -> None:
        nonlocal findings
        findings += 1
        print(f"INTEGRITY {message}")

    for source, record in receipt["file_results"].items():
        if not isinstance(record, dict):
            continue
        source_name = Path(source).name
        for framework, result in record.get("framework_results", {}).items():
            if not isinstance(result, dict) or not result.get("success"):
                continue
            output_files = [str(path) for path in result.get("output_files", [])]
            if not output_files:
                flag(f"{framework} {source_name}: success without artifacts")
                continue
            # The writer keys identities by path.resolve(); output_files
            # entries carry unresolved paths (/var/... vs /private/var/... on
            # macOS), so resolve before lookup.
            identities = {
                str(Path(str(entry.get("path"))).resolve()): str(entry.get("sha256"))
                for entry in result.get("artifact_identities", [])
                if isinstance(entry, dict)
            }
            for artifact in output_files:
                path = Path(artifact)
                if not path.is_file():
                    flag(
                        f"{framework} {source_name}: recorded artifact "
                        f"missing on disk: {path.name}"
                    )
                    continue
                resolved = str(path.resolve())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if resolved not in identities:
                    flag(
                        f"{framework} {source_name}: no identity recorded "
                        f"for artifact {path.name}"
                    )
                elif identities[resolved] != digest:
                    flag(
                        f"{framework} {source_name}: identity sha256 "
                        f"mismatch for artifact {path.name}"
                    )

    registry = FRAMEWORK_REGISTRY
    unsupported_pairs = {
        (str(diag.get("file")), str(diag.get("framework")))
        for diag in receipt.get("unsupported_framework_renderings", [])
        if isinstance(diag, dict)
    }
    unsupported_files = {file for file, _framework in unsupported_pairs}
    for file, framework in unsupported_pairs:
        spec = registry.get(framework)
        if spec is None:
            flag(f"unsupported diagnostic names unknown framework {framework}")
        elif spec.get("supports_continuous", False):
            flag(
                f"{framework} claims unsupported but registry says "
                "supports_continuous=True"
            )
    for file in unsupported_files:
        record = receipt["file_results"].get(file)
        if not isinstance(record, dict):
            continue
        for framework, spec in registry.items():
            if not spec.get("supports_continuous", False):
                continue
            result = record.get("framework_results", {}).get(framework)
            if not isinstance(result, dict) or not result.get("success"):
                flag(
                    f"{Path(file).name}: continuous-capable backend "
                    f"{framework} did not succeed on an unsupported-"
                    "elsewhere model"
                )
    return findings


def _determinism_mismatches(
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[int, list[str]]:
    """Compare two receipts of the SAME workload for byte-identical artifacts.

    For every successful (source, framework) rendering present in both
    receipts, the recorded ``output_files`` lists must match by basename and
    every artifact must be byte-identical across the two runs. Returns
    ``(mismatch_count, diagnostics)`` - a nonzero count means the renderer
    embeds wall-clock or otherwise run-dependent values in artifact bytes.
    """
    import hashlib

    diagnostics: list[str] = []
    mismatches = 0

    def artifact_map(receipt: dict[str, Any]) -> dict[tuple[str, str], list[Path]]:
        mapping: dict[tuple[str, str], list[Path]] = {}
        for source, record in receipt["file_results"].items():
            if not isinstance(record, dict):
                continue
            for framework, result in record.get("framework_results", {}).items():
                if not isinstance(result, dict) or not result.get("success"):
                    continue
                key = (str(Path(source).name), str(framework))
                mapping[key] = [
                    Path(str(path))
                    for path in result.get("output_files", [])
                    if Path(str(path)).is_file()
                ]
        return mapping

    first_map = artifact_map(first)
    second_map = artifact_map(second)
    for key in sorted(set(first_map) & set(second_map)):
        first_files = sorted(path.name for path in first_map[key])
        second_files = sorted(path.name for path in second_map[key])
        source_name, framework = key
        if first_files != second_files:
            mismatches += 1
            diagnostics.append(f"{framework} {source_name}: artifact file set differs")
            continue
        for name in first_files:
            first_bytes = next(
                path.read_bytes() for path in first_map[key] if path.name == name
            )
            second_bytes = next(
                path.read_bytes() for path in second_map[key] if path.name == name
            )
            if first_bytes != second_bytes:
                mismatches += 1
                first_digest = hashlib.sha256(first_bytes).hexdigest()[:12]
                second_digest = hashlib.sha256(second_bytes).hexdigest()[:12]
                diagnostics.append(
                    f"{framework} {source_name}: {name} differs "
                    f"({first_digest} vs {second_digest})"
                )
    return mismatches, diagnostics


def _parity_mismatches(receipt: dict[str, Any]) -> tuple[int, list[str]]:
    """Compare extracted A/B/C/D matrix shapes across backends per model.

    For every source rendered successfully by at least two Python-emitting
    backends (pymdp, jax, pytorch, numpyro), the extracted canonical matrix
    shapes must agree; the Julia backends' dimension constants
    (NUM_STATES/OBSERVATIONS/ACTIONS) must agree with the Python-derived
    dims. Shapes come from the conservative extractor in
    ``gnn.render.emitted_artifact_checks.matrix_shapes``; sources whose
    artifacts yield no clean literal shapes are simply not compared.
    Returns ``(mismatch_count, diagnostics)``.
    """
    from gnn.render.emitted_artifact_checks import (
        julia_dimension_constants,
        julia_render_factorization,
        matrix_shapes,
    )

    per_source: dict[str, dict[str, dict[str, tuple[int, ...]]]] = {}
    for source, record in receipt["file_results"].items():
        if not isinstance(record, dict):
            continue
        source_name = Path(source).name
        for framework, result in record.get("framework_results", {}).items():
            if not isinstance(result, dict) or not result.get("success"):
                continue
            for artifact in result.get("output_files", []):
                path = Path(str(artifact))
                if (
                    path.suffix != ".py"
                    or not path.is_file()
                    or path.stat().st_size > UNDEFINED_SCAN_MAX_BYTES
                ):
                    continue
                shapes = matrix_shapes(path.read_text(errors="replace"))
                if shapes:
                    per_source.setdefault(source_name, {}).setdefault(
                        framework, {}
                    ).update(shapes)
                break

    # Julia backends (rxinfer, activeinference_jl) allocate from dimension
    # constants; check them against the Python backends' derived dims
    # (n = A columns, m = A rows, k = B action axis).
    julia_by_source: dict[str, dict[str, dict[str, int]]] = {}
    for source, record in receipt["file_results"].items():
        if not isinstance(record, dict):
            continue
        source_name = Path(source).name
        for framework, result in record.get("framework_results", {}).items():
            if framework not in ("rxinfer", "activeinference_jl"):
                continue
            if not isinstance(result, dict) or not result.get("success"):
                continue
            for artifact in result.get("output_files", []):
                path = Path(str(artifact))
                if path.suffix != ".jl" or not path.is_file():
                    continue
                artifact_text = path.read_text(errors="replace")
                if julia_render_factorization(artifact_text) == "hierarchical":
                    # Native per-level factorized render: its dimension
                    # constants describe the fast level only, not the joint
                    # expansion the Python backends emit.
                    break
                constants = julia_dimension_constants(artifact_text)
                if constants:
                    julia_by_source.setdefault(source_name, {})[framework] = constants
                break

    diagnostics: list[str] = []
    mismatches = 0
    for source_name in sorted(per_source):
        entries = per_source[source_name]
        for letter in "ABCD":
            per_backend = {
                framework: shapes[letter]
                for framework, shapes in entries.items()
                if letter in shapes
            }
            if len(per_backend) < 2:
                continue
            if len(set(per_backend.values())) > 1:
                mismatches += 1
                diagnostics.append(f"{source_name} {letter}: {per_backend}")
    for source_name, julia_entries in sorted(julia_by_source.items()):
        derived = {}
        for shapes in per_source.get(source_name, {}).values():
            if "A" in shapes and "B" in shapes:
                derived = {
                    "NUM_STATES": shapes["A"][1],
                    "NUM_OBSERVATIONS": shapes["A"][0],
                    "NUM_ACTIONS": shapes["B"][2] if len(shapes["B"]) == 3 else 1,
                }
                break
        if not derived:
            continue
        for framework, constants in sorted(julia_entries.items()):
            for key, expected in derived.items():
                actual = constants.get(key)
                if actual is not None and actual != expected:
                    mismatches += 1
                    diagnostics.append(
                        f"{source_name} {framework}: {key}={actual} "
                        f"but Python backends imply {expected}"
                    )
    return mismatches, diagnostics


def main() -> int:
    # Quiet the module's INFO chatter; METRIC lines stay parseable.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("gnn").setLevel(logging.WARNING)

    frameworks = get_supported_frameworks()
    output_dir = Path(tempfile.mkdtemp(prefix="gnn-render-bench-"))

    started = time.perf_counter()
    output_dir_second = Path(tempfile.mkdtemp(prefix="gnn-render-bench-2-"))
    process_render(
        CORPUS,
        output_dir,
        frameworks=frameworks,
        strict_validation=False,
        run_id=RUN_ID,
    )
    process_render(
        CORPUS,
        output_dir_second,
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
    receipt_second_path = output_dir_second / "render_processing_summary.json"
    receipt_second = (
        json.loads(receipt_second_path.read_text())
        if receipt_second_path.is_file()
        else {}
    )

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
        first_party_import_findings,
    ) = _score_conformance(receipt)
    determinism_mismatches, determinism_diagnostics = _determinism_mismatches(
        receipt, receipt_second
    )
    receipt_integrity_findings = _verify_receipt_integrity(receipt)
    parity_mismatches, parity_diagnostics = _parity_mismatches(receipt)
    conformance_success = rendered - conformance_failures
    success_rate = (rendered / attempts * 100.0) if attempts else 0.0

    print(f"METRIC render_conformance_success_count={conformance_success}")
    print(f"METRIC render_success_count={rendered}")
    print(f"METRIC render_success_rate={success_rate:.2f}")
    print(f"METRIC render_attempts={attempts}")
    print(f"METRIC render_receipt_integrity_findings={receipt_integrity_findings}")
    print(f"METRIC render_determinism_mismatches={determinism_mismatches}")
    print(f"METRIC render_unsupported={len(unsupported)}")
    print(f"METRIC render_syntax_errors={syntax_errors}")
    print(f"METRIC render_contract_violations={contract_violation_count}")
    print(f"METRIC render_undefined_name_findings={undefined_name_findings}")
    print(f"METRIC render_undefined_scan_skipped={undefined_scan_skipped}")
    print(f"METRIC render_first_party_import_findings={first_party_import_findings}")
    print(f"METRIC render_parity_mismatches={parity_mismatches}")
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
    for diag in parity_diagnostics[:25]:
        print(f"PARITY {diag}")
    for diag in determinism_diagnostics[:25]:
        print(f"NONDETERMINISM {diag}")
    if len(determinism_diagnostics) > 25:
        print(f"NONDETERMINISM ... {len(determinism_diagnostics) - 25} more omitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
