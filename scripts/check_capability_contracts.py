#!/usr/bin/env python3
"""Audit roadmap-visible capability claims against source support."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def _contains(path: str, patterns: Iterable[str]) -> bool:
    text = _read(path)
    return all(pattern in text for pattern in patterns)


def _maintained_test_directory_counts() -> tuple[int, int]:
    tests_root = REPO_ROOT / "src" / "tests"
    directories = [
        path
        for path in tests_root.iterdir()
        if path.is_dir() and not path.name.startswith("__")
    ]
    with_direct_tests = [path for path in directories if any(path.glob("test_*.py"))]
    return len(directories), len(with_direct_tests)


def run_audit() -> List[str]:
    """Return a list of contract violations."""
    failures: List[str] = []

    stale_patterns = (
        "Last Updated**: 2026-05-08",
        "2,250 passed",
        "7 skipped, 1 xpassed",
    )
    for doc in (
        "TO-DO.md",
        "DOCS.md",
        "ARCHITECTURE.md",
        ".agent_rules/README.md",
        "doc/gnn/README.md",
        "doc/gnn/modules/02_tests.md",
        "src/AGENTS.md",
        "src/tests/TEST_SUITE_SUMMARY.md",
    ):
        text = _read(doc)
        for pattern in stale_patterns:
            if pattern in text:
                failures.append(f"{doc}: stale measured-doc pattern remains: {pattern}")

    improvement_text = _read("doc/gnn/operations/improvement_analysis.md")
    for pattern in (
        "100% success rate",
        "Execution time: ~2 minutes",
        "Week 1:",
        "Week 2:",
        "Implementation Plan",
        "Implementation Timeline",
    ):
        if pattern in improvement_text:
            failures.append(
                "doc/gnn/operations/improvement_analysis.md: "
                f"stale live-status wording remains: {pattern}"
            )

    cli_text = _read("src/cli/__init__.py")
    for command in ('add_parser("templates"', 'add_parser("show"', 'add_parser("pull"'):
        if command not in cli_text:
            failures.append(
                f"src/cli/__init__.py: missing CLI command contract {command}"
            )
    if not _exists("src/cli/templates.py"):
        failures.append("src/cli/templates.py: template library implementation missing")
    else:
        templates_text = _read("src/cli/templates.py")
        for required in (
            "importlib import resources",
            "_validate_template_record",
            "template_assets",
            "PurePosixPath",
            "destination.is_symlink()",
        ):
            if required not in templates_text:
                failures.append(
                    f"src/cli/templates.py: missing package-data/template safety guard {required}"
                )
    if not _exists("src/cli/template_index.json"):
        failures.append("src/cli/template_index.json: external template index missing")
    elif "pomdp-gridworld-3x3" not in _read("src/cli/template_index.json"):
        failures.append("src/cli/template_index.json: gridworld template missing")

    todo_text = _read("TO-DO.md")
    if (
        "**Next Target**: v1.8.0" in todo_text
        and "v1.7.0 remains foundation-only/deferred" not in todo_text
    ):
        failures.append(
            "TO-DO.md: v1.8.0 is next while v1.7.0 remains unchecked without an explicit deferred/foundation-only status"
        )
    if (
        "**Current Version**: 1.9.0" in todo_text
        and "**Next Target**: v2.0.0" not in todo_text
    ):
        failures.append("TO-DO.md: v1.9.0 release must set v2.0.0 as next target")
    if (
        "**Current Version**: 2.0.0" in todo_text
        and "**Next Target**: v3.0.0" not in todo_text
    ):
        failures.append("TO-DO.md: v2.0.0 release must set v3.0.0 as next target")

    readme_tests = _read("src/tests/README.md")
    maintained_dirs, direct_test_dirs = _maintained_test_directory_counts()
    expected_count_text = (
        f"{maintained_dirs} maintained first-level subdirectories; "
        f"{direct_test_dirs} contain direct test files"
    )
    if expected_count_text not in readme_tests:
        failures.append(
            "src/tests/README.md: maintained test-directory count drift; "
            f"expected '{expected_count_text}'"
        )

    main_acceptance_without_output_dir = _main_commands_missing_isolated_output_dir(
        todo_text
    )
    for command in main_acceptance_without_output_dir:
        failures.append(
            "TO-DO.md: acceptance pipeline command must use an isolated /tmp output dir: "
            f"{command}"
        )

    guarded_pending_items = (
        "Multi-Agent Message Passing (RxInfer)",
        "Categorical Symmetries (DisCoPy)",
        "Reactive WebSocket GUI",
        "Audio Parameter Streaming",
        "3D Matrix Visualization",
        "GNN Template Library Engine",
        "MCP Local HTTP Orchestration",
        "Model-Family Acceptance Harness",
        "Cross-Step Evidence Ledger",
        "Interpretability Summaries",
        "Semantic Round-Trip Gates",
        "Cross-Framework Result Comparisons",
        "Release Readiness Ledger",
        "Durable Observation Streams",
        "Long-Running Pipeline Sessions",
        "Auditable Container Plans",
        "Autonomous Candidate Scoring",
        "Reviewed Self-Editing GNN Files",
        "Autonomous Ecology Controls",
    )
    release_evidence_by_item = {
        "GNN Template Library Engine": (
            "gnn templates list",
            "gnn templates show pomdp-gridworld-3x3",
            "gnn pull` to `/tmp/gnn-pull",
            "combined CLI/MCP/capability suite `32 passed`",
        ),
        "MCP Local HTTP Orchestration": (
            "authenticated\nMCP HTTP tests (`12 passed`",
            "combined CLI/MCP/capability suite `32 passed`",
            "`just lint` passes",
        ),
        "Model-Family Acceptance Harness": (
            "v1.9.0 focused family/report suite",
            "all-family strict acceptance passed for 9 families",
            "profiled unsupported skips",
            "`0` raw failed Step 11/12 counts",
        ),
        "Cross-Step Evidence Ledger": (
            "Step 3/5/6/11/12/15/16/23",
            "artifact links",
            "all-family strict acceptance passed for 9 families",
        ),
        "Interpretability Summaries": (
            "variable/edge inventories",
            "matrix-shape tables",
            "telemetry presence",
        ),
        "Release Readiness Ledger": (
            "release gates passed",
            "collect-only inventory",
            "full suite evidence",
        ),
        "Semantic Round-Trip Gates": (
            "semantic fidelity gate passed for 9 families",
            "gnn_semantic_fidelity_ledger_v1",
            "variables, edges, dimensions, parameter shapes, equations, time, and ontology mappings",
            "scripts/run_semantic_fidelity_gate.py",
        ),
        "Cross-Framework Result Comparisons": (
            "cross-framework reliability gate passed for 9 families",
            "gnn_cross_framework_reliability_ledger_v1",
            "GridWorld compared PyMDP, RxInfer, and ActiveInference.jl",
            "explicit unsupported statuses",
            "scripts/run_cross_framework_reliability.py",
        ),
        "Durable Observation Streams": (
            "src/pipeline/durable_streams.py",
            "scripts/run_v3_orchestration_acceptance.py",
            "model-family acceptance all passed for 9 families",
        ),
        "Long-Running Pipeline Sessions": (
            "src/pipeline/run_session.py",
            "src/pipeline/session_acceptance.py",
            "model-family acceptance all passed for 9 families",
        ),
        "Auditable Container Plans": (
            "src/pipeline/container_plan.py",
            "scripts/run_v3_orchestration_acceptance.py",
            "model-family acceptance all passed for 9 families",
        ),
    }
    for item in guarded_pending_items:
        if f"- [x] **{item}**" in todo_text:
            required_evidence = release_evidence_by_item.get(item)
            if not required_evidence or not all(
                evidence in todo_text for evidence in required_evidence
            ):
                failures.append(
                    f"TO-DO.md: {item} is marked complete before release-readiness evidence"
                )
    for path in ("input/multi_agent_models", "src/tests/audio", "src/tests/gui"):
        if path in todo_text and not _exists(path):
            failures.append(f"TO-DO.md: acceptance path does not exist: {path}")
    for path in (
        "input/model_family_manifest.json",
        "scripts/run_model_family_acceptance.py",
        "src/tests/pipeline/test_model_family_acceptance.py",
        "src/tests/analysis/test_interpretability_summary.py",
        "src/tests/report/test_model_family_report.py",
    ):
        if path in todo_text and not _exists(path):
            failures.append(f"TO-DO.md: acceptance path does not exist: {path}")

    roadmap_sections = _split_todo_sections(todo_text)
    early_versions = ("v1.8.0", "v1.9.0", "v2.0.0", "v3.0.0")
    autonomy_patterns = (
        "Self-Modifying",
        "self-editing",
        "self editing",
        "rewrite their own",
        "autonomous ecology",
    )
    for version in early_versions:
        section = roadmap_sections.get(version, "")
        for pattern in autonomy_patterns:
            if pattern.lower() in section.lower():
                failures.append(
                    f"TO-DO.md: autonomy/self-editing claim appears before v4.0.0 in {version}"
                )
    if (
        "v4.0.0" not in roadmap_sections
        or "--autonomous" not in roadmap_sections["v4.0.0"]
    ):
        failures.append("TO-DO.md: bounded autonomous mode must be scoped under v4.0.0")

    for required in (
        "input/model_family_manifest.json",
        "scripts/run_model_family_acceptance.py",
        "src/pipeline/model_family_acceptance.py",
        "src/analysis/interpretability.py",
        "src/report/model_family.py",
    ):
        if not _exists(required):
            failures.append(f"v1.9 model-family contract missing: {required}")

    for required in (
        "scripts/run_semantic_fidelity_gate.py",
        "scripts/run_cross_framework_reliability.py",
        "src/pipeline/semantic_fidelity.py",
        "src/pipeline/cross_framework_reliability.py",
        "src/report/semantic_fidelity.py",
        "src/report/cross_framework_reliability.py",
        "src/tests/pipeline/test_semantic_fidelity_gate.py",
        "src/tests/pipeline/test_cross_framework_reliability.py",
    ):
        if not _exists(required):
            failures.append(f"v2.0 reliability contract missing: {required}")

    if "pymdp,rxinfer,activeinference_jl" not in _read(
        "input/model_family_manifest.json"
    ):
        failures.append(
            "input/model_family_manifest.json: GridWorld must profile a real multi-backend comparison"
        )

    if "WebSocket" in todo_text:
        if not _contains(
            "src/gui/websocket_bridge.py",
            (
                "model.load",
                "matrix.patch",
                "validation.result",
                "model.export",
                "error",
            ),
        ):
            failures.append(
                "src/gui/websocket_bridge.py: missing required GUI message types"
            )

    docs_with_three = [
        path
        for path in ("TO-DO.md", "src/advanced_visualization/README.md")
        if re.search(r"Three\.js|three\.js", _read(path))
    ]
    if docs_with_three and not _contains(
        "src/visualization/matrix/visualizer.py",
        ("generate_threejs_tensor_explorer", "three@"),
    ):
        failures.append(
            "Three.js is documented but matrix Three.js renderer is missing"
        )

    if "GNN_MCP_TOKEN" in todo_text or "authenticated HTTP" in todo_text:
        if not _contains(
            "src/mcp/server_http.py", ("GNN_MCP_TOKEN", "Authorization", "Bearer")
        ):
            failures.append(
                "MCP HTTP auth is documented but bearer-token gate is missing"
            )
        server_http_text = _read("src/mcp/server_http.py")
        if "GNN_MCP_ALLOW_INSECURE_LOCAL" not in server_http_text:
            failures.append(
                "src/mcp/server_http.py: insecure local HTTP opt-in variable missing"
            )
        if "is_loopback_client" not in server_http_text:
            failures.append(
                "src/mcp/server_http.py: insecure local HTTP opt-in must be loopback-gated"
            )
        if "get_environment_info" in _safe_allowlist_literal(server_http_text):
            failures.append(
                "src/mcp/server_http.py: get_environment_info must not be safe by default"
            )
        if "get_system_info" in _safe_allowlist_literal(server_http_text):
            failures.append(
                "src/mcp/server_http.py: get_system_info must not be safe by default"
            )
        if "GNN_MCP_SAFE_RESOURCES" not in server_http_text:
            failures.append(
                "src/mcp/server_http.py: missing explicit HTTP resource allowlist"
            )
        if "is_safe_http_resource" not in server_http_text:
            failures.append(
                "src/mcp/server_http.py: mcp.resource.get must be default-denied"
            )
        if "get_http_capabilities" not in server_http_text:
            failures.append(
                "src/mcp/server_http.py: HTTP capabilities must be allowlist-filtered"
            )
        do_post_index = server_http_text.find("def do_POST")
        auth_index = server_http_text.find("is_authorized(", do_post_index)
        rate_index = server_http_text.find("is_rate_limited(", do_post_index)
        if auth_index != -1 and rate_index != -1 and auth_index < rate_index:
            failures.append(
                "src/mcp/server_http.py: rate limiting must run before bearer auth"
            )

    acceptance_text = _read("src/pipeline/model_family_acceptance.py")
    if "_load_pipeline_summary" not in acceptance_text:
        failures.append(
            "src/pipeline/model_family_acceptance.py: missing pipeline summary parsing"
        )
    if "_selected_steps_passed" not in acceptance_text:
        failures.append(
            "src/pipeline/model_family_acceptance.py: missing selected-step status gate"
        )
    if "acceptance_profile_defaults" not in _read("input/model_family_manifest.json"):
        failures.append(
            "input/model_family_manifest.json: missing manifest-level acceptance profiles"
        )
    if "pipeline_passed = False" not in acceptance_text:
        failures.append(
            "src/pipeline/model_family_acceptance.py: missing pipeline-summary fail-closed path"
        )
    if "allow_unsupported_reason_patterns" in acceptance_text:
        failures.append(
            "src/pipeline/model_family_acceptance.py: reason-pattern fallback must not certify unsupported steps"
        )
    if "allow_unsupported_reason_patterns" in _read("input/model_family_manifest.json"):
        failures.append(
            "input/model_family_manifest.json: unsupported steps must use explicit profiled skip reasons, not reason patterns"
        )
    if "profiled_unsupported_skip" not in acceptance_text:
        failures.append(
            "src/pipeline/model_family_acceptance.py: missing explicit profiled unsupported skip evidence"
        )
    if (
        "return_code in {0, 2}" in acceptance_text
        and "pipeline_summary" not in acceptance_text
    ):
        failures.append(
            "src/pipeline/model_family_acceptance.py: return code 2 is accepted without summary evidence"
        )
    for required in (
        "_reset_family_dir",
        "STEP_ARTIFACT_REQUIREMENTS",
        "missing_artifact_evidence",
        "_pipeline_run_outcome_acceptable",
        "partial_render_failure",
    ):
        if required not in acceptance_text:
            failures.append(
                "src/pipeline/model_family_acceptance.py: "
                f"missing model-family oracle hardening marker {required}"
            )

    rxinfer_toml_text = _read("src/render/rxinfer/toml_generator.py")
    if not all(
        required in rxinfer_toml_text
        for required in (
            "agent_ids",
            "agent_initial_positions",
            "agent_target_positions",
        )
    ):
        failures.append("RxInfer compact multi-agent keys are missing")
    for required in (
        "_validate_topology_references",
        "Malformed topology edge",
        "members must be a list",
    ):
        if required not in rxinfer_toml_text:
            failures.append(
                "src/render/rxinfer/toml_generator.py: "
                f"missing topology fail-closed marker {required}"
            )

    rxinfer_renderer_text = _read("src/render/rxinfer/rxinfer_renderer.py")
    execute_text = _read("src/execute/processor.py")
    for required in ("script_sha256", "metadata_provenance"):
        if required not in rxinfer_renderer_text or required not in execute_text:
            failures.append(
                "RxInfer execution metadata missing sidecar provenance/hash guard: "
                f"{required}"
            )

    audio_processor_text = _read("src/audio/processor.py")
    audio_streaming_text = _read("src/audio/streaming.py")
    for required in (
        "telemetry_provenance",
        "relative_to(execution_output_dir.resolve())",
        'write_stream_summary([], output_dir / "audio_stream_chunks.json")',
    ):
        if required not in audio_processor_text:
            failures.append(
                f"src/audio/processor.py: missing audio streaming guard {required}"
            )
    if '"streaming_safe": False' not in audio_streaming_text:
        failures.append(
            "src/audio/streaming.py: empty telemetry chunks must not be streaming-safe"
        )

    if "pip install" in _read("src/render/discopy/discopy_renderer.py"):
        failures.append("DisCoPy renderer still emits runtime dependency installation")

    if not _exists("src/pipeline/autonomous.py"):
        failures.append("Autonomous proposal loop implementation missing")
    autonomous_text = (
        _read("src/pipeline/autonomous.py")
        if _exists("src/pipeline/autonomous.py")
        else ""
    )
    for required in ("source_mutation_performed", "cluster_mutation_performed"):
        if required not in autonomous_text:
            failures.append(
                f"Autonomous proposal loop missing non-mutation marker: {required}"
            )
    if "--autonomous" not in _read("src/utils/argument_utils.py"):
        failures.append("Pipeline argument parser missing --autonomous")

    return failures


def _main_commands_missing_isolated_output_dir(todo_text: str) -> list[str]:
    """Return acceptance commands that run src/main.py without /tmp output."""
    failures: list[str] = []
    for line in todo_text.splitlines():
        command = line.strip()
        if not command or command.startswith("#"):
            continue
        if "src/main.py" not in command:
            continue
        if "--output-dir /tmp/" not in command:
            failures.append(command)
    return failures


def _safe_allowlist_literal(source: str) -> str:
    """Extract the default MCP safe-tool allowlist source text."""
    match = re.search(
        r"DEFAULT_SAFE_HTTP_TOOL_NAMES\s*=\s*frozenset\(\s*\{(.*?)\}\s*\)",
        source,
        re.DOTALL,
    )
    return match.group(1) if match else ""


def _split_todo_sections(todo_text: str) -> dict[str, str]:
    """Return roadmap sections keyed by semantic version heading."""
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"^## .*?(v\d+\.\d+\.\d+).*?$", todo_text, re.MULTILINE))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(todo_text)
        sections[match.group(1)] = todo_text[start:end]
    return sections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; failures are strict by default",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Report failures without a nonzero exit",
    )
    args = parser.parse_args(argv)

    failures = run_audit()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 0 if args.warn_only else 1
    print("Capability contracts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
