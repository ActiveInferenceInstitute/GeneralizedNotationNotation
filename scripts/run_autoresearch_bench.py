#!/usr/bin/env python3
"""Deterministic benchmark for the GNN MCP + execute surfaces (deep-horizon wave 2).

One fixed, seeded, network-free workload exercising the four load-bearing
surfaces this session hardens:

1. parse      ``gnn.processing.processor.parse_gnn_file`` over four canonical
              discrete-state exemplars from ``input/gnn_files`` (result dicts
              digest-pinned at warm-up, volatile ``*_timestamp`` keys stripped).
2. serialize  canonical-JSON dumps + SHA-256 digests of the parse dicts,
              re-digested and compared against the warm-up digest every rep.
3. dispatch   ``gnn.mcp.processor.handle_mcp_request`` JSON-RPC traffic
              (``tools/list`` plus ``tools/call validate_export_format`` in a
              positive and a negative variant) and
              ``gnn.utils.mcp_dispatch.run_pipeline_step_mcp`` driving the real
              ``gnn.validation.process_validation`` step.
4. envelope   ``gnn.execute.subprocess_envelope.run_subprocess_envelope``
              spawning a trivial interpreter child with an enforced timeout,
              plus one deliberate timeout probe pinning the
              ``error_type == "TimeoutExpired"`` contract.

Every timed phase carries deterministic assertions (digest equality, JSON-RPC
result stability, envelope contract keys); any mismatch raises, so a broken
surface fails the benchmark instead of producing a number.

Printed metrics (one per line, stdout):

    METRIC mcp_execute_bench_ms=<best-of-3 total workload wall time in ms>
    METRIC parse_files_per_s=...
    METRIC serialize_roundtrips_per_s=...
    METRIC mcp_dispatch_per_s=...
    METRIC envelope_spawns_per_s=...
    METRIC mcp_setup_ms=<untimed warm-up/registration cost, informational>
    METRIC determinism_checks=<number of deterministic assertions passed>
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from shutil import copy2
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_RELPATHS: tuple[str, ...] = (
    "input/gnn_files/discrete/simple_mdp.md",
    "input/gnn_files/discrete/markov_chain.md",
    "input/gnn_files/discrete/hmm_baseline.md",
    "input/gnn_files/basics/static_perception.md",
)

PARSE_INNER = 8
SERIALIZE_INNER = 8
TOOLS_LIST_INNER = 20
TOOL_CALL_INNER = 20
WRAPPER_CALLS = 4
ENVELOPE_SPAWNS = 15
ENVELOPE_TIMEOUT_S = 30
WARMUP_REPS = 1
TIMED_REPS = 3

# 22 run_tool_envelope dependents + 18 run_pipeline_step_mcp dependents
# (pinned by tests/utils/test_mcp_dispatch.py) plus core introspection tools.
MIN_REGISTERED_TOOLS = 40
EXPECTED_TOOL = "validate_export_format"
NEGATIVE_FORMAT = "__no_such_gnn_export_format__"

# Result-dict keys whose values change run to run (parse/validation
# provenance, duration telemetry). Stripped before digesting so cross-rep
# and cross-run comparisons compare content, not clock.
_VOLATILE_RE = re.compile(
    r"(timestamp|duration|elapsed|_seconds$|_ms$|_time$)", re.IGNORECASE
)

checks = 0


def _check(condition: bool, what: str) -> None:
    global checks
    if not condition:
        raise AssertionError(f"determinism check failed: {what}")
    checks += 1


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and _VOLATILE_RE.search(k))
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _digest(obj: Any) -> str:
    payload = json.dumps(_strip_volatile(obj), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonrpc(method: str, params: dict[str, Any], req_id: int) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params,
    }


def phase_parse(
    parse_gnn_file: Callable[..., dict[str, Any]],
    files: list[Path],
    expected: dict[str, str],
) -> float:
    """Parse every exemplar PARSE_INNER times; digest must stay pinned."""
    t0 = time.perf_counter()
    for _ in range(PARSE_INNER):
        for path in files:
            result = parse_gnn_file(path)
            _check(
                result.get("success") is True,
                f"parse success for {path.name}",
            )
            _check(
                _digest(result) == expected[path.name],
                f"parse digest drift for {path.name}",
            )
    return time.perf_counter() - t0


def phase_serialize(
    parse_gnn_file: Callable[..., dict[str, Any]],
    files: list[Path],
    contents: list[str],
    expected: dict[str, str],
) -> float:
    """Reparse each file, then round-trip canonical JSON SERIALIZE_INNER times."""
    t0 = time.perf_counter()
    for path, content in zip(files, contents):
        result = parse_gnn_file(path, content=content)
        reference = _digest(result)
        _check(
            reference == expected[path.name],
            f"serialize reparse drift for {path.name}",
        )
        for _ in range(SERIALIZE_INNER):
            payload = json.dumps(_strip_volatile(result), sort_keys=True, default=str)
            _check(
                hashlib.sha256(payload.encode("utf-8")).hexdigest() == reference,
                f"canonical-JSON round-trip drift for {path.name}",
            )
    return time.perf_counter() - t0


def phase_dispatch(
    handle_mcp_request: Callable[[dict[str, Any]], dict[str, Any]],
    run_pipeline_step_mcp: Callable[..., dict[str, Any]],
    validation_step: Callable[..., Any],
    output_dir: Path,
    bench_logger: logging.Logger,
    expected: dict[str, Any],
) -> float:
    """Drive the JSON-RPC dispatcher and the canonical pipeline-step envelope."""
    t0 = time.perf_counter()

    for _ in range(TOOLS_LIST_INNER):
        resp = handle_mcp_request(_jsonrpc("tools/list", {}, 1))
        tools = resp.get("result", {}).get("tools")
        _check(isinstance(tools, list), "tools/list returns a tool list")
        _check(
            len(tools) == expected["tool_count"],
            f"tools/list count drifted: {len(tools)} != {expected['tool_count']}",
        )

    for _ in range(TOOL_CALL_INNER):
        positive = handle_mcp_request(
            _jsonrpc(
                "tools/call",
                {"name": EXPECTED_TOOL, "arguments": {"format_name": "json"}},
                2,
            )
        )
        _check(
            _digest(positive.get("result")) == expected["call_digest"],
            "tools/call positive result drifted",
        )
        negative = handle_mcp_request(
            _jsonrpc(
                "tools/call",
                {"name": EXPECTED_TOOL, "arguments": {"format_name": NEGATIVE_FORMAT}},
                3,
            )
        )
        _check(
            _digest(negative.get("result")) == expected["neg_digest"],
            "tools/call negative result drifted",
        )

    for _ in range(WRAPPER_CALLS):
        result = run_pipeline_step_mcp(
            validation_step,
            wrapper_name="process_validation_mcp",
            logger=bench_logger,
            target_directory=str(output_dir.parent / "target"),
            output_directory=str(output_dir),
            verbose=False,
            label="Validation",
        )
        _check(
            result.get("success") is True,
            f"pipeline-step envelope failed: {result}",
        )
        _check(
            _digest(result) == expected["wrapper_digest"],
            "pipeline-step envelope result drifted",
        )

    return time.perf_counter() - t0


def phase_envelope(run_subprocess_envelope: Callable[..., dict[str, Any]]) -> float:
    """Spawn trivial children through the shared envelope; pin the timeout path."""
    command = [sys.executable, "-c", "print('envelope-ok')"]
    slow = [sys.executable, "-c", "import time; time.sleep(1.2)"]
    t0 = time.perf_counter()
    for _ in range(ENVELOPE_SPAWNS):
        envelope = run_subprocess_envelope(command, timeout=ENVELOPE_TIMEOUT_S)
        _check(
            envelope["success"] is True,
            f"envelope success flag wrong: {envelope.get('error')}",
        )
        _check(envelope["return_code"] == 0, "envelope return code not 0")
        _check("envelope-ok" in envelope["stdout"], "envelope stdout not captured")
    timeout_envelope = run_subprocess_envelope(slow, timeout=1)
    _check(
        timeout_envelope["success"] is False,
        "envelope timeout probe must fail",
    )
    _check(
        timeout_envelope.get("error_type") == "TimeoutExpired",
        f"envelope timeout error_type drifted: {timeout_envelope.get('error_type')}",
    )
    return time.perf_counter() - t0


def run_rep(context: dict[str, Any]) -> dict[str, float]:
    """One full workload pass; returns per-phase wall times."""
    t_parse = phase_parse(
        context["parse_gnn_file"], context["files"], context["expected"]
    )
    t_serialize = phase_serialize(
        context["parse_gnn_file"],
        context["files"],
        context["contents"],
        context["expected"],
    )
    t_dispatch = phase_dispatch(
        context["handle_mcp_request"],
        context["run_pipeline_step_mcp"],
        context["validation_step"],
        context["output_dir"],
        context["bench_logger"],
        context["expected"],
    )
    t_envelope = phase_envelope(context["run_subprocess_envelope"])
    return {
        "parse": t_parse,
        "serialize": t_serialize,
        "dispatch": t_dispatch,
        "envelope": t_envelope,
        "total": t_parse + t_serialize + t_dispatch + t_envelope,
    }


def main() -> int:
    bench_logger = logging.getLogger("autoresearch_bench")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING)

    setup_start = time.perf_counter()
    from gnn.execute.subprocess_envelope import run_subprocess_envelope
    from gnn.mcp.processor import handle_mcp_request, register_module_tools
    from gnn.processing.multi_format_processor import process_gnn_multi_format
    from gnn.processing.processor import parse_gnn_file
    from gnn.utils.mcp_dispatch import run_pipeline_step_mcp
    from gnn.validation import process_validation

    # Full module-tool discovery (the same code path initialize() uses) so the
    # dispatcher sees the real live surface, not a curated subset. Discovery
    # itself initializes the process-wide singleton.
    register_module_tools(None)

    tools_resp = handle_mcp_request(_jsonrpc("tools/list", {}, 0))
    tools = tools_resp.get("result", {}).get("tools", [])
    names = [tool.get("name") for tool in tools]
    _check(
        len(tools) >= MIN_REGISTERED_TOOLS,
        f"tool count {len(tools)} < {MIN_REGISTERED_TOOLS}",
    )
    _check(EXPECTED_TOOL in names, f"{EXPECTED_TOOL} not registered")

    expected: dict[str, Any] = {"tool_count": len(tools)}

    with tempfile.TemporaryDirectory(prefix="gnn-w2-bench-") as tmp:
        tmp_path = Path(tmp)
        target_dir = tmp_path / "target"
        output_dir = tmp_path / "output"
        target_dir.mkdir()
        output_dir.mkdir()

        files: list[Path] = []
        contents: list[str] = []
        for rel in MODEL_RELPATHS:
            source = REPO_ROOT / rel
            destination = target_dir / source.name
            copy2(source, destination)
            files.append(destination)
            contents.append(source.read_text(encoding="utf-8"))

        for path, content in zip(files, contents):
            result = parse_gnn_file(path, content=content)
            _check(result.get("success") is True, f"warm parse failed for {path.name}")
            expected[path.name] = _digest(result)

        positive = handle_mcp_request(
            _jsonrpc(
                "tools/call",
                {"name": EXPECTED_TOOL, "arguments": {"format_name": "json"}},
                2,
            )
        )
        _check(
            positive.get("result", {}).get("success") is True,
            f"positive tools/call failed: {positive}",
        )
        expected["call_digest"] = _digest(positive.get("result"))

        negative = handle_mcp_request(
            _jsonrpc(
                "tools/call",
                {"name": EXPECTED_TOOL, "arguments": {"format_name": NEGATIVE_FORMAT}},
                3,
            )
        )
        _check(
            negative.get("result", {}).get("success") is True,
            f"negative tools/call errored: {negative}",
        )
        _check(
            negative.get("result", {}).get("is_valid") is False,
            "negative tools/call should report is_valid False",
        )
        expected["neg_digest"] = _digest(negative.get("result"))

        # Step-3 artifacts: process_validation (step 6) consumes the
        # parsed-model manifest that step 3 emits, so the bench produces it
        # once with the real step-3 processor
        # (``process_gnn_multi_format`` — the callable ``3_gnn.py`` wraps),
        # which resolves and writes ``3_gnn_output/gnn_processing_results.json``
        # under the base output directory itself.
        _check(
            process_gnn_multi_format(target_dir, output_dir, bench_logger) is True,
            "bench step-3 (process_gnn_multi_format) failed",
        )
        _check(
            (output_dir / "3_gnn_output" / "gnn_processing_results.json").exists(),
            "step-3 manifest missing",
        )

        wrapper = run_pipeline_step_mcp(
            process_validation,
            wrapper_name="process_validation_mcp",
            logger=bench_logger,
            target_directory=str(target_dir),
            output_directory=str(output_dir),
            verbose=False,
            label="Validation",
        )
        _check(
            wrapper.get("success") is True,
            f"warm pipeline-step envelope failed: {wrapper}",
        )
        expected["wrapper_digest"] = _digest(wrapper)

        setup_seconds = time.perf_counter() - setup_start

        context = {
            "parse_gnn_file": parse_gnn_file,
            "handle_mcp_request": handle_mcp_request,
            "run_pipeline_step_mcp": run_pipeline_step_mcp,
            "run_subprocess_envelope": run_subprocess_envelope,
            "validation_step": process_validation,
            "files": files,
            "contents": contents,
            "output_dir": output_dir,
            "bench_logger": bench_logger,
            "expected": expected,
        }

        for _ in range(WARMUP_REPS):
            run_rep(context)

        best: dict[str, float] | None = None
        for _ in range(TIMED_REPS):
            rep = run_rep(context)
            if best is None or rep["total"] < best["total"]:
                best = rep

    assert best is not None
    dispatch_calls = TOOLS_LIST_INNER + TOOL_CALL_INNER * 2 + WRAPPER_CALLS

    print(f"METRIC mcp_execute_bench_ms={best['total'] * 1000.0:.1f}")
    print(f"METRIC parse_files_per_s={PARSE_INNER * len(files) / best['parse']:.1f}")
    print(
        "METRIC serialize_roundtrips_per_s="
        f"{len(files) * (1 + SERIALIZE_INNER) / best['serialize']:.1f}"
    )
    print(f"METRIC mcp_dispatch_per_s={dispatch_calls / best['dispatch']:.1f}")
    print(
        f"METRIC envelope_spawns_per_s={(ENVELOPE_SPAWNS + 1) / best['envelope']:.1f}"
    )
    print(f"METRIC mcp_setup_ms={setup_seconds * 1000.0:.1f}")
    print(f"METRIC determinism_checks={checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
