# validation-worker REPORT — fleet 3, 2026-09-04

Scope: `src/validation/` (all files) + `src/6_validation.py`. Branch main @ f64ac9085 baseline; edits in place, no git operations, no dependency changes.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/validation/structure.py` | **NEW** (175 ln) | Shared helpers both validators duplicate/need: `extract_content_from_dict` (was verbatim-duplicated in two files), `clamp01`, `display_file_name`, `DirectedEdge` + `cycle_nodes` (exact Tarjan SCC cycle detection) |
| `src/validation/workflow.py` | **NEW** (392 ln) | Extracted the ~300-line `process_validation` monolith from `__init__.py`: `_locate_gnn_results`, `_load_or_init_results`, `_run_stage`, `_validate_file`, `_record_average_scores`, `_write_receipts`, `_log_summary`, public `validate_directory` + `StageServices` (DI) |
| `src/validation/__init__.py` | 374 → 96 ln | Thin package facade: `process_validation` now binds stage callables from package globals at call time (preserves the `monkeypatch.setattr(validation, ...)` test seam) and forwards `validation_level`/`strict`; `__version__` 1.6.0 → 1.7.0; `__all__` += `validate_directory`, `StageServices`, `validate_content` |
| `src/validation/semantic_validator.py` | dedup + fixes | Uses shared `structure` helpers; replaced over-approximating DFS cycle detector with exact Tarjan membership (warning text format unchanged); PEP 585/604 typing; removed `cast` |
| `src/validation/performance_profiler.py` | dedup + contract fix | Uses shared `structure` helpers (kills 69-line duplicate); **error receipt now uniform** (`file_name`, `metrics`, `warnings`, `performance_score: 0.0`, `recovery: True` — previously violated the module's own documented best-effort contract); success dict adds `recovery: False`; modern typing |
| `src/validation/consistency_checker.py` | dedup | Tarjan moved to shared `structure.cycle_nodes` (local adapter keeps filtering semantics); score/file-name helpers shared; zero behavior change (pinned by existing contract tests) |
| `src/validation/mcp.py` | hygiene + additive | Import order fixed (`from . import` above logger); inline `import json` hoisted; `validate_gnn_file_mcp` success result now includes a `semantic` key (full rule-based semantic result) — additive, existing keys untouched; modern typing |
| `src/6_validation.py` | docstring only (51 ln, < 150 ✓) | Documents that `--strict` now actually raises the semantic validation level |
| `src/validation/AGENTS.md`, `README.md`, `SPEC.md`, `SKILL.md` | docs of record | New files, new API entries, kwargs contract, uniform error contract, updated test-file list, version/last-updated |
| `src/tests/validation/test_workflow_contracts.py` | **NEW** (18 tests) | Contract pins listed below |

## API deltas (all additive or internal)

- **New public**: `validation.validate_content(content, validation_level="standard")` — no-I/O semantic validation of in-memory text (companion to `process_semantic_validation`, same receipt shape); `validation.validate_directory(...)` + `validation.StageServices` — the extracted workflow with injectable stage callables.
- **Behavior-fixed**: `profile_performance` failure receipt now matches the documented best-effort contract (adds `recovery: True`, `performance_score: 0.0`, `file_name`, `metrics`, `warnings`); success adds `recovery: False`. Consumers use `.get(...)` (verified `__init__`/workflow + tests), so new keys are safe.
- **Behavior-fixed**: `SemanticValidator` level-3 cycle warning now lists exactly the cycle members (Tarjan SCC) instead of every node that can reach a cycle — false-positive reduction; message format unchanged. `ConsistencyChecker` cycle semantics byte-identical (existing regression tests pass).
- **Newly honored kwargs**: `process_validation(..., validation_level=...)` and `strict=True` (was silently ignored; `--strict` CLI flag previously did nothing). Default path (`standard` level) byte-equivalent.
- **Additive MCP**: `validate_gnn_file` result gains `semantic` key. Tool names unchanged (pinned by `src/tests/mcp/test_mcp_audit.py`).
- **Unchanged**: all `__all__` symbols' signatures, exit-code semantics (bool → 0/1), output JSON shapes, log message texts, accumulation behavior, `6_validation.py` thin-orchestrator wiring.

## Verification (all green)

```
uv run ruff check src/validation src/tests/validation src/6_validation.py
→ All checks passed!
uv run ruff format src/validation src/tests/validation src/6_validation.py
→ 4 files reformatted, 9 left unchanged (then clean)
uv run --extra dev mypy src/validation --config-file pyproject.toml
→ Success: no issues found in 7 source files
uv run pytest src/tests/validation/ -q
→ 59 passed in 0.10s   (41 pre-existing + 18 new)
```

- `just` binary is not installed on this host; the mission's `just test-mod validation` was executed as its exact underlying recipe (`justfile: uv run pytest src/tests/{{MODULE}}/ -v`) — same result, 59 passed. No deps installed.
- New test coverage pins: kwargs forwarding (`validation_level`/`strict`/template kwargs), stage-exception → persisted recovery receipts (semantic + performance), consistency recovery-mode recording, accumulation across passes (total_files, source_directories, avg scores), missing-manifest → False without receipt, `StageServices` DI composition order, profiler success/error contract, exact cycle membership (cycle members only; acyclic clean), structure helpers (clamp bounds, self-loops, shared-extractor identity between both validators), MCP `semantic` key + missing-file fast-fail.

## Follow-ups for other workers (docs/manuscript owners)

- `doc/gnn/modules/06_validation.md` still says kwargs are "accepted but not consumed" and doesn't mention `validate_content`/`validate_directory`/`StageServices` or the uniform profiler error contract — needs the same deltas (not my scope).
- `doc/PIPELINE_SCRIPTS.md` / `doc/gnn/mcp/tool_reference.md`: `validate_gnn_file` now returns `semantic`; worth one line.
- No `manuscript/` impact found (no references to validation internals).

## Follow-up ideas (out of scope today)

1. Semantic validator still validates dict inputs via regex on reconstructed text; feeding it `structure.py`-style structured parsing (like `consistency_checker._parse_structured_model`) would make level-1/2 checks input-format-agnostic. Requires an output-contract review across steps 7/8.
2. `_load_or_init_results` accumulation trusts whatever JSON is on disk; a schema-version stamp on the receipt would harden cross-version resumes.
3. Recursive Tarjan in `structure.cycle_nodes` hits Python's recursion limit on >~900-node chains; iterative SCC would future-proof the scaling studies.
4. `mcp.py` `validate_gnn_file_mcp`'s structural checks and `check_schema_compliance_mcp` overlap; could unify on one section-scanning helper.
