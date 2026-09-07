# REPORT — security worker (fleet 3, 2026-09-04)

Scope: `src/gnn/security/` (processor.py, mcp.py, `__init__.py`, AGENTS.md, README.md, SPEC.md, SKILL.md) + `src/gnn/18_security.py`. Branch main @ f64ac9085; no commits made (in-place edits only, per fleet rules).

## Files changed + why

| File | Change |
|---|---|
| `src/gnn/security/processor.py` | Refactor (see below) + new `scan_source` API |
| `src/gnn/security/__init__.py` | Version 1.6.0 → 1.7.0; export the full policy/gate surface; fix `FEATURES` (`mcp_integration` was `False` with a stale "No mcp.py exists" comment — mcp.py exists and registers 4 tools) |
| `src/gnn/security/AGENTS.md` | Layered API reference, new examples, version history, test list |
| `src/gnn/security/README.md` | New components, exports, scan_source example, test list |
| `src/gnn/security/SKILL.md` | API block + Key Exports updated |
| `src/gnn/security/SPEC.md` | Interface mapping, functional requirements, component table |
| `tests/security/test_security_policy_and_source.py` | **New**: 44 tests, 7 classes |
| `src/gnn/18_security.py` | Unchanged (already thin, 55 ln) |

## Refactor details (composability)

- **Pure policy layer**: policy semantics extracted from `process_security` into `resolve_security_policy(...)` + frozen `ResolvedSecurityPolicy` dataclass with `to_receipt()`. Single source of truth; never raises; invalid requests fail closed (`is_valid=False` + human-readable `error`). `process_security` now delegates — its 60-line inline resolver is gone.
- **Shared severity helpers**: `findings_at_or_above(findings, block_on)` (fail-closed: unknown/missing severity ranks as high) and `count_by_severity(findings)` replace 3 inline copies of the threshold filter (process_security, scan_script_for_execution, summary counting).
- **Finding factory**: `_make_finding()` gives all detectors one canonical dict shape; per-detector key omissions preserved (tests pin shapes).
- **Pattern tables hoisted**: `_SENSITIVE_PATTERNS`, `_VULN_PATTERNS`, `_CREDENTIAL_PATTERNS`, `_DANGEROUS_CALLS`, `_DANGEROUS_METHODS` moved from function bodies to module constants — the AST tables were being rebuilt on every call.
- **Imports cleaned**: `ast`/`stat`/`shutil`/`subprocess` hoisted from 5 function bodies to module level; `Tuple`/`Iterable` typed; stale `#!/usr/bin/env python3`-only header replaced with a layering docstring.
- **Typed error**: `SecurityScanError(Exception)` replaces bare `raise Exception(...)` in `perform_security_check`. Subclasses `Exception` deliberately — all existing `except Exception` consumers (MCP wrappers, orchestrator) keep working.
- **Behavioral fixes** (regression-proofed by tests):
  - World-writable check now inspects `stat.S_IWOTH` on the file mode directly; previously it gated on `os.access(W_OK)` *for the current process*, which could mask a world-writable file (read-only mount, other-user owner).
  - `perform_security_check` reports the real octal permission mode instead of the hardcoded `"readable"` string.
  - `check_vulnerabilities` / `generate_security_recommendations` read with explicit `encoding="utf-8", errors="replace"` instead of locale-dependent `open()`.
  - Gate deny-receipts (invalid threshold, unreadable) deduped via `_deny_verdict`; both keep the `block_on` key.

## New functionality (additive, 1.7.0)

- **`scan_source(source, *, file_name="<memory>.py", block_on=None) -> dict`** — scan Python source *text* without a file. The render step (Step 11) can validate generated code before writing it to disk; tests/MCP can scan snippets. With `block_on`, returns the same verdict fields as `scan_script_for_execution` (`ok`/`blocked`/`decision`/`block_on`) so one handling path serves both.
- **`resolve_security_policy` / `ResolvedSecurityPolicy`** — pre-validate a policy before scanning anything (pure, no fs).
- **`findings_at_or_above` / `count_by_severity`** — reusable severity algebra.
- **`scan_script_for_execution`** now exported from the package root (it was processor-only despite being the Step-12 gate entry point used by `execute/processor.py`).
- `SecurityScanError` exported; `get_module_info` listed in `__all__`.

## API deltas

- Added (exported): `resolve_security_policy`, `ResolvedSecurityPolicy`, `scan_source`, `findings_at_or_above`, `count_by_severity`, `SecurityScanError`, `scan_script_for_execution`, `get_module_info`.
- Changed (compatible): `perform_security_check` raises `SecurityScanError` instead of bare `Exception`; its `file_permissions` value is now the real octal mode (was constant `"readable"` — no test or consumer pinned the constant).
- Unchanged: `process_security` signature/behavior, all receipt/verdict key sets, exit codes, logging conventions, `mcp.py` (untouched — 4 tools, all tests pass).

## Verification (output tails)

```
uv run ruff check src/gnn/security tests/security
  → All checks passed!
uv run --extra dev mypy src/gnn/security --config-file pyproject.toml
  → Success: no issues found in 3 source files
uv run pytest tests/security/ -q
  → 102 passed in 1.00s   (58 pre-existing + 44 new)
just test-mod security: `just` NOT installed on host → ran its exact recipe
  (uv run pytest tests/security/ -v): 102 passed, exit 0.
Smoke: 18_security.py on a clean model → "✅ Security processing completed
successfully", exit 0; empty/invalid-policy → deny receipts as before.
Smoke: execute-gate call pattern (scan .py with subprocess.getoutput) →
  ok=False, decision="deny".
Note: tests/api/test_comprehensive_api.py had 10 transient NameError
failures ('website' not defined) on one run — a concurrent wave peer editing
website/report modules mid-run; clean rerun: 47 passed. Not security-related.
uv run ruff format --check src/gnn/security tests/security
  → 11 files already formatted (after `ruff format` reformatted
    src/gnn/security/processor.py + tests/security/test_security_policy_and_source.py;
    ruff check / mypy / 102 tests re-verified green post-format)
Version pins: docs/VERSION_MAP.md has no security entries; src/gnn/mcp's 1.6.0 is
  that subsystem's independent version — the security 1.7.0 bump pins nothing.
```

## Doc / manuscript follow-ups (other workers own these)

- `docs/gnn/mcp/tool_reference.md`: security tools unchanged; no action needed.
- `docs/` pipeline docs referencing Step 18 outputs: unchanged contract (security_results.json + security_summary.md), no action.
- If another worker wires `scan_source` into `render/` (recommended: validate rendered Python before writing), the doc cross-ref belongs in `src/gnn/render/AGENTS.md`, not mine.

## Follow-up ideas

1. Wire `scan_source` into `render/processor.py` so Step 11 gates generated Python *before* disk writes (Step 12's gate then never sees preventable findings).
2. Consider `decision` enum + `TypedDict`s for verdict/receipt dicts once the repo adopts runtime typing for module contracts.
3. `mcp.py` `scan_gnn_file_mcp` duplicates a coarser substring pattern set; it could delegate to `scan_source`/`_VULN_PATTERNS` for consistency (left untouched — outside the delta risk budget for this pass).
4. `_JULIA_SUSPICIOUS_PATTERNS` could gain `ccall` / `unsafe_load` patterns for FFI-level review.

## Addendum (format + version-pin verification, post-report)

```
uv run ruff format --check src/gnn/security tests/security
  (initial) → 2 files would be reformatted: src/gnn/security/processor.py,
              tests/security/test_security_policy_and_source.py
  (fixed via uv run ruff format) → 11 files already formatted
  re-verified after formatting: ruff check "All checks passed!",
  mypy "Success: no issues found in 3 source files",
  pytest tests/security/ → 102 passed.
```

Version pins: `docs/VERSION_MAP.md` has no security-module entries; all 1.6.0
hits in `src/gnn/mcp/` are the MCP subsystem's independent version; the security
1.7.0 bump pins nothing.

## Addendum 2 (final hardening pass)

- **AGENTS.md claim-strength fix**: deleted the duplicated "## Security Features" section (aspirational RBAC/encryption/key-management/exfiltration claims with no code behind them); replaced the false "Path Traversal Checks" bullet with the real permission checks (world-writable `stat.S_IWOTH` detection + real octal mode reporting).
- **Tests hardened**: deleted the stdlib-tautology `TestWorldWritablePosixOnly` test; added a deterministic `test_stat_failure_reports_unknown_permissions` (monkeypatched `Path.stat` → OSError → `file_permissions == "unknown"` branch covered identically on every platform); `test_perform_security_check_unreadable_file_raises_typed_error` now pins ONLY `SecurityScanError` (previously a `PermissionError` tolerance would pass even if the typed wrapping were silently removed).
- **`scan_source` docstring**: documents the deliberate deny-receipt asymmetry vs the file gate (bare `deny_invalid_policy` without the `policy_validation` finding; `block_on` echo present in both branches — verified in code).
- Re-verified after all edits: `ruff format --check` → 3 files already formatted; `ruff check` → All checks passed!; mypy → Success (3 files); pytest tests/security → **102 passed** (44 in the new file: -1 tautology, +1 monkeypatch test).
