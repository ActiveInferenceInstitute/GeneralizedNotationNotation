# Security Module

This module (Pipeline Step 18) performs security scanning of GNN pipeline files: injection-pattern detection, Python AST analysis, severity-based scoring, recommendations, policy resolution, and a pre-execution gate for rendered scripts before Step 12 runs them.

## Module Structure

```
src/gnn/security/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # Scanning, scoring, recommendations, pre-exec gate
├── mcp.py                         # MCP tool registrations
├── AGENTS.md                      # Agent scaffolding and workflows
├── SPEC.md                        # Architectural specification
├── SKILL.md                       # Capability API
└── README.md                      # This documentation
```

## Core Components

### `process_security(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`

Main entry point, called by `18_security.py` (Step 18).

- Scans all GNN files in `target_dir` (pattern matching + Python AST analysis)
- Selects analysis depth via `security_level` (`basic` = no scanning, `standard` = scan and report, `strict` = scan + block on high severity)
- An explicit `block_on` kwarg makes any scanning level enforcement-capable; strict mode cannot disable vulnerability scanning
- Writes `security_results.json` and `security_summary.md`

**Returns:** `bool` — True if processing succeeded.

### `perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]`

Sensitive-data and integrity check on a single file: credential-pattern scanning (`password`, `secret`, `api_key`, `token`, `private_key` — matches are context-redacted), SHA-256 hash of the exact bytes inspected, the real POSIX permission mode (octal), and a 0-100 security score. Raises `SecurityScanError` (a subclass of `Exception`) when the file cannot be read.

### `check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`

Vulnerability scan using regex pattern tables and Python AST analysis (`shell=True`, dangerous calls, dynamic execution constructs), plus world-writable mode checks (`.py` only). Findings sort deterministically; unreadable files produce a low-severity finding instead of an exception.

### `calculate_security_score(vulnerabilities) -> float`

Severity-weighted security score (0-100; 100 = no findings).

### `resolve_security_policy(...) -> ResolvedSecurityPolicy`

**New in 1.7.0.** Pure, total policy resolution and validation (the single
source of truth behind `process_security`): normalizes `security_level` /
`block_on` / `check_vulnerabilities`, enforces the strict-and-enforced
scan-must-stay-on rule, and returns a frozen dataclass with
`to_receipt()` for the `security_results.json` policy block. Never raises;
invalid requests set `is_valid=False` with a human-readable `error`.

### `findings_at_or_above(findings, block_on)` / `count_by_severity(findings)`

**New in 1.7.0.** Shared severity-threshold filter (fail-closed on unknown
severities) and severity histogram used by both the Step 18 receipt and the
pre-execution gate.

### `scan_source(source, *, file_name="<memory>.py", block_on=None)`

**New in 1.7.0.** Scan Python source *text* without a file — validate
rendered/generated code before writing it to disk. Returns findings and, when
`block_on` is given, the same verdict fields as `scan_script_for_execution`.

### `scan_script_for_execution(script_path: Path, *, block_on: str = "high") -> Dict[str, Any]`

Pre-execution security gate (RED_TEAM V-01/V-06): applies the Python AST scanner to a rendered `.py` script *before* Step 12 executes it, returning `{ok, blocked, findings, scanned, block_on, decision}`. Findings at/above `block_on` severity set `ok=False` (unknown severities fail closed). `.jl` scripts get an advisory regex sweep plus a `julia -e Meta.parseall` syntax probe (parse failure = high severity; 30 s timeout). Wired into `execute.processor.execute_single_script`; escape hatch: `GNN_ALLOW_UNSAFE_EXEC=1`. Exported from the package root since 1.7.0.

### Exports (`from security import ...`)

- `process_security`, `perform_security_check`, `check_vulnerabilities`
- `generate_security_recommendations`, `calculate_security_score`, `generate_security_summary`
- `resolve_security_policy`, `ResolvedSecurityPolicy`, `findings_at_or_above`, `count_by_severity` (new in 1.7.0)
- `scan_source`, `scan_script_for_execution`, `SecurityScanError`
- `FEATURES`, `__version__`, `get_module_info`

## Usage Examples

### Basic security processing

```python
from gnn.security import process_security
from pathlib import Path

success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    verbose=True,
    security_level="standard",
)
```

### Validate generated code before writing it (1.7.0)

```python
from gnn.security import scan_source

verdict = scan_source(rendered_python, file_name="rendered_model.py", block_on="high")
if not verdict["ok"]:
    raise RuntimeError(f"unsafe render: {verdict['blocked']}")
```

### File-level scan

```python
from gnn.security import check_vulnerabilities, perform_security_check

vulns = check_vulnerabilities(Path("output/11_render_output/model_pymdp.py"))
result = perform_security_check(Path("models/sensitive_model.md"))
print(result["security_score"])
```

## Integration with Pipeline

### Pipeline Step 18: Security Processing

`18_security.py` is a thin orchestrator: it parses the standardized `--target-dir`, `--output-dir`, `--recursive`, `--verbose` arguments and delegates to `process_security()`.

### Output Structure

```
output/18_security_output/
├── security_results.json   # Scan findings, scores, per-file results
└── security_summary.md     # Human-readable summary
```

## Security Features

- **Injection Pattern Scanning**: Regex patterns for OS command injection, suspicious imports, and script-injection constructs
- **Python AST Analysis**: Detects `shell=True`, dangerous calls, and dynamic execution
- **Julia Script Analysis**: Advisory regex sweep plus `Meta.parseall` syntax probe
- **Pre-Execution Gate**: Rendered scripts blocked before execution on high-severity findings; `GNN_ALLOW_UNSAFE_EXEC=1` escape hatch
- **Sensitive-Data Detection**: Credential-pattern scanning with redacted context

## Dependencies

- **Required (stdlib)**: pathlib, json, logging, hashlib, re, datetime
- **Optional**: Julia interpreter (for the `.jl` `Meta.parseall` probe; advisory regex sweep without it)

## Testing

Tests live in `tests/security/`: `test_security_overall.py`, `test_security_functional.py`, `test_pre_exec_gate.py`, `test_security_mcp_tools.py`, `test_security_policy_and_source.py` (policy resolver, `scan_source`, severity helpers, permission semantics — added 1.7.0), `test_sandbox.py`, `test_pygments_archetype_redos.py`.

```bash
uv run --extra dev python -m pytest tests/security/ --cov=src/gnn/security
```

## References

- Project overview: ../../../README.md
- Pipeline details: ../../../docs/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
