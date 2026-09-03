# Security Module

This module (Pipeline Step 18) performs security scanning of GNN pipeline files: injection-pattern detection, Python AST analysis, severity-based scoring, recommendations, and a pre-execution gate for rendered scripts before Step 12 runs them.

## Module Structure

```
src/security/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # Scanning, scoring, recommendations, pre-exec gate
├── mcp.py                         # MCP tool registrations
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

Sensitive-data and integrity check on a single file: credential-pattern scanning (`password`, `secret`, `api_key`, `token`, `private_key` — matches are context-redacted), SHA-256 hash of the exact bytes inspected, and a 0.0-1.0 security score.

### `check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`

Vulnerability scan using regex patterns and Python AST analysis (`shell=True`, dangerous calls, dynamic execution constructs).

### `calculate_security_score(vulnerabilities) -> float`

Severity-weighted security score (0.0-1.0).

### `scan_script_for_execution(script_path: Path, *, block_on: str = "high") -> Dict[str, Any]`

Pre-execution security gate (RED_TEAM V-01/V-06): applies the Python AST scanner to a rendered `.py` script *before* Step 12 executes it, returning `{ok, blocked, findings, scanned}`. Findings at/above `block_on` severity set `ok=False`. `.jl` scripts get an advisory regex sweep plus a `julia -e Meta.parseall` syntax probe (parse failure = high severity; 30 s timeout). Wired into `execute.processor.execute_single_script`; escape hatch: `GNN_ALLOW_UNSAFE_EXEC=1`.

### Exports (`from security import ...`)

- `process_security`, `perform_security_check`, `check_vulnerabilities`
- `generate_security_recommendations`, `calculate_security_score`, `generate_security_summary`
- `FEATURES`, `__version__`

## Usage Examples

### Basic security processing

```python
from security import process_security
from pathlib import Path

success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    verbose=True,
    security_level="standard",
)
```

### File-level scan

```python
from security import check_vulnerabilities, perform_security_check

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

Tests live in `src/tests/security/`: `test_security_overall.py`, `test_security_functional.py`, `test_pre_exec_gate.py`, `test_security_mcp_tools.py`, `test_sandbox.py`, `test_pygments_archetype_redos.py`.

```bash
uv run --extra dev python -m pytest src/tests/security/ --cov=src/security
```

## References

- Project overview: ../../README.md
- Pipeline details: ../../doc/pipeline/README.md

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
