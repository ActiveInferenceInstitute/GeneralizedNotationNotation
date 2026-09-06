# Specification: Security

## Design Requirements

The `src/gnn/security/` module provides comprehensive security validation and generated-code scanning for the GNN pipeline (Step 18).

## Interface Mapping

- `18_security.py`: Thin orchestrator binding `security.processor.process_security()`
- `processor.py`: Core security engine, layered: pure policy resolution → pure scanning → threshold verdicts → orchestration
- `mcp.py`: MCP tool registration for security validation operations

## Functional Requirements

- **Generated Code Scanning**: Analyze rendered scripts (Step 11 output) for unsafe patterns, injection risks, and dynamic execution constructs — both on disk (`scan_script_for_execution`) and in memory before writing (`scan_source`, 1.7.0)
- **AST Analysis**: Python AST scanner for `shell=True`, dangerous calls, and dynamic execution, with import-alias and from-import call-alias tracking
- **Policy Resolution**: `resolve_security_policy()` is the single source of truth for level/threshold/scan-override semantics; returns a frozen `ResolvedSecurityPolicy` and never raises (invalid requests fail closed with `is_valid=False` + `error`)
- **Pre-Execution Gate**: `scan_script_for_execution()` blocks rendered scripts with findings at/above `block_on` before Step 12 runs them (`.jl` scripts: advisory sweep + `Meta.parseall` probe); unknown severities fail closed
- **Report Generation**: Produce structured security findings with severity levels and remediation guidance

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `process_security()` | Function | Top-level entry point called by orchestrator |
| `resolve_security_policy()` | Function | Pure policy resolution and validation (1.7.0) |
| `ResolvedSecurityPolicy` | Dataclass | Frozen resolved policy with `to_receipt()` (1.7.0) |
| `perform_security_check()` | Function | Per-file sensitive-data and integrity check (raises `SecurityScanError`) |
| `check_vulnerabilities()` | Function | Per-file pattern + AST + permission vulnerability scan |
| `scan_source()` | Function | In-memory Python source scanning with optional verdict fields (1.7.0) |
| `scan_script_for_execution()` | Function | Pre-execution gate for rendered scripts |
| `findings_at_or_above()` / `count_by_severity()` | Functions | Shared severity-threshold filter (fail-closed) and histogram (1.7.0) |
| `SecurityScanError` | Exception | Typed per-file scan failure (1.7.0) |
| `mcp.py` | MCP Tools | Security validation and audit tools |

## Standards

- Findings classified by severity: info, low, medium, high (`_SEVERITY_RANK`)
- Blocking is level-dependent: `basic` never blocks, `standard` reports only, `strict` blocks on high; an explicit `block_on` makes any scanning level enforcement-capable
- Reports generated in both JSON and Markdown formats
