# Specification: Security

## Design Requirements

The `src/security/` module provides comprehensive security validation and generated-code scanning for the GNN pipeline (Step 18).

## Interface Mapping

- `18_security.py`: Thin orchestrator binding `security.processor.process_security()`
- `processor.py`: Core security scanning engine — pattern matching, vulnerability detection, dependency auditing
- `mcp.py`: MCP tool registration for security validation operations

## Functional Requirements

- **Generated Code Scanning**: Analyze rendered scripts (Step 11 output) for unsafe patterns, injection risks, and dynamic execution constructs
- **AST Analysis**: Python AST scanner for `shell=True`, dangerous calls, and dynamic execution
- **Pre-Execution Gate**: `scan_script_for_execution()` blocks rendered scripts with high-severity findings before Step 12 runs them (`.jl` scripts: advisory sweep + `Meta.parseall` probe)
- **Report Generation**: Produce structured security findings with severity levels and remediation guidance

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `process_security()` | Function | Top-level entry point called by orchestrator |
| `perform_security_check()` | Function | Per-file sensitive-data and integrity check |
| `check_vulnerabilities()` | Function | Per-file pattern + AST vulnerability scan |
| `scan_script_for_execution()` | Function | Pre-execution gate for rendered scripts |
| `mcp.py` | MCP Tools | Security validation and audit tools |

## Standards

- Findings classified by severity: info, low, medium, high (`_SEVERITY_RANK`)
- Blocking is level-dependent: `basic` never blocks, `standard` reports only, `strict` blocks on high; an explicit `block_on` makes any scanning level enforcement-capable
- Reports generated in both JSON and Markdown formats
