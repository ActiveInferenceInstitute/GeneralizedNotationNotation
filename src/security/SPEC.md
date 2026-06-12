# Specification: Security

## Design Requirements

The `src/security/` module provides comprehensive security validation and generated-code scanning for the GNN pipeline (Step 18).

## Interface Mapping

- `18_security.py`: Thin orchestrator binding `security.processor.process_security()`
- `processor.py`: Core security scanning engine — pattern matching, vulnerability detection, dependency auditing
- `mcp.py`: MCP tool registration for security validation operations

## Functional Requirements

- **Generated Code Scanning**: Analyze rendered scripts (Step 11 output) for unsafe patterns, injection risks, and resource abuse
- **Dependency Auditing**: Check imported packages against known vulnerability databases
- **Permission Validation**: Verify file system access patterns and network connectivity in generated code
- **Report Generation**: Produce structured security findings with severity levels and remediation guidance

## Components

| Component | Type | Description |
|-----------|------|-------------|
| `SecurityProcessor` | Class | Main scanning engine with configurable rule sets |
| `process_security()` | Function | Top-level entry point called by orchestrator |
| `mcp.py` | MCP Tools | Security validation and audit tools |

## Standards

- All findings classified by severity: Critical, High, Medium, Low, Info
- Reports generated in both JSON and Markdown formats
- Zero false-positive policy for Critical-severity findings
- Non-blocking — reports findings but does not halt pipeline
