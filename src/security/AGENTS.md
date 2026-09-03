# Security Module - Agent Scaffolding

## Module Overview

**Purpose**: Security scanning and validation for GNN pipeline files — injection-pattern and Python AST vulnerability detection, severity scoring, recommendations, and a pre-execution gate for rendered scripts.

**Pipeline Step**: Step 18: Security validation (18_security.py)

**Category**: Security / Vulnerability Scanning

**Status**: Production Ready

**Version**: 3.2.0

**Last Updated**: 2026-09-02

---

## Core Functionality

1. Scan GNN files for injection patterns and suspicious constructs (pattern matching + Python AST analysis)
2. Score files by vulnerability severity (0.0-1.0 security score)
3. Generate actionable security recommendations per file
4. Gate rendered scripts before Step 12 executes them (`scan_script_for_execution`)
5. Write `security_results.json` and `security_summary.md`

---

## API Reference

### Public Functions

#### `process_security(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main security processing function called by orchestrator (18_security.py). Scans all GNN files in `target_dir` and writes reports.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to scan
- `output_dir` (Path): Output directory for security reports
- `verbose` (bool): Enable verbose logging (default: False)
- `security_level` (str, via kwargs): `"basic"` (no scanning), `"standard"` (scan, report only), `"strict"` (scan + block on high severity) (default: `"standard"`)
- `block_on` (str, via kwargs): Explicit blocking threshold (`low`/`medium`/`high`); makes any level enforcement-capable
- `check_vulnerabilities` (bool, via kwargs): Force scanning on/off (strict mode cannot disable it)
- `**kwargs`: Additional security options

**Returns**: `bool` - True if security processing succeeded, False otherwise

**Example**:
```python
from security import process_security
from pathlib import Path

success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    verbose=True,
    security_level="strict",
)
```

#### `perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Sensitive-data and integrity check on a single file: scans for credential patterns (`password`, `secret`, `api_key`, `token`, `private_key`), hashes the exact bytes inspected (SHA-256), and scores the result.

**Parameters**:
- `file_path` (Path): Path to file to check
- `verbose` (bool): Enable verbose logging

**Returns**: `Dict[str, Any]` - Security check results with:
- `file_path` (str), `file_name` (str), `file_size` (int)
- `file_hash` (str): SHA-256 of the raw bytes
- `sensitive_patterns` (List[Dict]): Matched patterns with line number and redacted context
- `file_permissions` (str)
- `security_score` (float): Security score (0.0–1.0)
- `check_timestamp` (str)

#### `check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
**Description**: Scan a file for security vulnerabilities using pattern matching and Python AST analysis.

**Parameters**:
- `file_path` (Path): Path to file to scan
- `verbose` (bool): Enable verbose logging

**Returns**: `List[Dict[str, Any]]` - List of detected vulnerability dicts

#### `generate_security_recommendations(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
**Description**: Generate security improvement recommendations for a file.

#### `scan_script_for_execution(script_path: Path, *, block_on: str = "high") -> Dict[str, Any]`
**Description**: Pre-execution security gate for rendered scripts (RED_TEAM V-01/V-06).
Applies the Python AST scanner to a rendered `.py` script *before* Step 12 runs
it, returning `{ok, blocked, findings, scanned}`. Findings at/above `block_on`
severity set `ok=False`; `.jl` scripts get an advisory textual sweep plus a
`Meta.parseall` syntax probe (parse failure = high severity). Wired into
`execute.processor.execute_single_script` (escape hatch:
`GNN_ALLOW_UNSAFE_EXEC=1`).

---

## Dependencies

### Required Dependencies
- `pathlib`, `json`, `hashlib`, `re` - Standard library

### Optional Dependencies
- Julia interpreter - `Meta.parseall` syntax probe for `.jl` scripts in the pre-execution gate (advisory regex sweep without it)

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Security Levels (actual `_SECURITY_LEVELS` in `processor.py`)
```python
_SECURITY_LEVELS = {
    "basic": {"scan_vulnerabilities": False, "default_block_on": None},
    "standard": {"scan_vulnerabilities": True, "default_block_on": None},
    "strict": {"scan_vulnerabilities": True, "default_block_on": "high"},
}
```
An explicit `block_on` kwarg makes any scanning level enforcement-capable.
Strict mode cannot disable vulnerability scanning, since doing so would
silently weaken the requested policy.

---

## Usage Examples

### Basic Security Validation
```python
from security.processor import process_security

success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    security_level="standard",
)
```

---

## Output Specification

### Output Products
- `security_results.json` - Processing results
- `security_summary.md` - Human-readable security summary

### Output Directory Structure
```
output/18_security_output/
├── security_results.json
└── security_summary.md
```

---

## Security Features

### Threat Detection
1. **Injection Pattern Scanning**: Regex patterns for OS command injection, suspicious imports, and script-injection constructs
2. **Python AST Analysis**: Detects `shell=True`, dangerous calls, and dynamic execution
3. **Julia Script Analysis**: Advisory regex sweep plus `Meta.parseall` syntax probe (30 s timeout)
4. **Path Traversal Checks**: File-path based checks during scanning

### Pre-Execution Gate
`scan_script_for_execution()` is wired into Step 12's `execute_single_script()`:
rendered scripts are scanned before execution and blocked on findings at or
above the configured severity. Escape hatch: `GNN_ALLOW_UNSAFE_EXEC=1`.

---

## Security Features

### Threat Detection
1. **Malicious Content Detection**: Pattern-based threat detection
2. **Suspicious Script Detection**: Script injection detection
3. **Data Exfiltration Detection**: Unauthorized data access patterns
4. **Cryptographic Validation**: Digital signature verification

### Access Control
1. **File Permission Validation**: OS-level permission checks
2. **Operation Authorization**: Role-based access control
3. **Audit Logging**: Comprehensive operation logging
4. **Security Context**: Security-aware operation context

### Data Protection
1. **Encryption Support**: Sensitive data encryption
2. **Secure Storage**: Protected data storage
3. **Key Management**: Encryption key lifecycle management
4. **Data Sanitization**: Secure data cleanup

---

## Error Handling

### Error Categories
1. **Scan Errors**: Per-file scan failures are logged and skipped
2. **Threat Findings**: High-severity findings block the pre-execution gate (configurable via `block_on`)
3. **Report Errors**: Write failures cause a False return
---

## Integration Points

### Orchestrated By
- **Script**: `18_security.py` (Step 18)
- **Function**: `process_security()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `src/execute/processor.py` - Imports `scan_script_for_execution` for the Step 12 pre-execution gate
- `src/18_security.py` - Thin orchestrator (Step 18)
- `src/tests/security/*` - Security tests

### Data Flow
```
GNN Files → Pattern + AST Scanning → Severity Scoring → security_results.json + security_summary.md
Rendered Scripts → scan_script_for_execution → Block or Allow → Step 12 Execution
```

---

## Testing

### Test Files
- `src/tests/security/test_security_overall.py` - Module-level tests
- `src/tests/security/test_security_functional.py` - Functional tests
- `src/tests/security/test_pre_exec_gate.py` - Pre-execution gate tests
- `src/tests/security/test_security_mcp_tools.py` - MCP tool tests
- `src/tests/security/test_sandbox.py`, `test_pygments_archetype_redos.py` - Auxiliary security tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest src/tests/security/ \
    --cov=src/security --cov-report=term-missing
```

### Key Test Scenarios
1. Injection-pattern and AST vulnerability detection
2. Pre-execution gate blocking behavior and escape hatch
3. Security scoring and recommendations
4. Error handling with unscannable files

---

## MCP Integration

### Tools Registered
- `process_security` - Run security scanning and compliance checks on pipeline files
- `scan_gnn_file` - Lightweight security scan of a single GNN file
- `get_security_report` - Read saved reports from a previous security run
- `list_security_checks` - List the security checks performed (CVE scan, injection detection, path traversal, etc.)

### MCP File Location
- `src/security/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Security validation reports false positives
**Symptom**: Valid models reported as having vulnerabilities
**Cause**: Pattern rules may flag benign constructs
**Solution**:
- Use `security_level="basic"` to disable scanning (report-only pipeline still runs)
- Set an explicit `block_on` threshold to control what blocks
- Use `--verbose` for detailed scan logs

#### Issue 2: Rendered script blocked before execution
**Symptom**: Step 12 refuses to run a rendered script
**Cause**: `scan_script_for_execution` found findings at/above the blocking severity
**Solution**:
- Review the findings in the block report; fix the rendered script if genuinely unsafe
- Lower the threshold via `block_on` if the finding is advisory
- Last-resort escape hatch: set `GNN_ALLOW_UNSAFE_EXEC=1` (use only when the script is trusted)

---

## Version History

### Current Version: 1.6.0 (module `__init__.py`), pipeline release 3.2.0

**Features**:
- Injection-pattern and Python AST vulnerability scanning
- Severity-based security scoring
- Pre-execution gate for rendered scripts (Python + Julia)
- Security recommendations

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced threat detection
- **Future**: Real-time security monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Common Weakness Enumeration](https://cwe.mitre.org/)

---

**Last Updated**: 2026-09-02
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: 3.2.0
**Architecture Compliance**: Thin Orchestrator Pattern

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
