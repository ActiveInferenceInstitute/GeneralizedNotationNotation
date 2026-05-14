# Security Module - Agent Scaffolding

## Module Overview

**Purpose**: Security validation, access control, and threat detection for the GNN processing pipeline

**Pipeline Step**: Step 18: Security validation (18_security.py)

**Category**: Security / Access Control

**Status**: ✅ Production Ready

**Version**: 1.6.0

**Last Updated**: 2026-04-16

---

## Core Functionality

### Primary Responsibilities
1. Security validation of GNN models and pipeline components
2. Access control and authorization management
3. Threat detection and vulnerability assessment
4. Secure data handling and encryption
5. Security policy enforcement
6. Audit logging and compliance reporting

### Key Capabilities
- Model security validation and risk assessment
- Access control for sensitive operations
- Threat detection and mitigation
- Data encryption and secure storage
- Security policy configuration
- Audit trail maintenance
- Compliance reporting

---

## API Reference

### Public Functions

#### `process_security(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main security processing function called by orchestrator (18_security.py). Validates security, assesses vulnerabilities, and checks compliance.

**Parameters**:
- `target_dir` (Path): Directory containing files to validate
- `output_dir` (Path): Output directory for security reports
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `security_level` (str, optional): Security validation level ("basic", "standard", "strict") (default: "standard")
- `check_vulnerabilities` (bool, optional): Enable vulnerability scanning (default: True)
- `check_compliance` (bool, optional): Enable compliance checking (default: True)
- `compliance_standards` (List[str], optional): Standards to check against (default: ["OWASP Top 10"])
- `**kwargs`: Additional security options

**Returns**: `bool` - True if security validation passed, False otherwise

**Example**:
```python
from security import process_security
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_security(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/18_security_output"),
    logger=logger,
    verbose=True,
    security_level="strict",
    compliance_standards=["OWASP Top 10", "CWE"]
)
```

#### `perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Perform comprehensive security check on a file (AST analysis for Python, pattern scanning for all files).

**Parameters**:
- `file_path` (Path): Path to file to check
- `verbose` (bool): Enable verbose logging

**Returns**: `Dict[str, Any]` - Security check results with:
- `file` (str): File path checked
- `vulnerabilities` (List[Dict]): Detected vulnerabilities
- `security_score` (float): Security score (0.0–1.0)
- `recommendations` (List[Dict]): Security improvement recommendations

#### `check_vulnerabilities(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
**Description**: Scan a file for security vulnerabilities using pattern matching and Python AST analysis.

**Parameters**:
- `file_path` (Path): Path to file to scan
- `verbose` (bool): Enable verbose logging

**Returns**: `List[Dict[str, Any]]` - List of detected vulnerability dicts

#### `generate_security_recommendations(file_path: Path, verbose: bool = False) -> List[Dict[str, Any]]`
**Description**: Generate security improvement recommendations for a file.

#### `calculate_security_score(vulnerabilities: List[Dict]) -> float`
**Description**: Calculate overall security score (0.0–1.0) based on vulnerability severity weights.

---

## Dependencies

### Required Dependencies
- `cryptography` - Encryption and hashing
- `pathlib` - Path manipulation
- `json` - Data serialization

### Optional Dependencies
- `PyYAML` - Configuration file parsing
- `requests` - External security service integration

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Security Levels
```python
SECURITY_LEVELS = {
    'basic': {
        'validate_file_integrity': True,
        'check_basic_permissions': True,
        'log_access': True
    },
    'standard': {
        'validate_file_integrity': True,
        'check_basic_permissions': True,
        'log_access': True,
        'scan_for_malicious_content': True,
        'validate_model_structure': True
    },
    'strict': {
        'validate_file_integrity': True,
        'check_basic_permissions': True,
        'log_access': True,
        'scan_for_malicious_content': True,
        'validate_model_structure': True,
        'encrypt_sensitive_data': True,
        'require_authorization': True
    }
}
```

### Security Policies
```python
SECURITY_POLICIES = {
    'allowed_file_types': ['.md', '.json', '.yaml'],
    'max_file_size_mb': 100,
    'require_encryption': False,
    'audit_all_operations': True,
    'block_suspicious_content': True
}
```

---

## Usage Examples

### Basic Security Validation
```python
from security.processor import process_security

success = process_security(
    target_dir="input/gnn_files",
    output_dir="output/18_security_output",
    security_level="standard"
)
```

### File Security Check
```python
from security.processor import perform_security_check

security_result = perform_security_check(
    file_path=Path("models/sensitive_model.md"),
    verbose=True
)

if security_result["security_score"] > 0.8:
    print("Security check passed")
else:
    print("Security issues found:")
    for vuln in security_result["vulnerabilities"]:
        print(f"  - {vuln['description']}")
```

### Vulnerability Scanning
```python
from security.processor import check_vulnerabilities

vulns = check_vulnerabilities(Path("output/11_render_output/model_pymdp.py"))
for vuln in vulns:
    print(f"  [{vuln['severity']}] {vuln['description']}")
```

---

## Output Specification

### Output Products
- `security_validation_report.json` - Comprehensive security report
- `access_control_log.json` - Access control audit log
- `threat_detection_report.json` - Threat detection results
- `security_summary.md` - Human-readable security summary

### Output Directory Structure
```
output/18_security_output/
├── security_validation_report.json
├── access_control_log.json
├── threat_detection_report.json
├── security_summary.md
└── security_audit_trail/
    ├── 2025-10-01_access_log.json
    └── threat_indicators.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~1-3 seconds per model
- **Memory**: ~20-50MB
- **Status**: ✅ Production Ready

### Expected Performance
- **Basic Validation**: < 1 second
- **Standard Validation**: 1-2 seconds
- **Strict Validation**: 2-5 seconds
- **Threat Detection**: Variable based on content

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

### Security Errors
1. **Access Denied**: Insufficient permissions
2. **Threat Detected**: Malicious content found
3. **Validation Failed**: Security requirements not met
4. **Encryption Error**: Cryptographic operation failure

### Recovery Strategies
- **Access Issues**: Request elevated permissions
- **Threats**: Isolate and report suspicious content
- **Validation**: Provide remediation guidance
- **Encryption**: Use alternative encryption methods

---

## Integration Points

### Orchestrated By
- **Script**: `18_security.py` (Step 18)
- **Function**: `process_security()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- All pipeline steps requiring security validation
- `tests.test_security_*` - Security tests

### Data Flow
```
File Input → Security Validation → Threat Detection → Access Control → Security Report → Pipeline Continuation
```

---

## Testing

### Test Files
- `src/tests/security/test_security_overall.py` - Module-level tests
- `src/tests/security/test_security_functional.py` - Functional tests

### Test Coverage
Measure on demand:

```bash
uv run pytest src/tests/test_security*.py \
    --cov=src/security --cov-report=term-missing
```
### Key Test Scenarios
1. Security validation with various threat types
2. Access control enforcement
3. Encryption and data protection
4. Audit logging functionality
5. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `security.validate_model` - Perform security check on a model file
- `security.scan_file` - Scan file for vulnerabilities
- `security.get_report` - Get security report
- `security.list_checks` - List available security checks

### Tool Endpoints
```python
@mcp_tool("security.validate_model")
def validate_model_security_tool(file_path, security_level="standard"):
    """Validate security aspects of a GNN model"""
    # Implementation
```

### MCP File Location
- `src/security/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Security validation reports false positives
**Symptom**: Valid models reported as having vulnerabilities  
**Cause**: Security rules too strict or outdated  
**Solution**: 
- Use `--security-level basic` for lenient validation
- Review security rules and update if needed
- Check compliance standards are appropriate
- Use `--verbose` flag for detailed validation logs

#### Issue 2: Access control checks fail
**Symptom**: Valid operations blocked by access control  
**Cause**: Permission configuration incorrect or overly restrictive  
**Solution**:
- Verify file permissions are correct
- Check access control configuration
- Review security policy settings
- Ensure user has required permissions

---

## Version History

### Current Version: 1.6.0

**Features**:
- Security validation
- Access control
- Threat detection
- Vulnerability assessment
- Compliance reporting

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced threat detection
- **Future**: Real-time security monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../README.md)
- [Architecture Guide](../../ARCHITECTURE.md)
- [Security Guide](../../doc/security/)

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Common Weakness Enumeration](https://cwe.mitre.org/)

---

**Last Updated**: 2026-04-16
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Version**: 1.6.0
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
