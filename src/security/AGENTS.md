# Security Module - Agent Scaffolding

## Module Overview

**Purpose**: Security validation, access control, and threat detection for the GNN processing pipeline

**Pipeline Step**: Step 18: Security validation (18_security.py)

**Category**: Security / Access Control

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

#### `process_security(target_dir, output_dir, **kwargs) -> bool`
**Description**: Main security processing function

**Parameters**:
- `target_dir`: Directory containing files to validate
- `output_dir`: Output directory for security reports
- `**kwargs`: Additional security options

**Returns**: `True` if security validation passed

#### `validate_model_security(file_path, security_level="standard") -> Dict[str, Any]`
**Description**: Validate security aspects of a GNN model

**Parameters**:
- `file_path`: Path to GNN file to validate
- `security_level`: Security validation level

**Returns**: Dictionary with security validation results

#### `check_access_permissions(file_path, operation) -> bool`
**Description**: Check access permissions for file operations

**Parameters**:
- `file_path`: File path to check
- `operation`: Operation to validate

**Returns**: `True` if access is permitted

#### `detect_threats(content, threat_types=None) -> List[Dict[str, Any]]`
**Description**: Detect potential security threats in content

**Parameters**:
- `content`: Content to analyze
- `threat_types`: Types of threats to detect

**Returns**: List of detected threat indicators

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

### Model Security Check
```python
from security.processor import validate_model_security

security_result = validate_model_security(
    file_path="models/sensitive_model.md",
    security_level="strict"
)

if security_result["passed"]:
    print("Model security validation passed")
else:
    print("Security issues found:")
    for issue in security_result["issues"]:
        print(f"  - {issue}")
```

### Access Control Check
```python
from security.processor import check_access_permissions

if check_access_permissions("models/confidential.md", "read"):
    print("Access granted")
    # Proceed with model processing
else:
    print("Access denied")
    # Handle unauthorized access
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
- `src/tests/test_security_integration.py` - Integration tests
- `src/tests/test_security_validation.py` - Validation tests
- `src/tests/test_security_access.py` - Access control tests

### Test Coverage
- **Current**: 87%
- **Target**: 90%+

### Key Test Scenarios
1. Security validation with various threat types
2. Access control enforcement
3. Encryption and data protection
4. Audit logging functionality
5. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `security.validate_model` - Validate model security
- `security.check_access` - Check access permissions
- `security.detect_threats` - Detect security threats
- `security.audit_access` - Audit access control
- `security.encrypt_data` - Encrypt sensitive data

### Tool Endpoints
```python
@mcp_tool("security.validate_model")
def validate_model_security_tool(file_path, security_level="standard"):
    """Validate security aspects of a GNN model"""
    # Implementation
```

---

**Last Updated**: October 1, 2025
**Status**: ✅ Production Ready