---
name: gnn-security-validation
description: GNN security validation and access control. Use when auditing security of generated code, validating input sanitization, checking dependency vulnerabilities, or enforcing security policies on pipeline outputs.
---

# GNN Security Validation (Step 18)

## Purpose

Validates security aspects of the GNN pipeline including generated code safety, input sanitization, dependency vulnerability scanning, and access control enforcement.

## Key Commands

```bash
# Run security validation
python src/18_security.py --target-dir input/gnn_files --output-dir output --verbose

# As part of pipeline
python src/main.py --only-steps 18 --verbose
```

## API

```python
from security import (
    process_security, perform_security_check,
    check_vulnerabilities, generate_security_recommendations,
    calculate_security_score, generate_security_summary
)

# Process security step (used by pipeline)
process_security(target_dir, output_dir, verbose=True)

# Perform security check
result = perform_security_check(target_data)

# Check vulnerabilities
vulns = check_vulnerabilities(target_data)

# Get security score
score = calculate_security_score(security_results)

# Generate recommendations
recs = generate_security_recommendations(security_results)
```

## Key Exports

- `process_security` — main pipeline processing function
- `perform_security_check` — comprehensive security audit
- `check_vulnerabilities` — vulnerability scanning
- `calculate_security_score` — numeric security score
- `generate_security_recommendations` / `generate_security_summary`

## Output

- Security reports in `output/18_security_output/`
- Vulnerability scan results
- Code safety audit logs


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `process_security`
- `scan_gnn_file`
- `get_security_report`

## References

- [AGENTS.md](AGENTS.md) — Module documentation
- [README.md](README.md) — Usage guide
- [SPEC.md](SPEC.md) — Module specification


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
