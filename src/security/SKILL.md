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
    process_security,
    perform_security_check,
    check_vulnerabilities,
    generate_security_recommendations,
    calculate_security_score,
    generate_security_summary,
    resolve_security_policy,
    scan_source,
    scan_script_for_execution,
    findings_at_or_above,
    count_by_severity,
    SecurityScanError,
)

# Process security step (used by pipeline)
process_security(target_dir, output_dir, verbose=True)

# Perform security check (sensitive-data + integrity, per file)
result = perform_security_check(Path("models/model.md"))

# Check vulnerabilities (pattern + AST scan, per file)
vulns = check_vulnerabilities(Path("output/11_render_output/model_pymdp.py"))

# Get security score
score = calculate_security_score(vulns)

# Generate recommendations (per file)
recs = generate_security_recommendations(Path("models/model.md"))

# New in 1.7.0: pure policy resolution (validate before scanning)
policy = resolve_security_policy("strict")
assert policy.is_valid and policy.block_on == "high"

# New in 1.7.0: scan Python source text before writing it
verdict = scan_source(rendered_python, block_on="high")
if verdict["ok"]:
    out_path.write_text(rendered_python)

# New in 1.7.0: severity helpers + typed error
blocked = findings_at_or_above(vulns, "high")
histogram = count_by_severity(vulns)
```

## Key Exports

- `process_security` — main pipeline processing function
- `perform_security_check` — sensitive-data + integrity audit (raises `SecurityScanError`)
- `check_vulnerabilities` — vulnerability scanning (regex + AST + permissions)
- `calculate_security_score` — numeric security score (0-100)
- `generate_security_recommendations` / `generate_security_summary`
- `resolve_security_policy` / `ResolvedSecurityPolicy` — pure policy validation (1.7.0)
- `scan_source` — in-memory Python source scanning (1.7.0)
- `scan_script_for_execution` — pre-execution gate (exported since 1.7.0)
- `findings_at_or_above` / `count_by_severity` — severity helpers (1.7.0)

## Output

- Security reports in `output/18_security_output/`
- Vulnerability scan results
- Code safety audit logs


## MCP Tools

This module registers tools with the GNN MCP server (see `mcp.py`):

- `get_security_report`
- `list_security_checks`
- `process_security`
- `scan_gnn_file`

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
