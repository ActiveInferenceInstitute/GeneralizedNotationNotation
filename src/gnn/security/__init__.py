"""
Security module for GNN Processing Pipeline.

Security scanning and validation for GNN pipeline files: injection-pattern
and Python AST vulnerability detection, severity scoring, recommendations,
policy resolution, and a pre-execution gate for rendered scripts.
"""

from typing import Any

__version__ = "3.3.0"
FEATURES: dict[str, Any] = {
    "vulnerability_detection": True,
    "security_scoring": True,
    "access_control": True,
    "security_recommendations": True,
    "policy_resolution": True,
    "source_scanning": True,
    "pre_execution_gate": True,
    "mcp_integration": True,
}

# Import processor functions - single source of truth
from .processor import (
    ResolvedSecurityPolicy,
    SecurityScanError,
    calculate_security_score,
    check_vulnerabilities,
    count_by_severity,
    findings_at_or_above,
    generate_security_recommendations,
    generate_security_summary,
    perform_security_check,
    process_security,
    resolve_security_policy,
    scan_script_for_execution,
    scan_source,
)

__all__: list[str] = [
    "process_security",
    "perform_security_check",
    "check_vulnerabilities",
    "scan_source",
    "scan_script_for_execution",
    "resolve_security_policy",
    "ResolvedSecurityPolicy",
    "SecurityScanError",
    "findings_at_or_above",
    "count_by_severity",
    "generate_security_recommendations",
    "calculate_security_score",
    "generate_security_summary",
    "FEATURES",
    "__version__",
    "get_module_info",
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "security",
        "version": __version__,
        "description": "Security validation, vulnerability scanning, and access control",
        "features": FEATURES,
    }
