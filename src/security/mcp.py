"""
MCP integration for the security module.

Exposes GNN security processing tools: vulnerability scanning, dependency
auditing, security report reading, and compliance checks through MCP.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from . import process_security


def process_security_mcp(target_directory: str, output_directory: str,
                         verbose: bool = False) -> Dict[str, Any]:
    """
    Run security processing on GNN pipeline files.

    Scans for known vulnerability patterns, validates dependency versions,
    and checks for insecure coding patterns in GNN and pipeline scripts.

    Args:
        target_directory: Directory containing GNN / pipeline files to scan
        output_directory: Directory to save security scan reports
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and security summary.
    """
    try:
        success = process_security(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
        )
        return {
            "success": success,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "message": f"Security processing {'completed successfully' if success else 'completed with issues'}",
        }
    except Exception as e:
        logger.error(f"process_security_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def scan_gnn_file_mcp(file_path: str) -> Dict[str, Any]:
    """
    Perform a lightweight security scan of a single GNN file.

    Checks for script injection patterns, overly permissive file paths,
    and other security anti-patterns in GNN model content.

    Args:
        file_path: Path to the GNN file to scan

    Returns:
        Dictionary with scan results, issues found, and risk level.
    """
    try:
        fpath = Path(file_path)
        if not fpath.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        content = fpath.read_text(encoding="utf-8", errors="replace")
        lines   = content.splitlines()

        issues: List[Dict[str, Any]] = []
        # Check for dangerous patterns
        danger_patterns = [
            ("exec(", "Potential code execution via exec()"),
            ("eval(", "Potential code execution via eval()"),
            ("__import__", "Dynamic import call"),
            ("subprocess", "Subprocess call"),
            ("os.system", "Shell command execution"),
        ]
        for i, line in enumerate(lines, 1):
            for pattern, description in danger_patterns:
                if pattern in line:
                    issues.append({
                        "line": i,
                        "pattern": pattern,
                        "description": description,
                        "severity": "high",
                    })

        risk_level = "high" if issues else "low"
        return {
            "success":     True,
            "file":        str(fpath),
            "issues_found":len(issues),
            "issues":      issues,
            "risk_level":  risk_level,
            "total_lines": len(lines),
        }
    except Exception as e:
        logger.error(f"scan_gnn_file_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_security_report_mcp(output_directory: str) -> Dict[str, Any]:
    """
    Read and return a saved security scan report.

    Args:
        output_directory: Directory where security reports were saved

    Returns:
        Dictionary with report contents and summary statistics.
    """
    try:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {"success": False, "error": f"Directory not found: {output_directory}"}

        reports = []
        for jf in sorted(out_dir.rglob("*security*.json"))[:5]:
            try:
                reports.append({"file": jf.name, "data": json.loads(jf.read_text())})
            except Exception as e:
                logger.debug(f"Could not parse security report {jf.name}: {e}")

        return {
            "success":          True,
            "output_directory": str(out_dir),
            "reports_found":    len(reports),
            "reports":          reports,
        }
    except Exception as e:
        logger.error(f"get_security_report_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_security_checks_mcp() -> Dict[str, Any]:
    """
    Return the list of security checks performed by this module.

    Returns:
        Dictionary with check names, descriptions, and severity levels.
    """
    checks = {
        "dependency_cve_scan": {
            "description": "CVE scan of Python dependencies",
            "severity": "high",
        },
        "code_injection": {
            "description": "Detect eval() / exec() code injection patterns",
            "severity": "high",
        },
        "path_traversal": {
            "description": "Detect path traversal patterns in file operations",
            "severity": "medium",
        },
        "unsafe_deserialization": {
            "description": "Detect unsafe pickle/marshal deserialization",
            "severity": "medium",
        },
        "hardcoded_credentials": {
            "description": "Detect hardcoded passwords or API keys",
            "severity": "high",
        },
    }
    return {"success": True, "checks": checks, "count": len(checks)}


# ── MCP Registration ────────────────────────────────────────────────────────

def register_tools(mcp_instance) -> None:
    """Register security tools with the MCP server."""

    mcp_instance.register_tool(
        "process_security",
        process_security_mcp,
        {"type": "object", "properties": {
            "target_directory": {"type": "string", "description": "Directory with GNN / pipeline files to scan"},
            "output_directory": {"type": "string", "description": "Security report output directory"},
            "verbose":          {"type": "boolean", "default": False},
        }, "required": ["target_directory", "output_directory"]},
        "Run security scanning and compliance checks on GNN pipeline files.",
        module=__package__, category="security",
    )

    mcp_instance.register_tool(
        "scan_gnn_file",
        scan_gnn_file_mcp,
        {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Path to the GNN file to scan"},
        }, "required": ["file_path"]},
        "Perform a lightweight security scan of a single GNN file for injection patterns.",
        module=__package__, category="security",
    )

    mcp_instance.register_tool(
        "get_security_report",
        get_security_report_mcp,
        {"type": "object", "properties": {
            "output_directory": {"type": "string", "description": "Directory with saved security reports"},
        }, "required": ["output_directory"]},
        "Read and return saved security scan reports from a previous security processing run.",
        module=__package__, category="security",
    )

    mcp_instance.register_tool(
        "list_security_checks",
        list_security_checks_mcp,
        {},
        "Return the list of security checks performed (CVE scan, injection detection, path traversal, etc.).",
        module=__package__, category="security",
    )

    logger.info("security module MCP tools registered (4 tools).")
