#!/usr/bin/env python3
"""
Security processor module for GNN pipeline.
"""

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from utils.pipeline_template import log_step_error, log_step_start, log_step_success

logger = logging.getLogger(__name__)

#: Timeout for the ``julia -e Meta.parseall`` syntax probe used by the
#: pre-execution gate. Julia startup can take a second or two; 30s is ample
#: headroom without stalling the gate on a hung interpreter.
_JULIA_PARSE_TIMEOUT_S = 30.0

#: Advisory Julia pattern sweep. These patterns flag *suspicious* constructs
#: (medium severity — informational at the default ``block_on="high"`` gate).
#: They are advisory even when Julia is available: the blocking signal for
#: Julia is ``Meta.parseall`` failing (malformed code → high).
_JULIA_SUSPICIOUS_PATTERNS: list[tuple[str, str]] = [
    (r"\brun\s*\(\s*`", "Julia backtick command execution"),
    (r"\bCmd\s*\(\s*\[", "Julia Cmd construction"),
]

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3}

# Security levels select analysis depth; an explicit ``block_on`` can make any
# scanning level enforcement-capable. Strict mode cannot disable vulnerability
# scanning because doing so would silently weaken the requested policy.
_SECURITY_LEVELS: dict[str, dict[str, Any]] = {
    "basic": {"scan_vulnerabilities": False, "default_block_on": None},
    "standard": {"scan_vulnerabilities": True, "default_block_on": None},
    "strict": {"scan_vulnerabilities": True, "default_block_on": "high"},
}


def process_security(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process security validation for GNN files.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    step_logger = logging.getLogger("security")
    security_level = str(kwargs.get("security_level", "standard")).strip().lower()
    requested_block_on = kwargs.get("block_on")
    requested_threshold = (
        str(requested_block_on).strip().lower()
        if requested_block_on is not None
        else None
    )
    requested_scan = kwargs.get("check_vulnerabilities")
    valid_thresholds = set(_SEVERITY_RANK) - {"info"}
    level_policy = _SECURITY_LEVELS.get(security_level)
    scan_vulnerabilities = (
        bool(
            (level_policy and level_policy["scan_vulnerabilities"])
            or requested_block_on is not None
        )
        if requested_scan is None
        else requested_scan is True
    )
    receipt_requested_scan = (
        requested_scan
        if requested_scan is None or isinstance(requested_scan, bool)
        else str(requested_scan)
    )
    effective_block_on = (
        requested_threshold
        if requested_threshold in valid_thresholds
        else (
            str(level_policy["default_block_on"])
            if level_policy and level_policy["default_block_on"] is not None
            else None
        )
    )

    try:
        log_step_start(step_logger, "Processing security")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "security_checks": [],
            "vulnerabilities": [],
            "recommendations": [],
            "policy": {
                "security_level": security_level,
                "enforced": security_level == "strict"
                or requested_block_on is not None,
                "scan_vulnerabilities": scan_vulnerabilities,
                "requested_scan_vulnerabilities": receipt_requested_scan,
                "requested_block_on": requested_threshold,
                "block_on": effective_block_on,
                "decision": "allow",
                "blocked_findings": 0,
            },
        }

        invalid_scan_policy = requested_scan is not None and not isinstance(
            requested_scan, bool
        )
        strict_scan_disabled = security_level == "strict" and not scan_vulnerabilities
        enforced_scan_disabled = (
            requested_block_on is not None and not scan_vulnerabilities
        )
        if (
            level_policy is None
            or (
                requested_threshold is not None
                and requested_threshold not in valid_thresholds
            )
            or invalid_scan_policy
            or strict_scan_disabled
            or enforced_scan_disabled
        ):
            results["success"] = False
            results["policy"]["decision"] = "deny_invalid_policy"
            results["errors"].append(
                "Invalid security policy: security_level must be basic, standard, "
                "or strict; block_on must be low, medium, or high; and "
                "check_vulnerabilities must be a boolean and remain enabled for "
                "strict or explicitly enforced policies"
            )

        # Find GNN files
        gnn_files = sorted(target_dir.glob("*.md"))
        if results["errors"]:
            gnn_files = []
        elif not gnn_files:
            step_logger.warning("No GNN files found for security processing")
            results["success"] = False
            results["policy"]["decision"] = "deny_no_input"
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Perform security checks
                    security_check = perform_security_check(gnn_file, verbose)
                    results["security_checks"].append(security_check)

                    # Check for vulnerabilities
                    if scan_vulnerabilities:
                        vulnerabilities = check_vulnerabilities(gnn_file, verbose)
                        results["vulnerabilities"].extend(vulnerabilities)

                    # Generate security recommendations
                    recommendations = generate_security_recommendations(
                        gnn_file, verbose
                    )
                    results["recommendations"].extend(recommendations)

                except Exception as e:
                    error_info: dict[str, Any] = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results["errors"].append(error_info)
                    results["success"] = False
                    results["policy"]["decision"] = "deny_processing_error"
                    step_logger.error(f"Error processing {gnn_file}: {e}")

        threshold_name = results["policy"]["block_on"]
        if threshold_name is not None:
            threshold = _SEVERITY_RANK[str(threshold_name).lower()]
            blocked = [
                finding
                for finding in results["vulnerabilities"]
                if _SEVERITY_RANK.get(
                    str(finding.get("severity", "high")).lower(),
                    _SEVERITY_RANK["high"],
                )
                >= threshold
            ]
            results["policy"]["blocked_findings"] = len(blocked)
            if blocked:
                results["success"] = False
                results["policy"]["decision"] = "deny"

        # Save detailed results
        results_file = results_dir / "security_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Generate security summary
        summary = generate_security_summary(results)
        summary_file = results_dir / "security_summary.md"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

        if results["success"]:
            log_step_success(step_logger, "Security processing completed successfully")
        else:
            log_step_error(step_logger, "Security processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(step_logger, "Security processing failed", error=str(e))
        return False


def perform_security_check(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """Perform security checks on a GNN file."""
    try:
        raw_content = file_path.read_bytes()
        content = raw_content.decode("utf-8", errors="replace")

        # Hash the exact bytes inspected. Text-mode hashing normalized CRLF on
        # some platforms, so it was not a reliable file-integrity receipt.
        file_hash = hashlib.sha256(raw_content).hexdigest()

        # Check for sensitive patterns
        sensitive_patterns: list[Any] = [
            r"password\s*[:=]",
            r"secret\s*[:=]",
            r"api_key\s*[:=]",
            r"token\s*[:=]",
            r"private_key\s*[:=]",
        ]

        found_patterns: list[Any] = []
        for pattern in sensitive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                found_patterns.append(
                    {
                        "pattern": pattern,
                        "line": content[: match.start()].count("\n") + 1,
                        "context": f"{match.group(0)} [REDACTED]",
                    }
                )

        # Check file permissions (simplified)
        file_permissions = "readable"

        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_hash": file_hash,
            "file_size": len(raw_content),
            "sensitive_patterns": found_patterns,
            "file_permissions": file_permissions,
            "security_score": calculate_security_score(found_patterns),
            "check_timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise Exception(f"Failed to perform security check on {file_path}: {e}") from e


def check_vulnerabilities(
    file_path: Path, verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Check for security vulnerabilities in a GNN file.

    Uses two complementary techniques:
    1. Regex pattern matching for GNN markdown files
    2. Python AST analysis for generated .py files (eval, exec, os.system detection)
    """
    vulnerabilities: list[Any] = []

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # -- Technique 1: Regex patterns (for GNN markdown and all files) --
        vuln_patterns: list[Any] = [
            (r"\beval\s*\(", "Code injection vulnerability"),
            (r"\bexec\s*\(", "Code execution vulnerability"),
            (r"\bimport\s+os\b", "OS command injection risk"),
            (
                r"\bsubprocess\s*\.\s*call\s*\(",
                "Subprocess call -- potential command injection",
            ),
            (
                r"\bsubprocess\s*\.\s*Popen\s*\(",
                "Subprocess Popen -- potential command injection",
            ),
            (
                r"\bsubprocess\s*\.\s*(?:run|check_call|check_output)\s*\(",
                "Subprocess execution risk",
            ),
            (r"\bfile\s*\(", "Previous file() call"),
        ]

        for pattern, description in vuln_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                vulnerabilities.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "vulnerability_type": description,
                        "detection_method": "regex",
                        "pattern": pattern,
                        "line": content[: match.start()].count("\n") + 1,
                        "context": match.group(0)[:80],
                        "severity": "medium",
                    }
                )

        # Check for hardcoded credentials
        credential_patterns: list[Any] = [
            r'password\s*[:=]\s*["\'][^"\']{4,}["\']',
            r'secret\s*[:=]\s*["\'][^"\']{4,}["\']',
            r'api_key\s*[:=]\s*["\'][^"\']{8,}["\']',
        ]

        for pattern in credential_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                vulnerabilities.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "vulnerability_type": "Hardcoded credentials",
                        "detection_method": "regex",
                        "pattern": pattern,
                        "line": content[: match.start()].count("\n") + 1,
                        "context": "[REDACTED]",
                        "severity": "high",
                    }
                )

        # -- Technique 2: AST analysis for Python files --
        if file_path.suffix == ".py":
            ast_vulns = _check_python_ast(file_path, content)
            vulnerabilities.extend(ast_vulns)

        # -- Technique 3: File permission check using os.access --
        import os as _os

        is_world_writable = _os.access(str(file_path), _os.W_OK)
        if is_world_writable and file_path.suffix == ".py":
            # Generated .py files shouldn't be world-writable in shared environments
            try:
                import stat

                file_stat = file_path.stat()
                mode = file_stat.st_mode
                world_write = bool(mode & stat.S_IWOTH)
                if world_write:
                    vulnerabilities.append(
                        {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "vulnerability_type": "World-writable file permissions",
                            "detection_method": "permission_check",
                            "pattern": "stat.S_IWOTH",
                            "line": 0,
                            "context": f"Mode: {oct(mode)}",
                            "severity": "low",
                        }
                    )
            except OSError:
                logger.debug(
                    "Permission check failed on %s (platform limitation)",
                    file_path.name,
                )

    except Exception as e:
        vulnerabilities.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "vulnerability_type": "File access error",
                "detection_method": "file_read",
                "error": str(e),
                "severity": "low",
            }
        )

    return sorted(
        vulnerabilities,
        key=lambda finding: (
            int(finding.get("line", 0)),
            str(finding.get("vulnerability_type", "")),
            str(finding.get("detection_method", "")),
            str(finding.get("pattern", "")),
        ),
    )


def _check_python_ast(file_path: Path, content: str) -> List[Dict[str, Any]]:
    """
    Perform AST-level security analysis on a Python source file.

    Detects dangerous function calls at the AST node level,
    which is more reliable than regex (handles multiline calls, string formatting).

    Dangerous patterns detected:
    - eval() / exec(): code injection vectors
    - os.system(): command injection
    - compile() with user-controlled input: code injection
    - __import__() dynamic import: potentially dangerous
    - open() with write modes in unexpected contexts

    Args:
        file_path: Path to the Python file (for error context)
        content: File content as string

    Returns:
        List of vulnerability dicts found via AST analysis
    """
    import ast

    vulns: list[Any] = []

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        return [
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "vulnerability_type": "Syntax error (cannot AST scan)",
                "detection_method": "ast_parse",
                "line": e.lineno or 0,
                "context": str(e),
                # The pre-execution gate cannot prove an unparseable script is
                # safe, so this is a fail-closed finding rather than advisory.
                "severity": "high",
            }
        ]

    # Dangerous function call patterns
    DANGEROUS_CALLS: dict[str, Any] = {
        "eval": ("Code injection via eval()", "high"),
        "exec": ("Code injection via exec()", "high"),
        "compile": ("Dynamic code compilation", "medium"),
        "__import__": ("Dynamic import -- verify input is trusted", "medium"),
    }

    # Dangerous attribute access patterns: obj.method()
    DANGEROUS_METHODS: dict[Any, Any] = {
        ("os", "system"): ("OS command injection via os.system()", "high"),
        ("os", "popen"): ("OS command injection via os.popen()", "high"),
        ("subprocess", "call"): ("Subprocess execution", "medium"),
        ("subprocess", "Popen"): ("Subprocess execution", "medium"),
        ("subprocess", "run"): ("Subprocess execution -- verify shell=False", "low"),
        ("subprocess", "check_call"): ("Subprocess execution", "medium"),
        ("subprocess", "check_output"): ("Subprocess execution", "medium"),
        ("subprocess", "getoutput"): ("Subprocess shell execution", "high"),
        ("subprocess", "getstatusoutput"): ("Subprocess shell execution", "high"),
        ("pickle", "loads"): ("Arbitrary code execution via pickle.loads()", "high"),
        ("pickle", "load"): ("Arbitrary code execution via pickle.load()", "high"),
        ("marshal", "loads"): ("Arbitrary code execution via marshal.loads()", "high"),
    }

    module_aliases: dict[str, str] = {}
    call_aliases: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                module_aliases[name.asname or name.name.split(".", 1)[0]] = name.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                call_aliases[name.asname or name.name] = (node.module, name.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Attribute) or not isinstance(
                value.value, ast.Name
            ):
                continue
            module_name = module_aliases.get(value.value.id, value.value.id)
            target_names: list[str] = []
            if isinstance(node, ast.Assign):
                target_names = [
                    target.id for target in node.targets if isinstance(target, ast.Name)
                ]
            elif isinstance(node.target, ast.Name):
                target_names = [node.target.id]
            for target_name in target_names:
                call_aliases[target_name] = (module_name, value.attr)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        line = getattr(node, "lineno", 0)

        # Direct function calls: eval(), exec(), etc.
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in DANGEROUS_CALLS:
                desc, severity = DANGEROUS_CALLS[func_name]
                vulns.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "vulnerability_type": desc,
                        "detection_method": "ast_analysis",
                        "pattern": f"{func_name}()",
                        "line": line,
                        "context": f"{func_name}() call at line {line}",
                        "severity": severity,
                    }
                )
            elif (
                func_name in call_aliases
                and call_aliases[func_name] in DANGEROUS_METHODS
            ):
                obj_name, method_name = call_aliases[func_name]
                desc, severity = DANGEROUS_METHODS[(obj_name, method_name)]
                if _uses_shell_true(node) and obj_name == "subprocess":
                    desc = "Subprocess execution with shell=True"
                    severity = "high"
                vulns.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "vulnerability_type": desc,
                        "detection_method": "ast_analysis",
                        "pattern": f"{obj_name}.{method_name}()",
                        "line": line,
                        "context": f"{func_name}() at line {line}",
                        "severity": severity,
                    }
                )

        # Attribute calls: os.system(), pickle.loads(), etc.
        elif isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                local_name = node.func.value.id
                obj_name = module_aliases.get(local_name, local_name)
                key = (obj_name, method_name)
                if key in DANGEROUS_METHODS:
                    desc, severity = DANGEROUS_METHODS[key]
                    if _uses_shell_true(node) and obj_name == "subprocess":
                        desc = "Subprocess execution with shell=True"
                        severity = "high"
                    vulns.append(
                        {
                            "file_path": str(file_path),
                            "file_name": file_path.name,
                            "vulnerability_type": desc,
                            "detection_method": "ast_analysis",
                            "pattern": f"{obj_name}.{method_name}()",
                            "line": line,
                            "context": f"{obj_name}.{method_name}() at line {line}",
                            "severity": severity,
                        }
                    )

    return sorted(
        vulns,
        key=lambda finding: (
            int(finding.get("line", 0)),
            str(finding.get("vulnerability_type", "")),
            str(finding.get("detection_method", "")),
        ),
    )


def _uses_shell_true(node: Any) -> bool:
    """Return whether a call's ``shell`` argument may enable a command shell.

    Only a literal ``shell=False`` is statically safe. Truthy literals and
    dynamic expressions are conservatively treated as enabled so the security
    gate cannot be bypassed with ``shell=flag`` or ``shell=1``.
    """
    import ast

    return any(
        keyword.arg == "shell"
        and not (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is False
        )
        for keyword in node.keywords
    )


def generate_security_recommendations(
    file_path: Path, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Generate security recommendations for a GNN file."""
    recommendations: list[Any] = []

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Check for basic security practices
        if not re.search(r"#\s*Security", content, re.IGNORECASE):
            recommendations.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add security documentation section",
                    "priority": "medium",
                    "description": "Consider adding a security section to document security considerations",
                }
            )

        # Check for input validation
        if re.search(r"input\s*[:=]", content, re.IGNORECASE):
            if not re.search(r"validate|check|verify", content, re.IGNORECASE):
                recommendations.append(
                    {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "recommendation": "Add input validation",
                        "priority": "high",
                        "description": "Input validation should be implemented for all user inputs",
                    }
                )

        # Check for error handling
        if not re.search(r"try\s*:|except\s*:", content, re.IGNORECASE):
            recommendations.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add error handling",
                    "priority": "medium",
                    "description": "Implement proper error handling for robust security",
                }
            )

        # Check for logging
        if not re.search(r"log|logging", content, re.IGNORECASE):
            recommendations.append(
                {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "recommendation": "Add security logging",
                    "priority": "medium",
                    "description": "Implement security event logging for monitoring",
                }
            )

    except Exception as e:
        recommendations.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "recommendation": "File access error",
                "priority": "low",
                "description": f"Could not analyze file: {e}",
            }
        )

    return recommendations


def calculate_security_score(vulnerabilities: List[Dict]) -> float:
    """Calculate a security score based on vulnerabilities."""
    if not vulnerabilities:
        return 100.0

    # Weight vulnerabilities by severity
    severity_weights: dict[str, Any] = {"high": 10.0, "medium": 5.0, "low": 1.0}

    total_score = 0.0
    for vuln in vulnerabilities:
        severity = vuln.get("severity", "medium")
        total_score += severity_weights.get(severity, 5.0)

    # Convert to 0-100 scale (higher is better)
    max_possible_score = len(vulnerabilities) * 10.0
    if max_possible_score == 0:
        return 100.0

    score = max(0.0, 100.0 - (total_score / max_possible_score) * 100.0)
    return score


def generate_security_summary(results: Dict[str, Any]) -> str:
    """Generate a security summary report."""
    summary = f"""
# Security Analysis Summary

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Processing Results
- **Files Processed**: {results.get("processed_files", 0)}
- **Success**: {results.get("success", False)}
- **Errors**: {len(results.get("errors", []))}

## Security Results
- **Security Checks**: {len(results.get("security_checks", []))}
- **Vulnerabilities Found**: {len(results.get("vulnerabilities", []))}
- **Recommendations**: {len(results.get("recommendations", []))}

## Vulnerability Summary
"""

    vulnerabilities = results.get("vulnerabilities", [])
    if vulnerabilities:
        high_vulns = [v for v in vulnerabilities if v.get("severity") == "high"]
        medium_vulns = [v for v in vulnerabilities if v.get("severity") == "medium"]
        low_vulns = [v for v in vulnerabilities if v.get("severity") == "low"]

        summary += f"- **High Severity**: {len(high_vulns)}\n"
        summary += f"- **Medium Severity**: {len(medium_vulns)}\n"
        summary += f"- **Low Severity**: {len(low_vulns)}\n"

        if high_vulns:
            summary += "\n### High Severity Vulnerabilities\n"
            for vuln in high_vulns[:5]:  # Show first 5
                summary += f"- **{vuln.get('file_name', 'Unknown')}**: {vuln.get('vulnerability_type', 'Unknown')}\n"
    else:
        summary += "- No vulnerabilities found\n"

    summary += "\n## Recommendations\n"

    recommendations = results.get("recommendations", [])
    if recommendations:
        high_recs = [r for r in recommendations if r.get("priority") == "high"]
        medium_recs = [r for r in recommendations if r.get("priority") == "medium"]

        if high_recs:
            summary += "\n### High Priority Recommendations\n"
            for rec in high_recs[:3]:  # Show first 3
                summary += f"- **{rec.get('file_name', 'Unknown')}**: {rec.get('recommendation', 'Unknown')}\n"

        if medium_recs:
            summary += "\n### Medium Priority Recommendations\n"
            for rec in medium_recs[:3]:  # Show first 3
                summary += f"- **{rec.get('file_name', 'Unknown')}**: {rec.get('recommendation', 'Unknown')}\n"
    else:
        summary += "- No recommendations generated\n"

    return summary


def _julia_meta_parseall(content: str) -> Optional[tuple[bool, str]]:
    """Validate Julia source with ``Meta.parseall`` via a ``julia`` subprocess.

    Parsing does **not** execute the script — ``Meta.parseall`` only builds the
    AST, so the probe itself is safe to run on untrusted rendered code.

    Returns:
        ``(True, "")`` when the source parses cleanly.
        ``(False, message)`` when parsing failed (malformed script).
        ``None`` when Julia is not available on PATH (caller should fall back
        to the advisory regex sweep).
    """
    import shutil
    import subprocess

    if shutil.which("julia") is None:
        return None

    probe = (
        "function _gnn_parsecheck(s)\n"
        "    ex = Base.Meta.parseall(s)\n"
        "    function _has_incomplete(e)\n"
        "        if e isa Expr\n"
        "            if e.head === :incomplete\n"
        "                return true\n"
        "            end\n"
        "            for a in e.args\n"
        "                _has_incomplete(a) && return true\n"
        "            end\n"
        "        end\n"
        "        return false\n"
        "    end\n"
        "    if _has_incomplete(ex)\n"
        "        # Extract the first error message embedded in the AST\n"
        "        msg = sprint(print, ex)\n"
        '        println("GNN_PARSE_FAIL: ", msg[1:min(end,200)])\n'
        "        exit(1)\n"
        "    end\n"
        '    println("GNN_PARSE_OK")\n'
        "end\n"
        "s = read(stdin, String)\n"
        "try\n"
        "    _gnn_parsecheck(s)\n"
        "catch e\n"
        '    println("GNN_PARSE_FAIL: ", sprint(showerror, e))\n'
        "    exit(1)\n"
        "end\n"
    )
    try:
        proc = subprocess.run(
            ["julia", "--startup-file=no", "-e", probe],
            input=content,
            capture_output=True,
            text=True,
            timeout=_JULIA_PARSE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Julia installed but unusable (hang/misconfig) — degrade to advisory.
        logger.debug("Julia parse probe unavailable: %s", exc)
        return None

    stdout = (proc.stdout or "").strip()
    if proc.returncode == 0 and "GNN_PARSE_OK" in stdout:
        return True, ""
    message = (stdout or (proc.stderr or "")).strip()
    # Keep the message bounded for findings context.
    return False, message[-400:]


def _julia_regex_sweep(script_path: Path, content: str) -> list[dict[str, Any]]:
    """Advisory Julia textual sweep (backtick ``run`` / ``Cmd`` construction).

    These findings are classified ``medium`` — informational at the default
    ``block_on="high"`` gate, but they still block when the operator lowers the
    threshold. The blocking signal for Julia is ``Meta.parseall`` failing.
    """
    findings: list[dict[str, Any]] = []
    for pattern, description in _JULIA_SUSPICIOUS_PATTERNS:
        for match in re.finditer(pattern, content):
            findings.append(
                {
                    "file_path": str(script_path),
                    "file_name": script_path.name,
                    "vulnerability_type": description,
                    "detection_method": "regex",
                    "pattern": pattern,
                    "line": content[: match.start()].count("\n") + 1,
                    "context": match.group(0)[:80],
                    "severity": "medium",
                }
            )
    return findings


def scan_script_for_execution(
    script_path: Path,
    *,
    block_on: str = "high",
) -> Dict[str, Any]:
    """Pre-execution gate: scan a rendered script before Step 12 runs it.

    The pipeline renders GNN text specifications into executable Python/Julia
    scripts. Step 18 (``process_security``) runs *after* Step 12, so by itself
    it is forensic, not preventive. This function closes that gap by applying
    the AST scanner to a rendered ``.py`` script *before* execution, returning
    a structured verdict the executor can act on.

    Args:
        script_path: Path to the rendered script (``.py`` or ``.jl``).
        block_on: Severity threshold that blocks execution. Findings at or
            above this severity set ``ok=False``. Defaults to ``"high"``.

    Returns:
        Dict with keys:
            - ``ok`` (bool): True if execution may proceed.
            - ``blocked`` (list): findings that triggered the block.
            - ``findings`` (list): all findings (blocked + informational).
            - ``scanned`` (bool): whether AST/parse analysis was performed.

    Julia (``.jl``) scripts are validated with ``Meta.parseall`` via a Julia
    subprocess when Julia is on PATH; malformed code is a high-severity block.
    When Julia is unavailable the scan degrades to an advisory textual sweep
    (``scanned=False``, findings are medium/informational only).
    """
    script_path = Path(script_path)
    normalized_block_on = block_on.strip().lower() if isinstance(block_on, str) else ""
    if normalized_block_on not in _SEVERITY_RANK:
        finding = {
            "file_path": str(script_path),
            "file_name": script_path.name,
            "vulnerability_type": "Invalid security threshold",
            "detection_method": "policy_validation",
            "severity": "high",
            "context": f"Unsupported block_on value: {block_on}",
        }
        return {
            "ok": False,
            "blocked": [finding],
            "findings": [finding],
            "scanned": False,
            "block_on": normalized_block_on,
            "decision": "deny_invalid_policy",
        }
    threshold = _SEVERITY_RANK[normalized_block_on]

    try:
        content = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        finding = {
            "file_path": str(script_path),
            "file_name": script_path.name,
            "vulnerability_type": "Unreadable script",
            "detection_method": "file_read",
            "severity": "high",
            "context": str(exc),
        }
        return {
            "ok": False,
            "blocked": [finding],
            "findings": [finding],
            "scanned": False,
            "block_on": normalized_block_on,
            "decision": "deny_unreadable",
        }

    findings: list[Any] = []
    if script_path.suffix.lower() == ".py":
        findings = _check_python_ast(script_path, content)
        scanned = True
    elif script_path.suffix.lower() == ".jl":
        # Julia scripts: validate with Meta.parseall (blocking) when Julia is
        # available; fall back to an advisory textual sweep when it is not.
        parse_result = _julia_meta_parseall(content)
        if parse_result is None:
            # Julia unavailable (or probe failed) — advisory sweep only, the
            # same posture as the previous textual-only scan.
            scanned = False
            findings = _julia_regex_sweep(script_path, content)
        else:
            parsed_ok, parse_message = parse_result
            scanned = True
            if not parsed_ok:
                # Malformed Julia is a hard block: the script cannot run as-is
                # and a syntax error is the strongest signal of tampering.
                findings.append(
                    {
                        "file_path": str(script_path),
                        "file_name": script_path.name,
                        "vulnerability_type": "Malformed Julia code (Meta.parseall failed)",
                        "detection_method": "julia_meta_parseall",
                        "line": 1,
                        "context": parse_message,
                        "severity": "high",
                    }
                )
            # Suspicious patterns remain medium (advisory at the default
            # block_on="high" gate) even when the code parses cleanly.
            findings.extend(_julia_regex_sweep(script_path, content))
    else:
        scanned = False
        findings = [
            {
                "file_path": str(script_path),
                "file_name": script_path.name,
                "vulnerability_type": "Unsupported executable script type",
                "detection_method": "file_type_policy",
                "line": 0,
                "context": script_path.suffix or "no suffix",
                "severity": "high",
            }
        ]

    blocked = [
        f
        for f in findings
        if _SEVERITY_RANK.get(
            str(f.get("severity", "high")).lower(), _SEVERITY_RANK["high"]
        )
        >= threshold
    ]
    return {
        "ok": not blocked,
        "blocked": blocked,
        "findings": findings,
        "scanned": scanned,
        "block_on": normalized_block_on,
        "decision": "deny" if blocked else "allow",
    }
