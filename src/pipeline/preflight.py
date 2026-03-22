#!/usr/bin/env python3
"""
Config Validation & Environment Checks — Pre-flight diagnostics.

Provides:
  - validate_config(): validates input/config.yaml structure and values
  - check_environment(): verifies required tools and dependencies
  - PreflightReport: structured result with issues and recommendations
"""

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreflightIssue:
    """A single preflight check result."""
    category: str  # config, dependency, environment, permission
    severity: str  # error, warning, info
    message: str
    fix: Optional[str] = None


@dataclass
class PreflightReport:
    """Complete preflight check result."""
    issues: List[PreflightIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0

    @property
    def is_ok(self) -> bool:
        return self.checks_failed == 0

    def add_pass(self, msg: str) -> None:
        self.checks_passed += 1
        logger.debug(f"✅ {msg}")

    def add_issue(self, category: str, severity: str, msg: str, fix: Optional[str] = None):
        self.issues.append(PreflightIssue(category=category, severity=severity, message=msg, fix=fix))
        if severity == "error":
            self.checks_failed += 1
        logger.warning(f"⚠️ [{category}] {msg}")

    def to_markdown(self) -> str:
        lines = ["# Preflight Check Report", ""]
        emoji = "🟢" if self.is_ok else "🔴"
        lines.append(f"{emoji} **{self.checks_passed} passed**, **{self.checks_failed} failed**")
        lines.append("")

        if self.issues:
            lines.append("## Issues")
            for issue in self.issues:
                sev = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
                lines.append(f"- {sev} **[{issue.category}]** {issue.message}")
                if issue.fix:
                    lines.append(f"  - Fix: `{issue.fix}`")

        return "\n".join(lines)


def validate_config(config_path: Optional[Path] = None) -> PreflightReport:
    """
    Validate pipeline config file.

    Args:
        config_path: Path to config.yaml. Defaults to input/config.yaml.

    Returns:
        PreflightReport with any issues found.
    """
    report = PreflightReport()
    config_path = config_path or Path("input/config.yaml")

    # Check file exists
    if not config_path.exists():
        report.add_issue("config", "warning", f"Config file not found: {config_path}",
                         fix="cp input/config.yaml.example input/config.yaml")
        return report

    report.add_pass(f"Config file exists: {config_path}")

    # Parse YAML
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except ImportError:
        # Manual parse
        config = {}
        report.add_issue("dependency", "info", "PyYAML not installed — limited config validation")
        return report
    except Exception as e:
        report.add_issue("config", "error", f"Config parse error: {e}")
        return report

    if not isinstance(config, dict):
        report.add_issue("config", "error", "Config file must be a YAML mapping")
        return report

    report.add_pass("Config is valid YAML")

    # Validate known sections
    if "llm" in config:
        llm = config["llm"]
        if "model" in llm:
            report.add_pass(f"LLM model configured: {llm['model']}")
        if "timeout_seconds" in llm:
            timeout = llm["timeout_seconds"]
            if not isinstance(timeout, (int, float)) or timeout < 0:
                report.add_issue("config", "error", f"Invalid llm.timeout_seconds: {timeout}")
            elif timeout > 3600:
                report.add_issue("config", "warning", f"Very large LLM timeout: {timeout}s")
            else:
                report.add_pass(f"LLM timeout: {timeout}s")

    return report


def check_environment() -> PreflightReport:
    """
    Check that required tools and dependencies are available.

    Returns:
        PreflightReport with environment status.
    """
    report = PreflightReport()

    # Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        report.add_pass(f"Python {py_version}")
    else:
        report.add_issue("environment", "error", f"Python {py_version} — requires ≥3.11",
                         fix="pyenv install 3.11")

    # Required packages
    for pkg in ["numpy", "pytest", "yaml"]:
        try:
            __import__(pkg)
            report.add_pass(f"Package: {pkg}")
        except ImportError:
            report.add_issue("dependency", "warning", f"Package not found: {pkg}",
                             fix=f"pip install {pkg}")

    # Optional packages
    for pkg, purpose in [("pydantic", "schemas"), ("fastapi", "API"), ("pygls", "LSP")]:
        try:
            __import__(pkg)
            report.add_pass(f"Optional: {pkg} ({purpose})")
        except ImportError:
            report.add_issue("dependency", "info", f"Optional package: {pkg} ({purpose})",
                             fix=f"pip install {pkg}")

    # Tools
    for tool in ["ollama", "ruff"]:
        if shutil.which(tool):
            report.add_pass(f"Tool: {tool}")
        else:
            sev = "info" if tool == "ollama" else "warning"
            report.add_issue("environment", sev, f"Tool not found: {tool}",
                             fix=f"See https://github.com/{tool}")

    # Directories
    for d in [Path("input/gnn_files"), Path("output")]:
        if d.exists():
            report.add_pass(f"Directory: {d}")
        else:
            report.add_issue("environment", "warning", f"Directory not found: {d}",
                             fix=f"mkdir -p {d}")

    return report


def run_preflight(config_path: Optional[Path] = None) -> PreflightReport:
    """Run all preflight checks and combine results."""
    config_report = validate_config(config_path)
    env_report = check_environment()

    combined = PreflightReport()
    combined.issues = config_report.issues + env_report.issues
    combined.checks_passed = config_report.checks_passed + env_report.checks_passed
    combined.checks_failed = config_report.checks_failed + env_report.checks_failed

    return combined
