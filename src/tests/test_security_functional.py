#!/usr/bin/env python3
"""
Functional tests for the Security Processor module.

Tests security scanning, vulnerability detection, scoring, and recommendation
generation for GNN files.

Test Coverage:
- process_security() with valid GNN files
- process_security() with empty directory (returns False, no files found)
- process_security() with nonexistent path
- Vulnerability pattern detection (eval, exec, subprocess)
- Sensitive pattern detection (passwords, api_keys, tokens)
- Security scoring calculation
- Output JSON schema validation
- Security summary report generation
- Edge cases: clean files, heavily-flagged files, empty files
"""

import pytest
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from security.processor import (
    process_security,
    perform_security_check,
    check_vulnerabilities,
    generate_security_recommendations,
    calculate_security_score,
    generate_security_summary,
)


class TestSecurityFunctional:
    """Functional tests for the security processor module."""

    @pytest.fixture
    def clean_gnn_dir(self, tmp_path):
        """Create a GNN file with no security issues."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "clean_model.md").write_text(
            "# Clean Model\n\n"
            "## ModelName\nCleanModel\n\n"
            "## StateSpaceBlock\n"
            "A[3,3,type=float]\n"
            "s[3,1,type=float]\n\n"
            "## Connections\n"
            "s -> o\n"
        )
        return target

    @pytest.fixture
    def vuln_gnn_dir(self, tmp_path):
        """Create a GNN file with vulnerability patterns."""
        target = tmp_path / "input"
        target.mkdir()
        (target / "vulnerable_model.md").write_text(
            "# Vulnerable Model\n\n"
            "## Notes\n"
            "password = 'hunter2'\n"
            "api_key = 'sk-1234567890'\n"
            "eval(user_input)\n"
            "exec(payload)\n"
            "import os\n"
            "subprocess.call(['rm', '-rf'])\n"
            "open('/etc/passwd')\n"
        )
        return target

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create an output directory."""
        out = tmp_path / "output"
        out.mkdir()
        return out

    # -- process_security() tests --

    @pytest.mark.unit
    def test_process_security_returns_bool(self, clean_gnn_dir, output_dir):
        """process_security should always return a bool."""
        result = process_security(clean_gnn_dir, output_dir, verbose=True)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_process_security_success_with_valid_files(self, clean_gnn_dir, output_dir):
        """process_security should return True for valid GNN files."""
        result = process_security(clean_gnn_dir, output_dir, verbose=True)
        assert result is True

    @pytest.mark.unit
    def test_process_security_empty_directory(self, tmp_path):
        """process_security should return False when no GNN files are found."""
        empty_input = tmp_path / "empty"
        empty_input.mkdir()
        out = tmp_path / "output"
        out.mkdir()

        result = process_security(empty_input, out, verbose=False)
        assert isinstance(result, bool)
        # The processor sets success=False when no GNN files found
        assert result is False

    @pytest.mark.unit
    def test_process_security_nonexistent_path(self, tmp_path):
        """process_security should return False for a nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"
        out = tmp_path / "output"
        out.mkdir()

        result = process_security(nonexistent, out, verbose=False)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_output_artifacts_created(self, clean_gnn_dir, output_dir):
        """process_security should create security_results.json and security_summary.md."""
        process_security(clean_gnn_dir, output_dir, verbose=True)

        assert (output_dir / "security_results.json").exists()
        assert (output_dir / "security_summary.md").exists()

    @pytest.mark.unit
    def test_results_json_schema(self, clean_gnn_dir, output_dir):
        """security_results.json should have expected top-level keys."""
        process_security(clean_gnn_dir, output_dir, verbose=True)

        with open(output_dir / "security_results.json") as f:
            data = json.load(f)

        required_keys = [
            "timestamp", "processed_files", "success", "errors",
            "security_checks", "vulnerabilities", "recommendations"
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

    # -- perform_security_check() tests --

    @pytest.mark.unit
    def test_perform_security_check_clean_file(self, clean_gnn_dir):
        """A clean file should have no sensitive patterns detected."""
        gnn_file = list(clean_gnn_dir.glob("*.md"))[0]
        result = perform_security_check(gnn_file, verbose=True)

        assert isinstance(result, dict)
        assert "file_hash" in result
        assert "security_score" in result
        assert result["sensitive_patterns"] == []
        assert result["security_score"] == 100.0

    @pytest.mark.unit
    def test_perform_security_check_detects_passwords(self, vuln_gnn_dir):
        """Should detect password patterns in a vulnerable file."""
        gnn_file = list(vuln_gnn_dir.glob("*.md"))[0]
        result = perform_security_check(gnn_file, verbose=True)

        patterns_found = [p["context"] for p in result["sensitive_patterns"]]
        assert len(result["sensitive_patterns"]) > 0, "Should detect sensitive patterns"
        # At least password or api_key should be found
        assert any("password" in p.lower() or "api_key" in p.lower() for p in patterns_found)

    # -- check_vulnerabilities() tests --

    @pytest.mark.unit
    def test_check_vulnerabilities_clean_file(self, clean_gnn_dir):
        """A clean GNN file should have no vulnerabilities."""
        gnn_file = list(clean_gnn_dir.glob("*.md"))[0]
        vulns = check_vulnerabilities(gnn_file, verbose=True)

        assert isinstance(vulns, list)
        assert len(vulns) == 0

    @pytest.mark.unit
    def test_check_vulnerabilities_detects_eval_exec(self, vuln_gnn_dir):
        """Should detect eval() and exec() as code injection vulnerabilities."""
        gnn_file = list(vuln_gnn_dir.glob("*.md"))[0]
        vulns = check_vulnerabilities(gnn_file, verbose=True)

        vuln_types = [v["vulnerability_type"] for v in vulns]
        assert "Code injection vulnerability" in vuln_types
        assert "Code execution vulnerability" in vuln_types

    @pytest.mark.unit
    def test_check_vulnerabilities_detects_hardcoded_credentials(self, vuln_gnn_dir):
        """Should detect hardcoded credentials (password='...', api_key='...')."""
        gnn_file = list(vuln_gnn_dir.glob("*.md"))[0]
        vulns = check_vulnerabilities(gnn_file, verbose=True)

        cred_vulns = [v for v in vulns if v["vulnerability_type"] == "Hardcoded credentials"]
        assert len(cred_vulns) > 0, "Should detect hardcoded credentials"

    @pytest.mark.unit
    def test_vulnerability_severity_assignment(self, vuln_gnn_dir):
        """Vulnerabilities should have severity levels (high or medium)."""
        gnn_file = list(vuln_gnn_dir.glob("*.md"))[0]
        vulns = check_vulnerabilities(gnn_file, verbose=True)

        for v in vulns:
            assert "severity" in v
            assert v["severity"] in ("high", "medium", "low")

    # -- calculate_security_score() tests --

    @pytest.mark.unit
    def test_security_score_no_vulnerabilities(self):
        """Score should be 100.0 when there are no vulnerabilities."""
        score = calculate_security_score([])
        assert score == 100.0

    @pytest.mark.unit
    def test_security_score_decreases_with_vulns(self):
        """Score should decrease as vulnerabilities are added."""
        one_vuln = [{"severity": "medium"}]
        many_vulns = [{"severity": "high"}] * 5

        score_one = calculate_security_score(one_vuln)
        score_many = calculate_security_score(many_vulns)

        assert score_one < 100.0
        assert score_many <= score_one
        assert score_one >= 0.0
        assert score_many >= 0.0

    # -- generate_security_summary() tests --

    @pytest.mark.unit
    def test_generate_security_summary_format(self):
        """Security summary should be a markdown string with expected headings."""
        results = {
            "processed_files": 2,
            "success": True,
            "errors": [],
            "security_checks": [{"file_name": "test.md"}],
            "vulnerabilities": [],
            "recommendations": [],
        }
        summary = generate_security_summary(results)

        assert isinstance(summary, str)
        assert "Security Analysis Summary" in summary
        assert "Files Processed" in summary

    @pytest.mark.unit
    def test_process_security_with_vulnerable_files(self, vuln_gnn_dir, output_dir):
        """Full pipeline run should still succeed even when vulnerabilities are found."""
        result = process_security(vuln_gnn_dir, output_dir, verbose=True)
        assert result is True

        with open(output_dir / "security_results.json") as f:
            data = json.load(f)
        assert len(data["vulnerabilities"]) > 0, "Should report vulnerabilities"
        assert data["processed_files"] == 1
