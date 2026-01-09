import pytest
import tempfile
from pathlib import Path
from security.processor import process_security, perform_security_check, check_vulnerabilities

class TestSecurityOverall:
    """Test suite for Security module."""

    @pytest.fixture
    def sample_gnn_file(self, safe_filesystem):
        """Create a sample GNN file for testing."""
        content = """
# Test Model

## Parameters
A = [[0.5, 0.5]]
"""
        return safe_filesystem.create_file("test_model.md", content)

    @pytest.fixture
    def vulnerable_gnn_file(self, safe_filesystem):
        """Create a GNN file with simulated vulnerabilities."""
        content = """
# Vulnerable Model
password = "super_secret_password"
import os
os.system("rm -rf /")
"""
        return safe_filesystem.create_file("vulnerable.md", content)

    def test_perform_security_check_clean(self, sample_gnn_file):
        """Test security check on a clean file."""
        result = perform_security_check(sample_gnn_file)
        assert result["file_name"] == sample_gnn_file.name
        # Should have high score
        assert result["security_score"] == 100.0
        assert len(result["sensitive_patterns"]) == 0

    def test_check_vulnerabilities_malicious(self, vulnerable_gnn_file):
        """Test detection of vulnerabilities."""
        vulns = check_vulnerabilities(vulnerable_gnn_file)
        
        # Check for specific vulnerabilities we noticed in the source patterns
        types = [v["vulnerability_type"] for v in vulns]
        
        # Expecting 'Hardcoded credentials' due to 'password ='
        assert any("Hardcoded credentials" in t for t in types)
        
        # Expecting 'OS command injection risk' due to 'import os'
        assert any("OS command injection risk" in t for t in types)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_process_security_integration(self, safe_filesystem, sample_gnn_file):
        """Test the full process_security flow."""
        target_dir = sample_gnn_file.parent
        output_dir = safe_filesystem.create_dir("security_output")
        
        success = process_security(target_dir, output_dir)
        
        assert success is True
        
        # Check output artifacts
        results_dir = output_dir
        assert results_dir.exists()
        assert (results_dir / "security_results.json").exists()
        assert (results_dir / "security_summary.md").exists()
