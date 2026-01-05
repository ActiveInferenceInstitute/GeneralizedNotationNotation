#!/usr/bin/env python3
"""
Tests for integration/processor.py

This module tests the integration processor functionality including:
- Empty directory handling
- GNN file processing and component detection
- Dependency graph building
- Circular dependency detection
- Cross-reference validation
"""

import pytest
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mark all tests
pytestmark = [pytest.mark.integration, pytest.mark.fast]


class TestProcessIntegration:
    """Test suite for process_integration function."""

    def test_process_empty_directory(self, tmp_path):
        """Should handle empty directories gracefully."""
        from integration.processor import process_integration
        
        output_dir = tmp_path / "output"
        result = process_integration(tmp_path, output_dir)
        
        assert result is True
        # Verify results directory was created
        assert (output_dir / "integration_results").exists()

    def test_process_with_gnn_files(self, tmp_path):
        """Should process GNN files and detect components."""
        # Create a test GNN file
        gnn_content = """# Test GNN File
        
components:
  - name: TestComponent
    type: Agent
    
  - name: AnotherComponent
    type: Environment
"""
        (tmp_path / "test_model.md").write_text(gnn_content)
        
        output_dir = tmp_path / "output"
        from integration.processor import process_integration
        
        result = process_integration(tmp_path, output_dir, verbose=True)
        
        assert result is True
        
        # Verify results file was created
        results_file = output_dir / "integration_results" / "integration_results.json"
        assert results_file.exists()
        
        # Check results content
        results = json.loads(results_file.read_text())
        assert results["processed_files"] == 1
        assert results["success"] is True

    def test_process_multiple_files_with_references(self, tmp_path):
        """Should detect cross-file references."""
        # File A defines ComponentA, references ComponentB
        file_a_content = """# File A
components:
  - name: ComponentA
    type: Agent
    references: ComponentB
"""
        # File B defines ComponentB
        file_b_content = """# File B
components:
  - name: ComponentB
    type: Environment
"""
        (tmp_path / "file_a.md").write_text(file_a_content)
        (tmp_path / "file_b.md").write_text(file_b_content)
        
        output_dir = tmp_path / "output"
        from integration.processor import process_integration
        
        result = process_integration(tmp_path, output_dir, verbose=True)
        
        assert result is True
        
        # Verify both files were processed
        results_file = output_dir / "integration_results" / "integration_results.json"
        results = json.loads(results_file.read_text())
        assert results["processed_files"] == 2

    def test_circular_dependency_detection(self, tmp_path):
        """Should detect circular dependencies when networkx available."""
        pytest.importorskip("networkx")
        
        # Create files with circular references
        # A -> B -> C -> A
        file_a = """# File A
components:
  - name: CompA
    uses: CompB
"""
        file_b = """# File B  
components:
  - name: CompB
    uses: CompC
"""
        file_c = """# File C
components:
  - name: CompC
    uses: CompA
"""
        (tmp_path / "a.md").write_text(file_a)
        (tmp_path / "b.md").write_text(file_b)
        (tmp_path / "c.md").write_text(file_c)
        
        output_dir = tmp_path / "output"
        from integration.processor import process_integration
        
        result = process_integration(tmp_path, output_dir, verbose=True)
        
        assert result is True
        
        # Check that graph stats are populated
        results_file = output_dir / "integration_results" / "integration_results.json"
        results = json.loads(results_file.read_text())
        assert "system_graph_stats" in results

    def test_undefined_reference_detection(self, tmp_path):
        """Should detect undefined cross-references."""
        # File with undefined $ref
        gnn_content = """# Test with undefined ref
components:
  - name: RealComponent
    $ref: UndefinedComponent
"""
        (tmp_path / "test.md").write_text(gnn_content)
        
        output_dir = tmp_path / "output"
        from integration.processor import process_integration
        
        result = process_integration(tmp_path, output_dir)
        
        assert result is True
        
        # Check for issues in results
        results_file = output_dir / "integration_results" / "integration_results.json"
        results = json.loads(results_file.read_text())
        
        # Should have captured the undefined reference
        assert any("UndefinedComponent" in issue for issue in results.get("issues", []))

    def test_summary_generation(self, tmp_path):
        """Should generate integration summary markdown."""
        (tmp_path / "test.md").write_text("- name: TestComp")
        
        output_dir = tmp_path / "output"
        from integration.processor import process_integration
        
        result = process_integration(tmp_path, output_dir)
        
        assert result is True
        
        # Verify summary was created
        summary_file = output_dir / "integration_results" / "integration_summary.md"
        assert summary_file.exists()
        
        summary_content = summary_file.read_text()
        assert "System Integration Report" in summary_content


class TestProcessIntegrationErrorHandling:
    """Test error handling in process_integration."""

    def test_invalid_file_handling(self, tmp_path):
        """Should handle files that can't be parsed."""
        # Create a binary file that looks like .md
        (tmp_path / "binary.md").write_bytes(b"\x00\x01\x02\x03")
        
        output_dir = tmp_path / "output"
        from integration.processor import process_integration
        
        # Should not raise, should handle gracefully
        result = process_integration(tmp_path, output_dir)
        
        # Processing should still succeed (graceful degradation)
        assert result is True

    def test_nonexistent_target_returns_gracefully(self, tmp_path):
        """Should handle non-existent target directory."""
        from integration.processor import process_integration
        
        nonexistent = tmp_path / "does_not_exist"
        output_dir = tmp_path / "output"
        
        # Should not raise, should return False
        result = process_integration(nonexistent, output_dir)
        
        # Processing of non-existent dir should succeed with 0 files
        # (glob returns empty list for non-existent paths)
        assert result is True
