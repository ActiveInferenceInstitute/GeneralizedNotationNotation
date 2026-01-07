import pytest
import json
from analysis.processor import process_analysis

class TestAnalysisOverall:
    """Test suite for Analysis module."""

    @pytest.fixture
    def sample_gnn_for_analysis(self, safe_filesystem):
        """Create a sample GNN file to analyze."""
        content = """
# Analysis Target

## StateSpaceBlock
s[10, type=float]

## Connections
s->s

## Time
Dynamic
"""
        return safe_filesystem.create_file("model_analysis.md", content)

    def test_process_analysis_flow(self, safe_filesystem, sample_gnn_for_analysis):
        """Test the analysis processing workflow."""
        target_dir = sample_gnn_for_analysis.parent
        output_dir = safe_filesystem.create_dir("analysis_output")
        
        # Need to ensure submodules of analysis don't crash.
        # analysis/processor imports from analyzer.py.
        # analyzer.py imports numpy etc.
        
        try:
            success = process_analysis(target_dir, output_dir, verbose=True)
            assert success is True
            
            # Check results
            results_dir = output_dir / "analysis_results"
            assert results_dir.exists()
            assert (results_dir / "analysis_results.json").exists()
            assert (results_dir / "analysis_summary.md").exists()
            
            with open(results_dir / "analysis_results.json", 'r') as f:
                data = json.load(f)
            
            assert data["processed_files"] == 1
            assert len(data["statistical_analysis"]) == 1
            
        except ImportError:
            pytest.skip("Skipping analysis test due to missing dependencies (numpy/matplotlib)")
        except Exception as e:
            pytest.fail(f"Analysis processing failed: {e}")

    def test_process_analysis_no_files(self, safe_filesystem):
        """Test behavior with no files."""
        empty_dir = safe_filesystem.create_dir("empty")
        output_dir = safe_filesystem.create_dir("output")
        
        success = process_analysis(empty_dir, output_dir)
        
        # Processor returns False if no files found (based on source: results["success"] = False)
        assert success is False
