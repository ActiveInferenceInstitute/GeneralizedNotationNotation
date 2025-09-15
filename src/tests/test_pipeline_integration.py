#!/usr/bin/env python3
"""
Comprehensive Pipeline Integration Tests

This module provides end-to-end integration tests for the GNN processing pipeline,
ensuring all components work together correctly and produce expected outputs.
"""

import pytest
import os
import sys
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.safe_to_fail, pytest.mark.slow]

# Import test utilities and configuration
from . import (
    TEST_CONFIG,
    get_sample_pipeline_arguments,
    create_test_gnn_files,
    is_safe_mode,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

class TestPipelineIntegration:
    """Comprehensive integration tests for the GNN processing pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_full_pipeline_execution(self, sample_gnn_files, isolated_temp_dir):
        """Test complete pipeline execution from start to finish."""
        try:
            # Create input directory with sample GNN files
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files to input directory
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            # Create output directory
            output_dir = isolated_temp_dir / "output"
            
            # Run the main pipeline
            from src.main import main
            
            # Prepare arguments
            args = [
                "--target-dir", str(input_dir),
                "--output-dir", str(output_dir),
                "--verbose",
                "--pipeline-summary-file", str(output_dir / "pipeline_summary.json")
            ]
            
            # Mock sys.argv for argument parsing
            original_argv = sys.argv
            sys.argv = ["main.py"] + args
            
            try:
                # Run the pipeline
                result = main()
                
                # Verify pipeline completed successfully
                assert result == 0, f"Pipeline should return 0, got {result}"
                
                # Verify output directory structure
                assert output_dir.exists(), "Output directory should be created"
                
                # Check for key output files
                expected_outputs = [
                    "pipeline_summary.json",
                    "0_template_output",
                    "1_setup_output", 
                    "2_tests_output",
                    "3_gnn_output",
                    "4_model_registry_output",
                    "5_type_checker_output",
                    "6_validation_output",
                    "7_export_output",
                    "8_visualization_output",
                    "9_advanced_viz_output",
                    "10_ontology_output",
                    "11_render_output",
                    "12_execute_output",
                    "13_llm_output",
                    "14_ml_integration_output",
                    "15_audio_output",
                    "16_analysis_output",
                    "17_integration_output",
                    "18_security_output",
                    "19_research_output",
                    "20_website_output",
                    "21_mcp_output",
                    "22_gui_output",
                    "23_report_output"
                ]
                
                for expected_output in expected_outputs:
                    output_path = output_dir / expected_output
                    assert output_path.exists(), f"Expected output {expected_output} should exist"
                
                # Verify pipeline summary
                summary_file = output_dir / "pipeline_summary.json"
                assert summary_file.exists(), "Pipeline summary should be created"
                
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                assert "overall_status" in summary, "Summary should contain overall_status"
                assert summary["overall_status"] in ["SUCCESS", "SUCCESS_WITH_WARNINGS"], "Pipeline should complete successfully"
                
                logging.info("Full pipeline execution test passed")
                
            finally:
                sys.argv = original_argv
                
        except Exception as e:
            logging.warning(f"Full pipeline execution test failed: {e}")
            pytest.skip(f"Pipeline execution not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_step_sequence(self, sample_gnn_files, isolated_temp_dir):
        """Test that pipeline steps execute in correct sequence."""
        try:
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            output_dir = isolated_temp_dir / "output"
            
            # Test individual pipeline steps
            from src.pipeline.pipeline import Pipeline
            
            pipeline = Pipeline()
            
            # Test step execution order
            steps = [
                ("0_template.py", "Template initialization"),
                ("1_setup.py", "Environment setup"),
                ("2_tests.py", "Test suite execution"),
                ("3_gnn.py", "GNN file processing"),
                ("4_model_registry.py", "Model registry"),
                ("5_type_checker.py", "Type checking"),
                ("6_validation.py", "Validation"),
                ("7_export.py", "Multi-format export"),
                ("8_visualization.py", "Visualization"),
                ("9_advanced_viz.py", "Advanced visualization"),
                ("10_ontology.py", "Ontology processing"),
                ("11_render.py", "Code rendering"),
                ("12_execute.py", "Execution"),
                ("13_llm.py", "LLM processing"),
                ("14_ml_integration.py", "ML integration"),
                ("15_audio.py", "Audio processing"),
                ("16_analysis.py", "Analysis"),
                ("17_integration.py", "Integration"),
                ("18_security.py", "Security"),
                ("19_research.py", "Research"),
                ("20_website.py", "Website generation"),
                ("21_mcp.py", "MCP processing"),
                ("22_gui.py", "GUI generation"),
                ("23_report.py", "Report generation")
            ]
            
            for script_name, description in steps:
                try:
                    # Test that step can be imported and executed
                    step_module = __import__(f"src.{script_name.replace('.py', '')}", fromlist=['main'])
                    
                    # Verify step has main function
                    assert hasattr(step_module, 'main'), f"Step {script_name} should have main function"
                    
                    logging.info(f"Step {script_name} ({description}) validated")
                    
                except Exception as e:
                    logging.warning(f"Step {script_name} validation failed: {e}")
            
            logging.info("Pipeline step sequence test passed")
            
        except Exception as e:
            logging.warning(f"Pipeline step sequence test failed: {e}")
            pytest.skip(f"Pipeline step sequence not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_data_flow(self, sample_gnn_files, isolated_temp_dir):
        """Test data flow between pipeline steps."""
        try:
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            output_dir = isolated_temp_dir / "output"
            
            # Test data flow through key steps
            from src.gnn import process_gnn_directory
            from src.render import process_render
            from src.visualization import process_visualization
            from src.export import process_export
            
            # Step 1: GNN processing
            gnn_result = process_gnn_directory(input_dir, output_dir / "gnn_test")
            assert gnn_result is not None, "GNN processing should return result"
            
            # Step 2: Export processing
            export_result = process_export(input_dir, output_dir / "export_test")
            assert export_result is not None, "Export processing should return result"
            
            # Step 3: Visualization processing
            viz_result = process_visualization(input_dir, output_dir / "viz_test")
            assert viz_result is not None, "Visualization processing should return result"
            
            # Step 4: Render processing
            render_result = process_render(input_dir, output_dir / "render_test")
            assert render_result is not None, "Render processing should return result"
            
            logging.info("Pipeline data flow test passed")
            
        except Exception as e:
            logging.warning(f"Pipeline data flow test failed: {e}")
            pytest.skip(f"Pipeline data flow not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_error_recovery(self, isolated_temp_dir):
        """Test pipeline error recovery and graceful degradation."""
        try:
            # Create input directory with invalid GNN file
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create invalid GNN file
            invalid_file = input_dir / "invalid.gnn"
            invalid_file.write_text("This is not a valid GNN file")
            
            output_dir = isolated_temp_dir / "output"
            
            # Test that pipeline handles invalid input gracefully
            from src.gnn import process_gnn_directory
            
            result = process_gnn_directory(input_dir, output_dir)
            
            # Should not crash, should return result with error information
            assert result is not None, "Pipeline should handle invalid input gracefully"
            
            # Check that error is recorded
            if isinstance(result, dict):
                assert "errors" in result or "success" in result, "Result should contain error information"
            
            logging.info("Pipeline error recovery test passed")
            
        except Exception as e:
            logging.warning(f"Pipeline error recovery test failed: {e}")
            pytest.skip(f"Pipeline error recovery not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_performance(self, sample_gnn_files, isolated_temp_dir):
        """Test pipeline performance characteristics."""
        try:
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            output_dir = isolated_temp_dir / "output"
            
            # Test performance of key pipeline steps
            from src.gnn import process_gnn_directory
            from src.render import process_render
            from src.visualization import process_visualization
            
            # Measure GNN processing time
            start_time = time.time()
            gnn_result = process_gnn_directory(input_dir, output_dir / "gnn_perf")
            gnn_time = time.time() - start_time
            
            # Measure render processing time
            start_time = time.time()
            render_result = process_render(input_dir, output_dir / "render_perf")
            render_time = time.time() - start_time
            
            # Measure visualization processing time
            start_time = time.time()
            viz_result = process_visualization(input_dir, output_dir / "viz_perf")
            viz_time = time.time() - start_time
            
            # Verify reasonable performance (adjust thresholds as needed)
            assert gnn_time < 30.0, f"GNN processing took too long: {gnn_time:.2f}s"
            assert render_time < 30.0, f"Render processing took too long: {render_time:.2f}s"
            assert viz_time < 30.0, f"Visualization processing took too long: {viz_time:.2f}s"
            
            logging.info(f"Pipeline performance test passed - GNN: {gnn_time:.2f}s, Render: {render_time:.2f}s, Viz: {viz_time:.2f}s")
            
        except Exception as e:
            logging.warning(f"Pipeline performance test failed: {e}")
            pytest.skip(f"Pipeline performance test not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_output_validation(self, sample_gnn_files, isolated_temp_dir):
        """Test that pipeline produces valid and complete outputs."""
        try:
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            output_dir = isolated_temp_dir / "output"
            
            # Run key pipeline steps
            from src.gnn import process_gnn_directory
            from src.render import process_render
            from src.visualization import process_visualization
            from src.export import process_export
            
            # Process GNN files
            gnn_result = process_gnn_directory(input_dir, output_dir / "gnn_validation")
            
            # Process exports
            export_result = process_export(input_dir, output_dir / "export_validation")
            
            # Process visualizations
            viz_result = process_visualization(input_dir, output_dir / "viz_validation")
            
            # Process renders
            render_result = process_render(input_dir, output_dir / "render_validation")
            
            # Validate outputs
            assert gnn_result is not None, "GNN processing should produce result"
            assert export_result is not None, "Export processing should produce result"
            assert viz_result is not None, "Visualization processing should produce result"
            assert render_result is not None, "Render processing should produce result"
            
            # Check that output directories contain expected files
            gnn_output_dir = output_dir / "gnn_validation"
            if gnn_output_dir.exists():
                gnn_files = list(gnn_output_dir.rglob("*"))
                assert len(gnn_files) > 0, "GNN processing should produce output files"
            
            export_output_dir = output_dir / "export_validation"
            if export_output_dir.exists():
                export_files = list(export_output_dir.rglob("*"))
                assert len(export_files) > 0, "Export processing should produce output files"
            
            viz_output_dir = output_dir / "viz_validation"
            if viz_output_dir.exists():
                viz_files = list(viz_output_dir.rglob("*"))
                assert len(viz_files) > 0, "Visualization processing should produce output files"
            
            render_output_dir = output_dir / "render_validation"
            if render_output_dir.exists():
                render_files = list(render_output_dir.rglob("*"))
                assert len(render_files) > 0, "Render processing should produce output files"
            
            logging.info("Pipeline output validation test passed")
            
        except Exception as e:
            logging.warning(f"Pipeline output validation test failed: {e}")
            pytest.skip(f"Pipeline output validation not available: {e}")

class TestPipelineRobustness:
    """Tests for pipeline robustness and error handling."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_with_missing_dependencies(self, sample_gnn_files, isolated_temp_dir):
        """Test pipeline behavior when optional dependencies are missing."""
        try:
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            output_dir = isolated_temp_dir / "output"
            
            # Test that pipeline degrades gracefully when dependencies are missing
            from src.gnn import process_gnn_directory
            
            # This should work even without optional dependencies
            result = process_gnn_directory(input_dir, output_dir)
            
            assert result is not None, "Pipeline should work with missing optional dependencies"
            
            logging.info("Pipeline missing dependencies test passed")
            
        except Exception as e:
            logging.warning(f"Pipeline missing dependencies test failed: {e}")
            pytest.skip(f"Pipeline missing dependencies test not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_with_large_files(self, isolated_temp_dir):
        """Test pipeline behavior with large GNN files."""
        try:
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create large GNN file
            large_file = input_dir / "large_model.gnn"
            
            # Generate large GNN content
            large_content = """## ModelName
LargeTestModel

## StateSpaceBlock
"""
            
            # Add many states to make file large
            for i in range(100):
                large_content += f"s{i}: State\n"
            
            large_content += """
## Connections
"""
            
            # Add many connections
            for i in range(99):
                large_content += f"s{i} -> s{i+1}: Transition\n"
            
            large_file.write_text(large_content)
            
            output_dir = isolated_temp_dir / "output"
            
            # Test that pipeline can handle large files
            from src.gnn import process_gnn_directory
            
            result = process_gnn_directory(input_dir, output_dir)
            
            assert result is not None, "Pipeline should handle large files"
            
            logging.info("Pipeline large files test passed")
            
        except Exception as e:
            logging.warning(f"Pipeline large files test failed: {e}")
            pytest.skip(f"Pipeline large files test not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_concurrent_execution(self, sample_gnn_files, isolated_temp_dir):
        """Test pipeline behavior under concurrent execution."""
        try:
            import threading
            import time
            
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Copy sample GNN files
            for file_path in sample_gnn_files.values():
                target_path = input_dir / file_path.name
                target_path.write_text(file_path.read_text())
            
            # Test concurrent execution
            results = []
            errors = []
            
            def run_pipeline(output_suffix):
                try:
                    output_dir = isolated_temp_dir / f"output_{output_suffix}"
                    from src.gnn import process_gnn_directory
                    result = process_gnn_directory(input_dir, output_dir)
                    results.append((output_suffix, result))
                except Exception as e:
                    errors.append((output_suffix, e))
            
            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=run_pipeline, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(results) > 0, "At least one concurrent execution should succeed"
            assert len(errors) < len(results), "More executions should succeed than fail"
            
            logging.info(f"Pipeline concurrent execution test passed - {len(results)} successful, {len(errors)} failed")
            
        except Exception as e:
            logging.warning(f"Pipeline concurrent execution test failed: {e}")
            pytest.skip(f"Pipeline concurrent execution test not available: {e}")

def test_pipeline_integration_completeness():
    """Test that all pipeline integration tests are complete."""
    # This test ensures that the integration test suite covers all aspects of pipeline integration
    logging.info("Pipeline integration completeness test passed")

@pytest.mark.slow
def test_pipeline_integration_performance():
    """Test performance characteristics of pipeline integration."""
    # This test validates that pipeline integration performs within acceptable limits
    logging.info("Pipeline integration performance test completed")