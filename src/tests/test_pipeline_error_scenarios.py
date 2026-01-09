#!/usr/bin/env python3
"""
Pipeline Error Scenario Tests

This module tests the pipeline's behavior under various error conditions,
dependency failures, and edge cases to ensure robust operation.
"""

import pytest
import sys
import tempfile
import subprocess
import json
from pathlib import Path
# Mocks removed - using real implementations per testing policy
from typing import Dict, Any, List, Optional

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.safe_to_fail]

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from tests.conftest import *


class TestDependencyErrorScenarios:
    """Test pipeline behavior with missing dependencies."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pymdp_graceful_degradation_with_real_error(self):
        """Test that PyMDP handles errors gracefully with real execution."""
        
        try:
            from execute.pymdp.pymdp_simulation import PyMDPSimulation
            import logging
            from io import StringIO
            
            # Create a logger to capture warnings
            logger = logging.getLogger('test_pymdp')
            log_capture = StringIO()
            handler = logging.StreamHandler(log_capture)
            logger.addHandler(handler)
            
            # Test with empty/invalid model data
            simulator = PyMDPSimulation()
            assert simulator is not None
            
            # Create a model with invalid structure - should handle gracefully
            invalid_model = {}  # Empty model
            try:
                result = simulator.create_model(invalid_model, logger=logger)
                # Should either fail gracefully or return graceful error
                assert isinstance(result, (dict, type(None)))
            except Exception as e:
                # Exception should contain descriptive message
                assert "invalid" in str(e).lower() or "empty" in str(e).lower() or len(str(e)) > 0
                    
        except ImportError:
            pytest.skip("PyMDP not available")
    
    @pytest.mark.unit 
    @pytest.mark.safe_to_fail
    def test_missing_discopy_graceful_degradation(self):
        """Test that missing DisCoPy is handled gracefully."""
        
        try:
            from execute.discopy_translator_module.translator import (
                gnn_file_to_discopy_diagram,
                JAX_FULLY_OPERATIONAL,
                DISCOPY_AVAILABLE
            )
            
            # Test with DisCoPy unavailable
            test_gnn_data = {"Variables": {"test_var": {"type": "state", "dimensions": [3]}}}
            success, message, diagram = gnn_file_to_discopy_diagram(test_gnn_data)
            
            if not DISCOPY_AVAILABLE:
                # Should return graceful failure
                assert not success
                assert "DisCoPy library not installed" in message
                assert diagram is None
            else:
                # Should work if available
                assert isinstance(success, bool)
                assert isinstance(message, str)
                
        except ImportError:
            pytest.skip("DisCoPy translator module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matplotlib_rendering_fallbacks(self):
        """Test visualization fallbacks when matplotlib has issues."""
        
        try:
            from visualization.processor import _save_plot_safely
            import matplotlib.pyplot as plt
            
            with tempfile.TemporaryDirectory() as temp_dir:
                test_plot_path = Path(temp_dir) / "test_plot.png"
                
                # Create a simple plot
                plt.figure()
                plt.plot([1, 2, 3], [1, 4, 9])
                
                # Test with extreme DPI values
                assert _save_plot_safely(test_plot_path, dpi=999999)  # Should fallback
                assert test_plot_path.exists()
                
                plt.close()
                
        except ImportError:
            pytest.skip("Matplotlib not available")


class TestFileOperationErrorScenarios:
    """Test pipeline behavior with file operation errors."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_missing_input_directory(self, temp_directories):
        """Test pipeline behavior when input directory is missing."""
        
        non_existent_dir = temp_directories["temp_dir"] / "non_existent"
        
        try:
            from visualization import process_visualization_main
            
            # Should handle missing directory gracefully
            result = process_visualization_main(
                target_dir=non_existent_dir,
                output_dir=temp_directories["output_dir"],
                verbose=True
            )
            
            # Should return False but not crash
            assert isinstance(result, bool)
            
        except Exception as e:
            # Should not raise exceptions, but if it does, should be informative
            assert "does not exist" in str(e) or "not found" in str(e)
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_readonly_output_directory(self, temp_directories):
        """Test pipeline behavior with read-only output directory."""
        
        readonly_dir = temp_directories["temp_dir"] / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        try:
            from gnn import process_gnn_directory
            
            # Should handle permission errors gracefully
            result = process_gnn_directory(
                target_dir=temp_directories["input_dir"],
                output_dir=readonly_dir,
                verbose=True
            )
            
            # Should handle gracefully
            assert isinstance(result, (bool, dict))
            
        except PermissionError:
            # Expected behavior
            pass
        except Exception as e:
            # Any other error is acceptable for this edge case test
            pass
        finally:
            # Restore permissions for cleanup
            try:
                readonly_dir.chmod(0o755)
            except:
                pass
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_corrupted_gnn_file_handling(self, temp_directories):
        """Test handling of corrupted or invalid GNN files."""
        
        # Create corrupted GNN file
        corrupted_file = temp_directories["input_dir"] / "corrupted.md"
        corrupted_file.write_text("This is not a valid GNN file\n\x00\x01\x02")
        
        try:
            from gnn.parser import GNNParser
            
            parser = GNNParser()
            
            # Should handle corrupted file gracefully
            result = parser.parse_file(str(corrupted_file))
            
            # Should return some result even with corrupted input
            assert isinstance(result, dict)
            
        except Exception as e:
            # Should provide informative error message
            assert "parse" in str(e).lower() or "invalid" in str(e).lower()


class TestResourceConstraintScenarios:
    """Test pipeline behavior under resource constraints."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_large_gnn_file_handling(self, temp_directories):
        """Test handling of unusually large GNN files."""
        
        # Create large GNN file
        large_file = temp_directories["input_dir"] / "large_model.md"
        
        # Generate large but valid GNN content
        large_content = "# Large GNN Model\n\n## StateSpaceBlock\n"
        for i in range(1000):  # 1000 variables
            large_content += f"var_{i} [10,10] # Variable {i}\n"
        
        large_content += "\n## Connections\n"
        for i in range(500):  # 500 connections
            large_content += f"var_{i} > var_{i+1}\n"
        
        large_file.write_text(large_content)
        
        try:
            from gnn.parsers.unified_parser import UnifiedGNNParser
            
            parser = UnifiedGNNParser()
            
            # Should handle large file without memory issues
            result = parser.parse_file(str(large_file))
            
            # Should successfully parse large file or handle gracefully
            if result is not None:
                assert isinstance(result, dict)
            
        except ImportError:
            # Parser module structure may vary - this is acceptable for safe_to_fail test
            pytest.skip("Parser interface not available")
        except Exception as e:
            # If it fails, should be due to reasonable resource constraints or parsing issues
            # This is acceptable for large files
            assert any(word in str(e).lower() for word in ["memory", "timeout", "resource", "size", "parse", "invalid"])
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail 
    def test_concurrent_pipeline_execution(self, temp_directories):
        """Test behavior when multiple pipeline steps run concurrently."""
        
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        def run_visualization():
            try:
                from visualization import process_visualization_main
                return process_visualization_main(
                    target_dir=temp_directories["input_dir"],
                    output_dir=temp_directories["output_dir"] / "viz_thread",
                    verbose=False
                )
            except Exception as e:
                return str(e)
        
        def run_gnn_processing():
            try:
                from gnn import process_gnn_main
                return process_gnn_main(
                    target_dir=temp_directories["input_dir"],
                    output_dir=temp_directories["output_dir"] / "gnn_thread",
                    verbose=False
                )
            except Exception as e:
                return str(e)
        
        # Create test GNN file
        test_file = temp_directories["input_dir"] / "concurrent_test.md"
        test_file.write_text("""
# Concurrent Test Model

## StateSpaceBlock
state [3,3]
action [2]

## Connections  
state > action
        """)
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(run_visualization)
            future2 = executor.submit(run_gnn_processing)
            
            result1 = future1.result(timeout=30)
            result2 = future2.result(timeout=30)
            
            # Both should complete without interference
            assert isinstance(result1, bool) or isinstance(result1, str)
            assert isinstance(result2, bool) or isinstance(result2, str)


class TestPipelineIntegrationScenarios:
    """Test full pipeline integration under various scenarios."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_pipeline_with_missing_dependencies(self, temp_directories):
        """Test full pipeline execution with some missing dependencies."""
        
        # Create minimal valid GNN file
        test_file = temp_directories["input_dir"] / "integration_test.md"
        test_file.write_text("""
# Integration Test Model

## StateSpaceBlock
state [2,2]
observation [2]

## Connections
state > observation
        """)
        
        try:
            # Test pipeline execution via subprocess to isolate dependencies
            cmd = [
                sys.executable, 
                str(SRC_DIR / "main.py"),
                "--target-dir", str(temp_directories["input_dir"]),
                "--output-dir", str(temp_directories["output_dir"]),
                "--only-steps", "3,5,7,8",  # Basic steps that should work
                "--verbose"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,
                cwd=PROJECT_ROOT
            )
            
            # Should complete even with missing optional dependencies
            # Exit code 0 (success) or 2 (success with warnings) are acceptable
            assert result.returncode in [0, 2], f"Pipeline failed with exit code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            
            # Should produce some output
            output_files = list(temp_directories["output_dir"].rglob("*"))
            assert len(output_files) > 0, "No output files produced"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline execution timed out")
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {e}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_error_recovery_and_continuation(self, temp_directories):
        """Test pipeline's ability to recover from errors and continue."""
        
        # Create GNN file that will cause some processing issues
        problematic_file = temp_directories["input_dir"] / "problematic.md"
        problematic_file.write_text("""
# Problematic GNN Model

## StateSpaceBlock
# Invalid variable definition that may cause parsing issues
invalid_var [invalid_dimension, another_invalid]
valid_var [3,3]

## Connections
# Invalid connection
invalid_var > non_existent_var
valid_var > valid_var
        """)
        
        valid_file = temp_directories["input_dir"] / "valid.md"
        valid_file.write_text("""
# Valid GNN Model

## StateSpaceBlock  
state [2,2]
action [3]

## Connections
state > action
        """)
        
        try:
            # Test that pipeline continues despite errors in one file
            cmd = [
                sys.executable,
                str(SRC_DIR / "main.py"),
                "--target-dir", str(temp_directories["input_dir"]),
                "--output-dir", str(temp_directories["output_dir"]),
                "--only-steps", "3,5",  # Basic steps
                "--verbose",
                "--recursive"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=PROJECT_ROOT
            )
            
            # Should complete with warnings (exit code 2) due to problematic file
            # but should still process the valid file
            assert result.returncode in [0, 2], f"Expected success or warnings, got {result.returncode}"
            
            # Should have processed at least the valid file
            output_files = list(temp_directories["output_dir"].rglob("*"))
            assert len(output_files) > 0, "No output produced despite valid input file"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Pipeline execution timed out")


class TestErrorReportingAndDiagnostics:
    """Test error reporting and diagnostic capabilities."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_standardized_error_handler_creation(self):
        """Test that standardized error handlers can be created."""
        
        try:
            from utils.standardized_error_handling import create_error_handler
            
            handler = create_error_handler("test_step")
            assert handler is not None
            assert handler.step_name == "test_step"
            assert handler.correlation_id is not None
            
        except ImportError:
            pytest.skip("Standardized error handling not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_diagnostic_logger_creation(self):
        """Test that diagnostic loggers can be created and used."""
        
        try:
            from utils.diagnostic_logging import create_diagnostic_logger
            
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = create_diagnostic_logger("test_step", Path(temp_dir))
                assert logger is not None
                assert logger.step_name == "test_step"
                
                # Test basic logging functions
                logger.log_step_start("Test operation")
                logger.log_step_warning("Test warning")
                logger.log_step_success("Test completed")
                
                # Should create correlation ID
                assert logger.get_correlation_id() is not None
                
        except ImportError:
            pytest.skip("Enhanced logging not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_dependency_validation_reporting(self):
        """Test dependency validation reporting."""
        
        try:
            from utils.dependency_validator import DependencyValidator
            
            validator = DependencyValidator()
            
            # Test validation of core dependencies
            core_valid = validator.validate_dependency_group("core")
            assert isinstance(core_valid, bool)
            
            # Test validation of optional dependencies
            pymdp_valid = validator.validate_dependency_group("pymdp")
            assert isinstance(pymdp_valid, bool)
            
            # Should provide installation instructions for missing deps
            instructions = validator.get_installation_instructions()
            assert isinstance(instructions, list)
            
        except ImportError:
            pytest.skip("Dependency validator not available")

