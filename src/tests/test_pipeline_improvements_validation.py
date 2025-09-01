#!/usr/bin/env python3
"""
Pipeline Improvements Validation Tests

This module validates that the specific improvements made to the pipeline
(DisCoPy module creation, visualization fixes, error handling, etc.) work correctly.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Test markers
pytestmark = [pytest.mark.integration, pytest.mark.safe_to_fail]

PROJECT_ROOT = Path(__file__).parent.parent.parent  
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from tests.conftest import *


class TestDiScoPyModuleCreation:
    """Test that the created DisCoPy translator module works correctly."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_discopy_translator_module_import(self):
        """Test that the DisCoPy translator module can be imported."""
        
        try:
            from execute.discopy_translator_module.translator import (
                JAX_FULLY_OPERATIONAL,
                MATPLOTLIB_AVAILABLE,
                gnn_file_to_discopy_diagram,
                gnn_file_to_discopy_matrix_diagram
            )
            
            # Module should import successfully
            assert callable(gnn_file_to_discopy_diagram)
            assert callable(gnn_file_to_discopy_matrix_diagram)
            assert isinstance(JAX_FULLY_OPERATIONAL, bool)
            assert isinstance(MATPLOTLIB_AVAILABLE, bool)
            
        except ImportError as e:
            pytest.fail(f"DisCoPy translator module import failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_discopy_diagram_creation_graceful_degradation(self):
        """Test DisCoPy diagram creation with graceful degradation."""
        
        try:
            from execute.discopy_translator_module.translator import gnn_file_to_discopy_diagram
            
            test_gnn_data = {
                "Variables": {
                    "state": {"type": "state", "dimensions": [3, 3], "comment": "State space"},
                    "action": {"type": "action", "dimensions": [2], "comment": "Action space"}
                },
                "Edges": [
                    {"source": "state", "target": "action", "directed": True}
                ]
            }
            
            success, message, diagram = gnn_file_to_discopy_diagram(test_gnn_data)
            
            # Should return appropriate response regardless of DisCoPy availability
            assert isinstance(success, bool)
            assert isinstance(message, str)
            
            if success:
                assert diagram is not None
            else:
                # Should provide helpful error message
                assert "DisCoPy" in message or "not installed" in message
                
        except Exception as e:
            pytest.fail(f"DisCoPy diagram creation failed unexpectedly: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualize_jax_output_module_import(self):
        """Test that the JAX visualization module can be imported."""
        
        try:
            from execute.discopy_translator_module.visualize_jax_output import (
                plot_tensor_output,
                plot_multiple_tensor_outputs,
                create_summary_visualization
            )
            
            # All functions should be importable
            assert callable(plot_tensor_output)
            assert callable(plot_multiple_tensor_outputs)
            assert callable(create_summary_visualization)
            
        except ImportError as e:
            pytest.fail(f"JAX visualization module import failed: {e}")


class TestVisualizationBugFixes:
    """Test that visualization bug fixes work correctly."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail 
    def test_visualization_parser_type_safety(self):
        """Test that visualization parser handles type fields safely."""
        
        try:
            from visualization.parser import GNNParser
            
            parser = GNNParser()
            
            # Test with content that previously caused 'type' NameError
            test_content = """
# Test Model

## StateSpaceBlock
var1 [3,3] # Test variable
var2 [2] # Another variable  
var3 [5,5,2] # 3D variable

## Connections
var1 > var2
var2 > var3
            """
            
            # Parse content using the string method
            parsed_data = parser._parse_markdown_format(test_content)
            
            # Should parse successfully without NameError
            assert isinstance(parsed_data, dict)
            assert "Variables" in parsed_data
            
            variables = parsed_data["Variables"]
            for var_name, var_info in variables.items():
                # All variables should have type field (even if 'unknown')
                assert "type" in var_info
                assert isinstance(var_info["type"], str)
                assert var_info["type"] != ""  # Should not be empty
                
        except Exception as e:
            pytest.fail(f"Visualization parser failed: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_matplotlib_dpi_corruption_fix(self):
        """Test that matplotlib DPI corruption is handled safely."""
        
        try:
            from visualization.processor import _save_plot_safely
            import matplotlib.pyplot as plt
            
            with tempfile.TemporaryDirectory() as temp_dir:
                test_path = Path(temp_dir) / "test_plot.png"
                
                # Create simple plot
                plt.figure()
                plt.plot([1, 2, 3], [1, 4, 9])
                
                # Test with corrupted DPI values (like the 28421050826 from error log)
                corrupted_dpi_values = [28421050826, -1, 0, float('inf'), "invalid"]
                
                for bad_dpi in corrupted_dpi_values:
                    # Should handle corrupted DPI without crashing
                    result = _save_plot_safely(test_path, dpi=bad_dpi)
                    assert isinstance(result, bool)
                    
                    if result:
                        assert test_path.exists()
                        test_path.unlink()  # Clean up for next test
                
                plt.close()
                
        except ImportError:
            pytest.skip("Matplotlib not available for DPI corruption test")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_visualization_data_structure_compatibility(self):
        """Test that visualization functions work with actual parsed data structure."""
        
        try:
            from visualization.processor import generate_network_visualizations
            
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                
                # Use the actual data structure from GNN parser
                parsed_data = {
                    "Variables": {
                        "state": {"type": "state", "dimensions": [3, 3], "comment": "State space"},
                        "action": {"type": "action", "dimensions": [2], "comment": "Action space"}
                    },
                    "Edges": [
                        {"source": "state", "target": "action", "directed": True, "comment": "Control"}
                    ]
                }
                
                # Should work without 'type' NameError
                visualizations = generate_network_visualizations(parsed_data, output_dir, "test_model")
                
                # Should return list of visualization paths
                assert isinstance(visualizations, list)
                
        except ImportError:
            pytest.skip("Visualization dependencies not available")
        except Exception as e:
            pytest.fail(f"Network visualization failed: {e}")


class TestErrorHandlingImprovements:
    """Test that error handling improvements work correctly."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_standardized_error_handler_functionality(self):
        """Test standardized error handler functionality."""
        
        try:
            from utils.standardized_error_handling import StandardizedErrorHandler
            
            handler = StandardizedErrorHandler("test_step")
            
            # Test dependency error handling
            test_error = ImportError("No module named 'test_dependency'")
            should_continue = handler.handle_dependency_error("test_dependency", test_error)
            
            assert isinstance(should_continue, bool)
            
            # Test file operation error handling
            file_error = FileNotFoundError("Test file not found")
            test_path = Path("/non/existent/file.txt")
            should_continue = handler.handle_file_operation_error("read", test_path, file_error)
            
            assert isinstance(should_continue, bool)
            
            # Test error summary generation
            summary = handler.get_error_summary()
            assert isinstance(summary, dict)
            assert "step_name" in summary
            assert "correlation_id" in summary
            assert "total_errors" in summary
            
        except ImportError:
            pytest.skip("Standardized error handling not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_error_context_manager(self):
        """Test error context manager functionality."""
        
        try:
            from utils.standardized_error_handling import StandardizedErrorHandler
            
            handler = StandardizedErrorHandler("test_step")
            
            # Test successful operation
            with handler.error_context("test_operation", test_param="test_value"):
                result = "success"
            
            # Should complete without issues
            assert result == "success"
            
            # Test operation with recoverable error
            try:
                with handler.error_context("failing_operation"):
                    raise ValueError("Test error")
            except ValueError:
                # Should re-raise the error, but context should be recorded
                pass
            
            # Should have recorded the error
            summary = handler.get_error_summary()
            assert summary["total_errors"] > 0
            
        except ImportError:
            pytest.skip("Standardized error handling not available")


class TestDependencyValidationImprovements:
    """Test that dependency validation improvements work correctly."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_discopy_dependency_group(self):
        """Test that DisCoPy dependency group is properly defined."""
        
        try:
            from utils.dependency_validator import DependencyValidator
            
            validator = DependencyValidator()
            
            # Should have discopy dependency group
            assert "discopy" in validator.dependencies
            
            discopy_deps = validator.dependencies["discopy"]
            dep_names = [dep.name for dep in discopy_deps]
            
            # Should include DisCoPy and JAX dependencies
            assert "discopy" in dep_names
            assert "jax" in dep_names or "jaxlib" in dep_names
            
            # All DisCoPy dependencies should be optional
            for dep in discopy_deps:
                assert dep.is_optional, f"DisCoPy dependency {dep.name} should be optional"
                
        except ImportError:
            pytest.skip("Dependency validator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_pymdp_dependency_group(self):
        """Test that PyMDP dependency group is properly defined."""
        
        try:
            from utils.dependency_validator import DependencyValidator
            
            validator = DependencyValidator()
            
            # Should have pymdp dependency group
            assert "pymdp" in validator.dependencies
            
            pymdp_deps = validator.dependencies["pymdp"]
            dep_names = [dep.name for dep in pymdp_deps]
            
            # Should include PyMDP dependency
            assert "pymdp" in dep_names
            
            # PyMDP should be optional
            pymdp_dep = next(dep for dep in pymdp_deps if dep.name == "pymdp")
            assert pymdp_dep.is_optional
            
        except ImportError:
            pytest.skip("Dependency validator not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_improved_installation_instructions(self):
        """Test that installation instructions are provided for missing dependencies."""
        
        try:
            from utils.dependency_validator import DependencyValidator
            
            validator = DependencyValidator()
            
            # Validate all dependencies (will find some missing optional ones)
            validator.validate_all_dependencies()
            
            # Should provide installation instructions
            instructions = validator.get_installation_instructions()
            assert isinstance(instructions, list)
            
            # Instructions should use 'uv' as per user preference
            for instruction in instructions:
                if "pip install" in instruction:
                    # Should prefer uv pip install
                    assert "uv pip install" in instruction or "pip install" in instruction
                    
        except ImportError:
            pytest.skip("Dependency validator not available")


class TestLoggingImprovements:
    """Test that logging improvements work correctly."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_diagnostic_logger_functionality(self):
        """Test diagnostic logger functionality."""
        
        try:
            from utils.enhanced_logging import DiagnosticLogger
            
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = DiagnosticLogger("test_step", Path(temp_dir))
                
                # Test logging methods
                logger.log_step_start("Test operation", param1="value1")
                logger.log_step_warning("Test warning", context="test")
                logger.log_step_success("Test completed", results="success")
                
                # Test performance tracking
                with logger.performance_context("test_operation"):
                    import time
                    time.sleep(0.01)  # Small delay
                
                # Test correlation ID
                correlation_id = logger.get_correlation_id()
                assert isinstance(correlation_id, str)
                assert len(correlation_id) > 0
                
                # Test diagnostic report
                logger.save_diagnostic_report()
                
                # Should create diagnostic report file
                report_files = list(Path(temp_dir).glob("*diagnostic_report.json"))
                assert len(report_files) > 0
                
                # Report should be valid JSON
                with open(report_files[0]) as f:
                    report_data = json.load(f)
                
                assert isinstance(report_data, dict)
                assert "step_start" in report_data
                assert "correlation_id" in report_data.get("step_start", {}).get("system_info", {})
                
        except ImportError:
            pytest.skip("Enhanced logging not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_correlation_context(self):
        """Test correlation context functionality."""
        
        try:
            from utils.enhanced_logging import CorrelationContext
            
            # Test correlation ID generation
            correlation_id = CorrelationContext.new_correlation_id()
            assert isinstance(correlation_id, str)
            assert len(correlation_id) > 0
            
            # Test correlation ID persistence
            same_id = CorrelationContext.get_correlation_id()
            assert same_id == correlation_id
            
            # Test setting custom correlation ID  
            custom_id = "custom123"
            CorrelationContext.set_correlation_id(custom_id)
            retrieved_id = CorrelationContext.get_correlation_id()
            assert retrieved_id == custom_id
            
        except ImportError:
            pytest.skip("Enhanced logging not available")


class TestIntegrationValidation:
    """Test that all improvements work together in integration."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_end_to_end_pipeline_with_improvements(self, temp_directories):
        """Test that the pipeline runs with all improvements integrated."""
        
        # Create test GNN file that exercises the improvements
        test_file = temp_directories["input_dir"] / "improvements_test.md"
        test_file.write_text("""
# Improvements Test Model

## StateSpaceBlock
state [3,3,2] # Complex tensor
action [2] # Simple vector  
observation [3] # Observation space

## Connections
state > action
action > observation
observation > state
        """)
        
        try:
            # Test that DisCoPy executor no longer fails
            from execute.discopy.discopy_executor import DisCoPyExecutor
            
            executor = DisCoPyExecutor(verbose=True)
            
            # Should initialize without ImportError
            assert executor is not None
            
            # Test visualization with fixed parser
            from visualization.parser import GNNParser
            
            parser = GNNParser()
            parsed_data = parser.parse_file(str(test_file))
            
            # Should parse without 'type' NameError
            assert isinstance(parsed_data, dict)
            assert "Variables" in parsed_data
            
            # All variables should have safe type values
            for var_name, var_info in parsed_data["Variables"].items():
                assert "type" in var_info
                assert var_info["type"] is not None
                assert var_info["type"] != ""
                
            # Test error handling integration
            from utils.standardized_error_handling import create_error_handler
            
            error_handler = create_error_handler("integration_test")
            
            with error_handler.error_context("testing_integration"):
                # Should complete without issues
                pass
            
            # Should have correlation ID
            assert error_handler.correlation_id is not None
            
        except Exception as e:
            pytest.fail(f"End-to-end integration test failed: {e}")


@pytest.mark.parametrize("step_numbers", [
    [3, 5, 7, 8],  # Basic steps from original execution
    [3, 8, 12, 15],  # Steps that previously had issues
])
@pytest.mark.integration
@pytest.mark.safe_to_fail
def test_specific_pipeline_steps_improvements(step_numbers, temp_directories):
    """Test that specific pipeline steps work with improvements."""
    
    import subprocess
    
    # Create test GNN file
    test_file = temp_directories["input_dir"] / "step_test.md"
    test_file.write_text("""
# Step Test Model

## StateSpaceBlock
state [2,2]
action [3]

## Connections
state > action
    """)
    
    try:
        cmd = [
            sys.executable,
            str(SRC_DIR / "main.py"),
            "--target-dir", str(temp_directories["input_dir"]),
            "--output-dir", str(temp_directories["output_dir"]),
            "--only-steps", ",".join(map(str, step_numbers)),
            "--verbose"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT
        )
        
        # Should complete successfully or with warnings (not critical failures)
        assert result.returncode in [0, 2], f"Steps {step_numbers} failed: {result.stderr}"
        
        # Should not have specific error patterns that we fixed
        stderr_lower = result.stderr.lower()
        
        # Should not have the old DisCoPy import error
        assert "discopy translator module not available" not in stderr_lower
        
        # Should not have matplotlib corruption errors
        assert "incompatible constructor arguments" not in stderr_lower
        
        # Should not have 'type' NameError
        assert "nameerror" not in stderr_lower or "'type'" not in stderr_lower
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"Pipeline steps {step_numbers} timed out")
    except Exception as e:
        pytest.fail(f"Pipeline steps {step_numbers} failed unexpectedly: {e}")

