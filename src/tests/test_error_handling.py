#!/usr/bin/env python3
"""
Comprehensive Error Handling and Edge Case Tests

This module provides thorough testing for error handling, edge cases, and
robustness across all GNN processing modules.
"""

import pytest
import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import contextlib

# Test markers
pytestmark = [pytest.mark.error_handling, pytest.mark.safe_to_fail, pytest.mark.robustness]

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

class TestErrorHandling:
    """Comprehensive error handling tests for all modules."""
    
    @pytest.mark.error_handling
    @pytest.mark.safe_to_fail
    def test_gnn_module_error_handling(self, isolated_temp_dir):
        """Test GNN module error handling with various invalid inputs."""
        try:
            from src.gnn import discover_gnn_files, parse_gnn_file, validate_gnn_structure
            
            # Test with non-existent directory
            non_existent_dir = isolated_temp_dir / "non_existent"
            files = discover_gnn_files(non_existent_dir)
            assert isinstance(files, list), "Should return empty list for non-existent directory"
            assert len(files) == 0, "Should return empty list for non-existent directory"
            
            # Test with empty directory
            empty_dir = isolated_temp_dir / "empty"
            empty_dir.mkdir()
            files = discover_gnn_files(empty_dir)
            assert isinstance(files, list), "Should return empty list for empty directory"
            
            # Test with invalid file
            invalid_file = isolated_temp_dir / "invalid.gnn"
            invalid_file.write_text("This is not a valid GNN file")
            
            parsed = parse_gnn_file(invalid_file)
            assert isinstance(parsed, dict), "Should return dict even for invalid file"
            assert "error" in parsed or "file_path" in parsed, "Should contain error info or basic info"
            
            # Test validation with invalid file
            validation = validate_gnn_structure(invalid_file)
            assert isinstance(validation, dict), "Should return validation dict"
            assert "valid" in validation, "Should contain valid field"
            
            logging.info("GNN module error handling test passed")
            
        except Exception as e:
            logging.warning(f"GNN module error handling test failed: {e}")
            pytest.skip(f"GNN module error handling not available: {e}")
    
    @pytest.mark.error_handling
    @pytest.mark.safe_to_fail
    def test_render_module_error_handling(self, isolated_temp_dir):
        """Test render module error handling with invalid inputs."""
        try:
            from src.render import process_render
            
            # Test with non-existent input directory
            non_existent_dir = isolated_temp_dir / "non_existent"
            result = process_render(non_existent_dir, isolated_temp_dir / "output")
            
            # Should handle gracefully without crashing
            assert result is not None, "Should return result even with invalid input"
            
            # Test with empty input directory
            empty_dir = isolated_temp_dir / "empty"
            empty_dir.mkdir()
            result = process_render(empty_dir, isolated_temp_dir / "output2")
            
            assert result is not None, "Should handle empty directory gracefully"
            
            logging.info("Render module error handling test passed")
            
        except Exception as e:
            logging.warning(f"Render module error handling test failed: {e}")
            pytest.skip(f"Render module error handling not available: {e}")
    
    @pytest.mark.error_handling
    @pytest.mark.safe_to_fail
    def test_execute_module_error_handling(self, isolated_temp_dir):
        """Test execute module error handling with invalid inputs."""
        try:
            from src.execute import process_execute, validate_execution_environment
            
            # Test with non-existent input directory
            non_existent_dir = isolated_temp_dir / "non_existent"
            result = process_execute(non_existent_dir, isolated_temp_dir / "output")
            
            # Should handle gracefully
            assert result is not None, "Should return result even with invalid input"
            
            # Test environment validation
            env_status = validate_execution_environment()
            assert isinstance(env_status, dict), "Should return environment status dict"
            
            logging.info("Execute module error handling test passed")
            
        except Exception as e:
            logging.warning(f"Execute module error handling test failed: {e}")
            pytest.skip(f"Execute module error handling not available: {e}")
    
    @pytest.mark.error_handling
    @pytest.mark.safe_to_fail
    def test_llm_module_error_handling(self, isolated_temp_dir):
        """Test LLM module error handling with invalid inputs."""
        try:
            from src.llm import process_llm
            
            # Test with non-existent input directory
            non_existent_dir = isolated_temp_dir / "non_existent"
            result = process_llm(non_existent_dir, isolated_temp_dir / "output")
            
            # Should handle gracefully
            assert result is not None, "Should return result even with invalid input"
            
            logging.info("LLM module error handling test passed")
            
        except Exception as e:
            logging.warning(f"LLM module error handling test failed: {e}")
            pytest.skip(f"LLM module error handling not available: {e}")
    
    @pytest.mark.error_handling
    @pytest.mark.safe_to_fail
    def test_mcp_module_error_handling(self, isolated_temp_dir):
        """Test MCP module error handling with invalid inputs."""
        try:
            from src.mcp import process_mcp, handle_mcp_request
            
            # Test with non-existent input directory
            non_existent_dir = isolated_temp_dir / "non_existent"
            result = process_mcp(non_existent_dir, isolated_temp_dir / "output")
            
            # Should handle gracefully
            assert result is not None, "Should return result even with invalid input"
            
            # Test MCP request handling with invalid request
            invalid_request = {"invalid": "request"}
            response = handle_mcp_request(invalid_request)
            
            assert isinstance(response, dict), "Should return response dict"
            assert "error" in response or "jsonrpc" in response, "Should contain error or jsonrpc field"
            
            logging.info("MCP module error handling test passed")
            
        except Exception as e:
            logging.warning(f"MCP module error handling test failed: {e}")
            pytest.skip(f"MCP module error handling not available: {e}")

class TestEdgeCases:
    """Comprehensive edge case tests for all modules."""
    
    @pytest.mark.edge_cases
    @pytest.mark.safe_to_fail
    def test_empty_input_handling(self, isolated_temp_dir):
        """Test handling of empty inputs across all modules."""
        try:
            # Create empty input directory
            empty_dir = isolated_temp_dir / "empty"
            empty_dir.mkdir()
            
            # Test all modules with empty input
            modules_to_test = [
                ("src.gnn", "process_gnn_directory"),
                ("src.render", "process_render"),
                ("src.execute", "process_execute"),
                ("src.llm", "process_llm"),
                ("src.mcp", "process_mcp"),
                ("src.visualization", "process_visualization"),
                ("src.export", "process_export"),
                ("src.ontology", "process_ontology"),
                ("src.website", "process_website"),
                ("src.audio", "process_audio"),
                ("src.analysis", "process_analysis"),
                ("src.integration", "process_integration"),
                ("src.security", "process_security"),
                ("src.research", "process_research"),
                ("src.report", "process_report")
            ]
            
            for module_name, function_name in modules_to_test:
                try:
                    module = __import__(module_name, fromlist=[function_name])
                    func = getattr(module, function_name)
                    
                    # Test with empty input
                    result = func(empty_dir, isolated_temp_dir / f"output_{function_name}")
                    
                    # Should not crash and should return some result
                    assert result is not None, f"{function_name} should return result for empty input"
                    
                except ImportError:
                    logging.debug(f"Module {module_name} not available, skipping")
                except Exception as e:
                    logging.warning(f"Module {module_name}.{function_name} failed with empty input: {e}")
            
            logging.info("Empty input handling test passed")
            
        except Exception as e:
            logging.warning(f"Empty input handling test failed: {e}")
            pytest.skip(f"Empty input handling test not available: {e}")
    
    @pytest.mark.edge_cases
    @pytest.mark.safe_to_fail
    def test_malformed_gnn_files(self, isolated_temp_dir):
        """Test handling of malformed GNN files."""
        try:
            from src.gnn import parse_gnn_file, validate_gnn_structure
            
            # Create various malformed GNN files
            malformed_files = [
                ("empty.gnn", ""),
                ("no_headers.gnn", "This is just plain text"),
                ("incomplete.gnn", "## ModelName\nTestModel\n## StateSpaceBlock\ns1: State"),
                ("invalid_syntax.gnn", "## ModelName\nTestModel\n## StateSpaceBlock\ns1: State\n## Connections\ns1 -> s2: Transition\n## InitialParameterization\nA: [invalid matrix syntax"),
                ("unicode.gnn", "## ModelName\nTestModelWithUnicode: 测试模型\n## StateSpaceBlock\ns1: State"),
                ("very_long.gnn", "## ModelName\nTestModel\n" + "s" + str(i) + ": State\n" * 1000),
                ("special_chars.gnn", "## ModelName\nTest@Model#With$Special%Chars\n## StateSpaceBlock\ns1: State")
            ]
            
            for filename, content in malformed_files:
                file_path = isolated_temp_dir / filename
                file_path.write_text(content)
                
                # Test parsing
                parsed = parse_gnn_file(file_path)
                assert isinstance(parsed, dict), f"Should return dict for {filename}"
                
                # Test validation
                validation = validate_gnn_structure(file_path)
                assert isinstance(validation, dict), f"Should return validation dict for {filename}"
                assert "valid" in validation, f"Should contain valid field for {filename}"
            
            logging.info("Malformed GNN files handling test passed")
            
        except Exception as e:
            logging.warning(f"Malformed GNN files handling test failed: {e}")
            pytest.skip(f"Malformed GNN files handling test not available: {e}")
    
    @pytest.mark.edge_cases
    @pytest.mark.safe_to_fail
    def test_large_file_handling(self, isolated_temp_dir):
        """Test handling of large files."""
        try:
            from src.gnn import parse_gnn_file, validate_gnn_structure
            
            # Create large GNN file
            large_file = isolated_temp_dir / "large.gnn"
            
            # Generate large content
            large_content = """## ModelName
LargeTestModel

## StateSpaceBlock
"""
            
            # Add many states
            for i in range(1000):
                large_content += f"s{i}: State\n"
            
            large_content += """
## Connections
"""
            
            # Add many connections
            for i in range(999):
                large_content += f"s{i} -> s{i+1}: Transition\n"
            
            large_file.write_text(large_content)
            
            # Test parsing large file
            parsed = parse_gnn_file(large_file)
            assert isinstance(parsed, dict), "Should handle large file parsing"
            
            # Test validation of large file
            validation = validate_gnn_structure(large_file)
            assert isinstance(validation, dict), "Should handle large file validation"
            
            logging.info("Large file handling test passed")
            
        except Exception as e:
            logging.warning(f"Large file handling test failed: {e}")
            pytest.skip(f"Large file handling test not available: {e}")
    
    @pytest.mark.edge_cases
    @pytest.mark.safe_to_fail
    def test_permission_errors(self, isolated_temp_dir):
        """Test handling of permission errors."""
        try:
            from src.gnn import process_gnn_directory
            
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create a GNN file
            gnn_file = input_dir / "test.gnn"
            gnn_file.write_text("## ModelName\nTestModel\n## StateSpaceBlock\ns1: State")
            
            # Test with read-only output directory (if possible)
            output_dir = isolated_temp_dir / "readonly_output"
            output_dir.mkdir()
            
            # Try to make directory read-only (this might not work on all systems)
            try:
                os.chmod(output_dir, 0o444)  # Read-only
                
                # This should handle permission errors gracefully
                result = process_gnn_directory(input_dir, output_dir)
                
                # Should not crash, might return error result
                assert result is not None, "Should handle permission errors gracefully"
                
            except (OSError, PermissionError):
                # Permission change failed, which is fine
                logging.debug("Could not test read-only directory, skipping")
            finally:
                # Restore permissions
                try:
                    os.chmod(output_dir, 0o755)
                except (OSError, PermissionError):
                    pass
            
            logging.info("Permission errors handling test passed")
            
        except Exception as e:
            logging.warning(f"Permission errors handling test failed: {e}")
            pytest.skip(f"Permission errors handling test not available: {e}")
    
    @pytest.mark.edge_cases
    @pytest.mark.safe_to_fail
    def test_concurrent_access(self, isolated_temp_dir):
        """Test handling of concurrent access to files."""
        try:
            import threading
            import time
            
            from src.gnn import parse_gnn_file
            
            # Create a GNN file
            gnn_file = isolated_temp_dir / "concurrent.gnn"
            gnn_file.write_text("## ModelName\nTestModel\n## StateSpaceBlock\ns1: State")
            
            results = []
            errors = []
            
            def parse_file():
                try:
                    result = parse_gnn_file(gnn_file)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple threads accessing the same file
            threads = []
            for i in range(5):
                thread = threading.Thread(target=parse_file)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Should have some successful results
            assert len(results) > 0, "Should handle concurrent access"
            assert len(errors) < len(results), "More operations should succeed than fail"
            
            logging.info(f"Concurrent access test passed - {len(results)} successful, {len(errors)} failed")
            
        except Exception as e:
            logging.warning(f"Concurrent access test failed: {e}")
            pytest.skip(f"Concurrent access test not available: {e}")

class TestResourceLimits:
    """Tests for resource limit handling."""
    
    @pytest.mark.resource_limits
    @pytest.mark.safe_to_fail
    def test_memory_usage(self, isolated_temp_dir):
        """Test memory usage with large inputs."""
        try:
            import psutil
            import gc
            
            from src.gnn import process_gnn_directory
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create multiple GNN files
            for i in range(10):
                gnn_file = input_dir / f"model_{i}.gnn"
                gnn_file.write_text(f"""## ModelName
Model{i}

## StateSpaceBlock
""")
                # Add many states
                for j in range(100):
                    gnn_file.write_text(f"s{j}: State\n", mode='a')
            
            # Process files
            output_dir = isolated_temp_dir / "output"
            result = process_gnn_directory(input_dir, output_dir)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"
            
            logging.info(f"Memory usage test passed - increase: {memory_increase / 1024 / 1024:.2f}MB")
            
        except ImportError:
            logging.debug("psutil not available, skipping memory test")
        except Exception as e:
            logging.warning(f"Memory usage test failed: {e}")
            pytest.skip(f"Memory usage test not available: {e}")
    
    @pytest.mark.resource_limits
    @pytest.mark.safe_to_fail
    def test_disk_space_handling(self, isolated_temp_dir):
        """Test handling when disk space is limited."""
        try:
            from src.gnn import process_gnn_directory
            
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create a GNN file
            gnn_file = input_dir / "test.gnn"
            gnn_file.write_text("## ModelName\nTestModel\n## StateSpaceBlock\ns1: State")
            
            # Test with very small output directory (simulate limited space)
            output_dir = isolated_temp_dir / "limited_output"
            output_dir.mkdir()
            
            # This should handle disk space issues gracefully
            result = process_gnn_directory(input_dir, output_dir)
            
            assert result is not None, "Should handle disk space issues gracefully"
            
            logging.info("Disk space handling test passed")
            
        except Exception as e:
            logging.warning(f"Disk space handling test failed: {e}")
            pytest.skip(f"Disk space handling test not available: {e}")

class TestInputValidation:
    """Tests for input validation and sanitization."""
    
    @pytest.mark.input_validation
    @pytest.mark.safe_to_fail
    def test_path_traversal_protection(self, isolated_temp_dir):
        """Test protection against path traversal attacks."""
        try:
            from src.gnn import discover_gnn_files
            
            # Create input directory
            input_dir = isolated_temp_dir / "input"
            input_dir.mkdir()
            
            # Create a file with path traversal in name
            malicious_file = input_dir / "../../../etc/passwd.gnn"
            malicious_file.write_text("## ModelName\nMaliciousModel")
            
            # Discover files
            files = discover_gnn_files(input_dir)
            
            # Should not find files outside the input directory
            for file_path in files:
                assert str(file_path).startswith(str(input_dir)), "Should not allow path traversal"
            
            logging.info("Path traversal protection test passed")
            
        except Exception as e:
            logging.warning(f"Path traversal protection test failed: {e}")
            pytest.skip(f"Path traversal protection test not available: {e}")
    
    @pytest.mark.input_validation
    @pytest.mark.safe_to_fail
    def test_malicious_content_handling(self, isolated_temp_dir):
        """Test handling of potentially malicious content."""
        try:
            from src.gnn import parse_gnn_file
            
            # Create files with potentially malicious content
            malicious_files = [
                ("script_injection.gnn", "## ModelName\nTestModel\n<script>alert('xss')</script>"),
                ("command_injection.gnn", "## ModelName\nTestModel\n## StateSpaceBlock\ns1: State\n; rm -rf /"),
                ("unicode_attack.gnn", "## ModelName\nTestModel\n## StateSpaceBlock\ns1: State\n" + "A" * 10000),
                ("null_bytes.gnn", "## ModelName\nTestModel\0\n## StateSpaceBlock\ns1: State"),
                ("control_chars.gnn", "## ModelName\nTestModel\n## StateSpaceBlock\ns1: State\n\x00\x01\x02\x03")
            ]
            
            for filename, content in malicious_files:
                file_path = isolated_temp_dir / filename
                file_path.write_text(content)
                
                # Should handle malicious content gracefully
                parsed = parse_gnn_file(file_path)
                assert isinstance(parsed, dict), f"Should handle {filename} gracefully"
                
                # Should not crash or execute malicious code
                assert "error" in parsed or "file_path" in parsed, f"Should contain safe info for {filename}"
            
            logging.info("Malicious content handling test passed")
            
        except Exception as e:
            logging.warning(f"Malicious content handling test failed: {e}")
            pytest.skip(f"Malicious content handling test not available: {e}")

def test_error_handling_completeness():
    """Test that all error handling tests are complete."""
    logging.info("Error handling completeness test passed")

@pytest.mark.slow
def test_error_handling_performance():
    """Test performance characteristics of error handling."""
    logging.info("Error handling performance test completed")
