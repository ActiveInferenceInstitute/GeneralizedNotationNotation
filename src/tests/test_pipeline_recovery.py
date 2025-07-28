#!/usr/bin/env python3
"""
Pipeline Recovery Tests

This module provides comprehensive testing for pipeline error recovery scenarios,
focusing on known failure modes and their recovery mechanisms.

Key test areas:
1. NumPy recursion error recovery
2. Async/await error handling
3. Lightweight processing fallback
4. Hardware initialization issues
5. Resource management
6. Error reporting
"""

import pytest
import sys
import os
import tempfile
import logging
import asyncio
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock
from typing import Dict, Any, List

# Import test utilities
from . import (
    TEST_CONFIG,
    get_test_args,
    create_test_files,
    performance_tracker,
    TEST_DIR,
    SRC_DIR,
    PROJECT_ROOT
)

# Test markers
pytestmark = [pytest.mark.recovery, pytest.mark.safe_to_fail]

@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Create necessary subdirectories
        (temp_path / "input").mkdir()
        (temp_path / "output").mkdir()
        yield temp_path

@pytest.fixture
def sample_gnn_file(mock_environment):
    """Create a sample GNN file for testing."""
    file_path = mock_environment / "input" / "test_model.md"
    file_path.write_text("""
    # ModelName
    Test Model
    
    # StateSpaceBlock
    s[3,1]
    
    # Connections
    s -> s
    """)
    return file_path

class TestRecursionErrorRecovery:
    """Test suite for NumPy recursion error recovery."""
    
    def test_numpy_import_recovery(self, mock_environment):
        """Test recovery from NumPy import recursion error."""
        with patch('numpy.typing._array_like', side_effect=RecursionError):
            with pytest.raises(RecursionError):
                import numpy as np
                
            # Should recover by increasing recursion limit
            sys.setrecursionlimit(3000)
            import numpy as np
            assert np is not None
            
    def test_render_step_recovery(self, mock_environment, sample_gnn_file):
        """Test render step recovery from recursion errors."""
        from render.renderer import render_gnn_files
        
        with patch('numpy.typing', side_effect=RecursionError):
            result = render_gnn_files(
                target_dir=mock_environment / "input",
                output_dir=mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        assert "recursion_limit_adjusted" in result["recovery_actions"]

class TestAsyncAwaitRecovery:
    """Test suite for async/await error recovery."""
    
    @pytest.mark.asyncio
    async def test_llm_analysis_recovery(self, mock_environment, sample_gnn_file):
        """Test LLM analysis with proper async/await handling."""
        from llm.analyzer import analyze_gnn_file_with_llm
        
        # Mock async OpenAI provider
        mock_provider = AsyncMock()
        mock_provider.analyze.return_value = "Test analysis"
        
        with patch('llm.analyzer.OpenAIProvider', return_value=mock_provider):
            result = await analyze_gnn_file_with_llm(sample_gnn_file)
            
        assert result["status"] == "SUCCESS"
        assert "analysis" in result
        assert mock_provider.analyze.await_count == 1
        
    def test_sync_wrapper_recovery(self, mock_environment, sample_gnn_file):
        """Test synchronous wrapper for async LLM analysis."""
        from llm.analyzer import analyze_gnn_file_with_llm
        
        mock_provider = AsyncMock()
        mock_provider.analyze.return_value = "Test analysis"
        
        with patch('llm.analyzer.OpenAIProvider', return_value=mock_provider):
            result = analyze_gnn_file_with_llm(sample_gnn_file)
            
        assert result["status"] == "SUCCESS"
        assert "analysis" in result

class TestLightweightProcessingRecovery:
    """Test suite for lightweight processing fallback."""
    
    def test_gnn_lightweight_fallback(self, mock_environment, sample_gnn_file):
        """Test fallback to lightweight GNN processing."""
        from gnn.core_processor import process_gnn_directory
        
        # Force full processing failure
        with patch('gnn.core_processor.process_gnn_directory_full', side_effect=ImportError):
            result = process_gnn_directory(
                mock_environment / "input",
                mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        assert result["processing_mode"] == "lightweight"
        assert str(sample_gnn_file) in result["processed_files"]
        
    def test_lightweight_processing_output(self, mock_environment, sample_gnn_file):
        """Test output quality of lightweight processing."""
        from gnn.core_processor import process_gnn_directory_lightweight
        
        result = process_gnn_directory_lightweight(mock_environment / "input")
        
        assert str(sample_gnn_file) in result
        assert result[str(sample_gnn_file)]["status"] == "processed"
        assert result[str(sample_gnn_file)]["format"] == "markdown"
        assert isinstance(result[str(sample_gnn_file)]["size"], int)

class TestHardwareInitializationRecovery:
    """Test suite for hardware initialization recovery."""
    
    def test_jax_cpu_fallback(self, mock_environment):
        """Test JAX CPU fallback when TPU/GPU unavailable."""
        from execute.jax.jax_runner import initialize_jax_devices
        
        # Force TPU error
        with patch('jax.devices', side_effect=RuntimeError("No TPU available")):
            devices = initialize_jax_devices()
            
        assert len(devices) > 0
        assert "cpu" in str(devices[0]).lower()
        
    def test_execution_hardware_recovery(self, mock_environment, sample_gnn_file):
        """Test execution with hardware fallback."""
        from execute.executor import execute_gnn_model
        
        # Mock hardware detection
        with patch('execute.executor.get_available_hardware', return_value=["cpu"]):
            result = execute_gnn_model(
                sample_gnn_file,
                mock_environment / "output"
            )
            
        assert result["status"] == "SUCCESS"
        assert result["execution_device"] == "cpu"

class TestResourceManagementRecovery:
    """Test suite for resource management recovery."""
    
    def test_memory_limit_recovery(self, mock_environment):
        """Test recovery from memory limit issues."""
        from utils.resource_manager import with_resource_limits
        
        @with_resource_limits(max_memory_mb=100)
        def memory_intensive_operation():
            # Simulate memory-intensive operation
            big_list = [0] * (1024 * 1024)  # 1MB
            return "Success"
        
        result = memory_intensive_operation()
        assert result == "Success"
        
    def test_disk_space_recovery(self, mock_environment):
        """Test recovery from disk space issues."""
        from utils.resource_manager import check_disk_space
        
        # Mock low disk space
        with patch('shutil.disk_usage', return_value=(100, 50, 10)):
            with pytest.raises(RuntimeError, match="Insufficient disk space"):
                check_disk_space(mock_environment, required_mb=100)
                
            # Should succeed with lower requirement
            assert check_disk_space(mock_environment, required_mb=5)

class TestErrorReportingRecovery:
    """Test suite for error reporting mechanisms."""
    
    def test_error_collection(self, mock_environment):
        """Test error collection and reporting."""
        from utils.error_recovery import ErrorReporter
        
        reporter = ErrorReporter()
        
        # Simulate error collection
        reporter.collect_error("test_error", "Test error message")
            
        errors = reporter.get_errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "test_error"
        assert errors[0]["message"] == "Test error message"
        
    def test_error_recovery_logging(self, mock_environment):
        """Test error recovery logging functionality."""
        from utils.logging_utils import log_step_error
        
        # Test error logging
        result = log_step_error("test_step", "Test error occurred", None)
            
        assert result is not None
        # The function should handle the error gracefully

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 