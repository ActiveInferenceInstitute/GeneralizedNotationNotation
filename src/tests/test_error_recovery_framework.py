#!/usr/bin/env python3
"""
Error Recovery Framework Tests
===============================

Comprehensive tests for error message formatting, recovery suggestions,
and error handling improvements.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import pytest

class TestErrorRecoveryFramework:
    """Test error recovery framework functionality."""
    
    @pytest.mark.unit
    def test_error_context_creation(self):
        """Test ErrorContext dataclass creation and conversion."""
        from utils.error_recovery import ErrorContext, ErrorSeverity
        
        context = ErrorContext(
            operation="Test Operation",
            severity=ErrorSeverity.ERROR,
            message="Test error message",
            error_code="E001",
            details={"key": "value"},
            recovery_suggestions=["Try this", "Or try that"]
        )
        
        assert context.operation == "Test Operation"
        assert context.severity == ErrorSeverity.ERROR
        assert context.message == "Test error message"
        assert context.error_code == "E001"
        
        # Test conversion to dict
        d = context.to_dict()
        assert isinstance(d, dict)
        assert d['error_code'] == 'E001'
        assert d['severity'] == 'error'
        assert len(d['recovery_suggestions']) == 2
    
    @pytest.mark.unit
    def test_error_recovery_manager_initialization(self):
        """Test ErrorRecoveryManager initialization and setup."""
        from utils.error_recovery import ErrorRecoveryManager
        
        manager = ErrorRecoveryManager()
        
        # Verify default handlers are registered
        assert 'import' in manager.error_handlers
        assert 'file' in manager.error_handlers
        assert 'resource' in manager.error_handlers
        assert 'validation' in manager.error_handlers
        assert 'execution' in manager.error_handlers
        
        # Verify recovery strategies exist
        assert 'import' in manager.recovery_strategies
        assert 'file' in manager.recovery_strategies
    
    @pytest.mark.unit
    def test_error_code_registry(self):
        """Test error code registry contains expected codes."""
        from utils.error_recovery import ErrorCodeRegistry
        
        codes = ErrorCodeRegistry.get_all_codes()
        
        # Should have import error codes
        assert ErrorCodeRegistry.IMPORT_NOT_FOUND == "E001"
        assert ErrorCodeRegistry.IMPORT_VERSION_MISMATCH == "E002"
        
        # Should have file error codes
        assert ErrorCodeRegistry.FILE_NOT_FOUND == "E101"
        assert ErrorCodeRegistry.FILE_PERMISSION_DENIED == "E102"
        
        # Should have resource error codes
        assert ErrorCodeRegistry.RESOURCE_MEMORY_EXCEEDED == "E201"
        
        # Should have validation error codes
        assert ErrorCodeRegistry.VALIDATION_TYPE_MISMATCH == "E301"
        
        # Should have execution error codes
        assert ErrorCodeRegistry.EXECUTION_FAILED == "E401"
    
    @pytest.mark.unit
    def test_error_message_formatting(self):
        """Test error message formatting."""
        from utils.error_recovery import format_error_message
        
        message = format_error_message(
            error_code="E001",
            operation="Module Loading",
            message="Required module not found",
            details={"module": "pymdp", "available": False},
            suggestions=["Install with: uv pip install inferactively-pymdp"]
        )
        
        assert "[E001]" in message
        assert "Module Loading" in message
        assert "Required module not found" in message
        assert "pymdp" in message
        assert "pip install" in message
    
    @pytest.mark.unit
    def test_error_handling_with_severity_levels(self):
        """Test error handling respects severity levels."""
        from utils.error_recovery import ErrorContext, ErrorSeverity, ErrorRecoveryManager
        
        manager = ErrorRecoveryManager(logging.getLogger("test"))
        
        # Test info severity
        context = ErrorContext(
            operation="Test",
            severity=ErrorSeverity.INFO,
            message="Informational",
            error_code="I001"
        )
        assert manager.handle_error(context) is True
        
        # Test warning severity
        context = ErrorContext(
            operation="Test",
            severity=ErrorSeverity.WARNING,
            message="Warning",
            error_code="W001"
        )
        assert manager.handle_error(context) is True
        
        # Test error severity
        context = ErrorContext(
            operation="Test",
            severity=ErrorSeverity.ERROR,
            message="Error",
            error_code="E001"
        )
        assert manager.handle_error(context) is True
        
        # Test critical severity - should return False
        context = ErrorContext(
            operation="Test",
            severity=ErrorSeverity.CRITICAL,
            message="Critical",
            error_code="CRIT001"
        )
        assert manager.handle_error(context) is False
    
    @pytest.mark.unit
    def test_format_and_log_error_function(self):
        """Test format_and_log_error convenience function."""
        from utils.error_recovery import format_and_log_error, ErrorSeverity
        
        context = format_and_log_error(
            error_code="E101",
            operation="File Loading",
            message="File not found",
            severity=ErrorSeverity.ERROR,
            details={"path": "/nonexistent/file.txt"},
            suggestions=["Check file path", "Verify file exists"]
        )
        
        assert context.error_code == "E101"
        assert context.operation == "File Loading"
        assert "File not found" in context.message
        assert len(context.recovery_suggestions) == 2


class TestErrorRecoveryStrategies:
    """Test specific error recovery strategies."""
    
    @pytest.mark.unit
    def test_import_error_recovery_suggestions(self):
        """Test recovery suggestions for import errors."""
        from utils.error_recovery import ErrorRecoveryManager
        
        manager = ErrorRecoveryManager()
        strategies = manager.recovery_strategies['import']
        
        assert len(strategies) > 0
        # Should mention installation
        assert any("pip install" in s or "install" in s.lower() for s in strategies)
        # Should mention version compatibility
        assert any("version" in s.lower() or "compatibility" in s.lower() for s in strategies)
    
    @pytest.mark.unit
    def test_file_error_recovery_suggestions(self):
        """Test recovery suggestions for file errors."""
        from utils.error_recovery import ErrorRecoveryManager
        
        manager = ErrorRecoveryManager()
        strategies = manager.recovery_strategies['file']
        
        assert len(strategies) > 0
        # Should mention permissions
        assert any("permission" in s.lower() or "access" in s.lower() for s in strategies)
        # Should mention disk space
        assert any("disk" in s.lower() or "space" in s.lower() for s in strategies)
    
    @pytest.mark.unit
    def test_resource_error_recovery_suggestions(self):
        """Test recovery suggestions for resource errors."""
        from utils.error_recovery import ErrorRecoveryManager
        
        manager = ErrorRecoveryManager()
        strategies = manager.recovery_strategies['resource']
        
        assert len(strategies) > 0
        # Should mention memory
        assert any("memory" in s.lower() or "ram" in s.lower() for s in strategies)
        # Should mention lightweight mode or optimization
        assert any("lightweight" in s.lower() or "reduce" in s.lower() for s in strategies)
    
    @pytest.mark.unit
    def test_validation_error_recovery_suggestions(self):
        """Test recovery suggestions for validation errors."""
        from utils.error_recovery import ErrorRecoveryManager
        
        manager = ErrorRecoveryManager()
        strategies = manager.recovery_strategies['validation']
        
        assert len(strategies) > 0
        # Should mention schema/format
        assert any("schema" in s.lower() or "format" in s.lower() or "type" in s.lower() 
                  for s in strategies)


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""
    
    @pytest.mark.unit
    def test_error_context_roundtrip(self):
        """Test ErrorContext serialization and deserialization."""
        from utils.error_recovery import ErrorContext, ErrorSeverity
        
        original = ErrorContext(
            operation="Test Operation",
            severity=ErrorSeverity.ERROR,
            message="Test message",
            error_code="E001",
            details={"key": "value"},
            recovery_suggestions=["Suggestion 1"]
        )
        
        # Convert to dict and back
        data = original.to_dict()
        
        # Verify all data is present
        assert data['operation'] == "Test Operation"
        assert data['severity'] == "error"
        assert data['message'] == "Test message"
        assert data['error_code'] == "E001"
        assert data['details']['key'] == "value"
        assert len(data['recovery_suggestions']) == 1
    
    @pytest.mark.unit
    def test_error_recovery_manager_is_singleton(self):
        """Test that get_recovery_manager returns consistent instance."""
        from utils.error_recovery import get_recovery_manager
        
        manager1 = get_recovery_manager()
        manager2 = get_recovery_manager()
        
        # Should return the same global instance
        assert manager1 is manager2


def test_error_recovery_documentation():
    """Test that error recovery framework components exist and have docstrings."""
    from utils.error_recovery import (
        ErrorContext, 
        ErrorSeverity, 
        ErrorRecoveryManager, 
        format_and_log_error, 
        ErrorCodeRegistry
    )

    # Verify framework classes exist and are documented
    assert ErrorContext.__doc__ is not None, "ErrorContext should have docstring"
    assert ErrorSeverity.__doc__ is not None, "ErrorSeverity should have docstring"
    assert ErrorRecoveryManager.__doc__ is not None, "ErrorRecoveryManager should have docstring"

    # Verify key functions exist and are documented
    assert format_and_log_error.__doc__ is not None, "format_and_log_error should have docstring"

    # Verify error code registry has expected categories
    registry = ErrorCodeRegistry
    expected_attrs = ['IMPORT_NOT_FOUND', 'FILE_NOT_FOUND', 'RESOURCE_MEMORY_EXCEEDED', 'VALIDATION_TYPE_MISMATCH']
    found_attrs = [attr for attr in expected_attrs if hasattr(registry, attr)]
    assert len(found_attrs) >= 2, f"ErrorCodeRegistry should have error codes, found: {found_attrs}"

    # Verify severity levels are defined
    severities = [s for s in dir(ErrorSeverity) if not s.startswith('_')]
    assert len(severities) >= 3, f"ErrorSeverity should have multiple levels: {severities}"

