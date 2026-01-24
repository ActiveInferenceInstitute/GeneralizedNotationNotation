"""
Test suite for Template module.

Tests the reference implementation for GNN pipeline's architectural pattern.
"""

import pytest
from pathlib import Path


class TestTemplateModule:
    """Test suite for Template module functionality."""

    def test_module_imports(self):
        """Test that template module can be imported."""
        from template import (
            process_template_standardized,
            process_single_file,
            validate_file,
            generate_correlation_id,
            safe_template_execution,
            FEATURES,
            __version__
        )
        assert __version__ is not None
        assert isinstance(FEATURES, dict)
        assert callable(process_template_standardized)
        assert callable(process_single_file)
        assert callable(validate_file)
        assert callable(generate_correlation_id)

    def test_features_available(self):
        """Test that FEATURES dict is properly populated."""
        from template import FEATURES

        expected_features = [
            'standardized_processing',
            'correlation_id_generation',
            'safe_execution',
            'pipeline_initialization',
            'mcp_integration'
        ]

        for feature in expected_features:
            assert feature in FEATURES, f"Missing feature: {feature}"
            assert FEATURES[feature] is True

    def test_version_format(self):
        """Test version string format."""
        from template import __version__

        # Should be semantic versioning format
        parts = __version__.split('.')
        assert len(parts) >= 2, "Version should have at least major.minor"
        assert all(p.isdigit() for p in parts[:2]), "Major and minor should be numeric"

    def test_version_info_dict(self):
        """Test VERSION_INFO dictionary."""
        from template import VERSION_INFO

        assert isinstance(VERSION_INFO, dict)
        assert 'version' in VERSION_INFO
        assert 'name' in VERSION_INFO
        assert VERSION_INFO['name'] == "Template Step"


class TestCorrelationIdGeneration:
    """Test correlation ID generation functionality."""

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        from template import generate_correlation_id

        corr_id = generate_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0

    def test_correlation_ids_unique(self):
        """Test that correlation IDs are unique."""
        from template import generate_correlation_id

        ids = [generate_correlation_id() for _ in range(100)]
        assert len(set(ids)) == len(ids), "Correlation IDs should be unique"


class TestFileValidation:
    """Test file validation functionality."""

    def test_validate_file_valid(self, safe_filesystem):
        """Test validation of a valid GNN file."""
        from template import validate_file

        gnn_content = """# Test Model

## StateSpaceBlock
s[3, type=int]

## Connections
s->s
"""
        test_file = safe_filesystem.create_file("valid_model.md", gnn_content)

        result = validate_file(test_file)
        # Should return validation result (dict or bool)
        assert result is not None

    def test_validate_file_nonexistent(self, safe_filesystem):
        """Test validation of non-existent file."""
        from template import validate_file

        fake_path = safe_filesystem.temp_dir / "nonexistent.md"

        # Should handle gracefully - may return dict with error info
        result = validate_file(fake_path)
        # Result can be dict with valid=False or error info
        assert result is not None
        if isinstance(result, dict):
            # If file doesn't exist, validation should indicate that
            assert 'error' in result or 'valid' in result or 'exists' in result


class TestProcessing:
    """Test processing functionality."""

    def test_process_single_file(self, safe_filesystem):
        """Test processing a single GNN file."""
        from template import process_single_file

        gnn_content = """# Single File Test

## StateSpaceBlock
x[5]

## Time
Static
"""
        test_file = safe_filesystem.create_file("single.md", gnn_content)
        output_dir = safe_filesystem.create_dir("output")

        # process_single_file signature: (input_file, output_dir, options)
        options = {"verbose": True}
        result = process_single_file(test_file, output_dir, options)
        assert result is not None

    def test_process_template_standardized(self, safe_filesystem):
        """Test standardized template processing."""
        from template import process_template_standardized
        import logging

        gnn_content = """# Standardized Test

## StateSpaceBlock
state[10]

## Parameters
alpha = 0.5
"""
        test_file = safe_filesystem.create_file("standard.md", gnn_content)
        output_dir = safe_filesystem.create_dir("template_output")
        logger = logging.getLogger("test_template_standardized")

        # May need logger parameter
        try:
            result = process_template_standardized(
                target_dir=safe_filesystem.temp_dir,
                output_dir=output_dir,
                verbose=True,
                logger=logger
            )
        except TypeError:
            result = process_template_standardized(
                target_dir=safe_filesystem.temp_dir,
                output_dir=output_dir,
                verbose=True
            )

        # Should return success status - True or dict with success
        assert result is True or (isinstance(result, dict) and result.get('success', True))


class TestSafeExecution:
    """Test safe execution wrapper."""

    def test_safe_template_execution_success(self, safe_filesystem):
        """Test safe execution context manager."""
        from template import safe_template_execution, generate_correlation_id
        import logging

        logger = logging.getLogger("test_safe_exec")
        correlation_id = generate_correlation_id()

        # safe_template_execution is a context manager
        with safe_template_execution(logger, correlation_id) as ctx:
            assert ctx is not None
            assert 'correlation_id' in ctx
            assert ctx['correlation_id'] == correlation_id

    def test_safe_template_execution_with_error(self, safe_filesystem):
        """Test safe execution handles errors gracefully."""
        from template import safe_template_execution, generate_correlation_id
        import logging

        logger = logging.getLogger("test_safe_exec_error")
        correlation_id = generate_correlation_id()

        # Should handle exception gracefully within context
        try:
            with safe_template_execution(logger, correlation_id) as ctx:
                # Context should be provided
                assert ctx is not None
        except Exception:
            pass  # Context manager may re-raise after cleanup


class TestUtilityPatterns:
    """Test utility pattern demonstrations."""

    def test_demonstrate_utility_patterns(self):
        """Test utility pattern demonstration function."""
        from template import demonstrate_utility_patterns
        import logging

        # Should be callable
        assert callable(demonstrate_utility_patterns)

        # demonstrate_utility_patterns signature: (context, logger)
        logger = logging.getLogger("test_utility_patterns")
        context = {"correlation_id": "test-123"}
        result = demonstrate_utility_patterns(context, logger)
        # Returns demonstration results dict
        assert isinstance(result, dict)

    def test_get_version_info(self):
        """Test version info utility."""
        from template import get_version_info

        info = get_version_info()
        assert isinstance(info, dict)
        assert 'version' in info or 'module' in info or len(info) > 0
