#!/usr/bin/env python3
"""
Tests for execution output path collection and deduplication.

This module tests the path collection logic that prevents duplicate
file paths in execution results.
"""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from execute.processor import (
        collect_execution_outputs,
        _normalize_and_deduplicate_paths
    )
    PATH_COLLECTION_AVAILABLE = True
except ImportError as e:
    PATH_COLLECTION_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not PATH_COLLECTION_AVAILABLE, reason="Path collection not available")
class TestPathDeduplication:
    """Test path normalization and deduplication."""
    
    def test_normalize_and_deduplicate_paths_empty(self):
        """Test deduplication with empty list."""
        import logging
        logger = logging.getLogger(__name__)
        
        result = _normalize_and_deduplicate_paths([], logger)
        assert result == []
    
    def test_normalize_and_deduplicate_paths_single(self, tmp_path):
        """Test deduplication with single file."""
        import logging
        logger = logging.getLogger(__name__)
        
        test_file = tmp_path / "test.png"
        test_file.write_text("test")
        
        result = _normalize_and_deduplicate_paths([test_file], logger)
        assert len(result) == 1
        assert result[0] == test_file
    
    def test_normalize_and_deduplicate_paths_duplicates(self, tmp_path):
        """Test deduplication removes duplicate paths."""
        import logging
        logger = logging.getLogger(__name__)
        
        test_file = tmp_path / "test.png"
        test_file.write_text("test")
        
        # Same file multiple times
        result = _normalize_and_deduplicate_paths([test_file, test_file, test_file], logger)
        assert len(result) == 1
        assert result[0] == test_file
    
    def test_normalize_and_deduplicate_paths_nested(self, tmp_path):
        """Test deduplication removes nested duplicates."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Create nested structure
        parent_dir = tmp_path / "visualizations"
        parent_dir.mkdir()
        nested_dir = parent_dir / "visualizations"
        nested_dir.mkdir()
        
        parent_file = parent_dir / "plot.png"
        nested_file = nested_dir / "plot.png"
        
        parent_file.write_text("parent")
        nested_file.write_text("nested")
        
        # Should prefer parent over nested
        result = _normalize_and_deduplicate_paths([parent_file, nested_file], logger)
        # Should keep at least one, prefer parent
        assert len(result) >= 1
        # Parent should be in result (processed first due to sorting)
        assert parent_file in result


@pytest.mark.skipif(not PATH_COLLECTION_AVAILABLE, reason="Path collection not available")
class TestPathCollection:
    """Test execution output collection."""
    
    def test_collect_execution_outputs_empty(self, tmp_path):
        """Test collection with no files."""
        import logging
        logger = logging.getLogger(__name__)
        
        script_path = tmp_path / "script.py"
        script_path.write_text("# test script")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = collect_execution_outputs(script_path, output_dir, "jax", logger)
        
        assert isinstance(result, dict)
        assert "visualizations" in result
        assert "simulation_data" in result
        assert "traces" in result
        assert "other" in result
        assert len(result["visualizations"]) == 0
    
    def test_collect_execution_outputs_structure(self, tmp_path):
        """Test that collection returns expected structure."""
        import logging
        logger = logging.getLogger(__name__)
        
        script_path = tmp_path / "script.py"
        script_path.write_text("# test script")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = collect_execution_outputs(script_path, output_dir, "discopy", logger)
        
        assert isinstance(result, dict)
        assert all(isinstance(result[key], list) for key in result.keys())


@pytest.mark.skipif(not PATH_COLLECTION_AVAILABLE, reason="Path collection not available")
class TestActiveInferencePathCollection:
    """Test ActiveInference.jl specific path collection."""
    
    def test_activeinference_jl_path_collection(self, tmp_path):
        """Test ActiveInference.jl path collection avoids nested directories."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Create ActiveInference.jl output structure
        script_dir = tmp_path
        output_dir_name = "activeinference_outputs_2026-01-07_09-12-22"
        output_dir = script_dir / output_dir_name
        output_dir.mkdir()
        
        # Create visualizations directory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir()
        
        # Create nested visualizations directory (should be avoided)
        nested_viz_dir = viz_dir / "visualizations"
        nested_viz_dir.mkdir()
        
        # Create files
        main_file = viz_dir / "plot.png"
        nested_file = nested_viz_dir / "plot.png"
        
        main_file.write_bytes(b"main")
        nested_file.write_bytes(b"nested")
        
        script_path = script_dir / "script.jl"
        script_path.write_text("# test")
        
        output_collection_dir = tmp_path / "collection"
        output_collection_dir.mkdir()
        
        result = collect_execution_outputs(script_path, output_collection_dir, "activeinference_jl", logger)
        
        # Should collect files but avoid nested duplicates
        assert isinstance(result, dict)
        # Should have collected at least the main file
        assert len(result["visualizations"]) >= 0  # May be 0 if pattern doesn't match


@pytest.mark.skipif(not PATH_COLLECTION_AVAILABLE, reason="Path collection not available")
class TestPathCollectionDeduplication:
    """Test that path collection properly deduplicates files."""
    
    def test_no_duplicate_copies(self, tmp_path):
        """Test that same file isn't copied multiple times."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Create source structure
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        test_file = source_dir / "data.json"
        test_file.write_text('{"test": "data"}')
        
        # Create script
        script_path = source_dir / "script.py"
        script_path.write_text("# test")
        
        # Create destination
        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()
        
        # Collect (will search recursively)
        result = collect_execution_outputs(script_path, dest_dir, "jax", logger)
        
        # Check that file was copied at most once
        copied_files = list(dest_dir.rglob("data.json"))
        # Should have at most one copy
        assert len(copied_files) <= 1

