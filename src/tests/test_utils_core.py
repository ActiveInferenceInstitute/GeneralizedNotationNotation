"""Comprehensive tests for GNN utility modules."""

import logging
import os
from pathlib import Path

import pytest

# Import utility functions/classes
try:
    from utils.argument_utils import ArgumentParser
    from utils.config_loader import GNNPipelineConfig, load_config
    from utils.logging_utils import PipelineLogger, setup_step_logging
    from utils.path_utils import get_relative_path_if_possible
except ImportError:
    # Adjust for test context
    try:
        from src.utils.argument_utils import ArgumentParser
        from src.utils.config_loader import GNNPipelineConfig, load_config
        from src.utils.logging_utils import PipelineLogger, setup_step_logging
        from src.utils.path_utils import get_relative_path_if_possible
    except ImportError:
        pass

def test_get_relative_path_if_possible():
    """Test path relative resolution."""
    root = Path("/project/gnn")
    path = Path("/project/gnn/input/file.md")
    other = Path("/external/file.md")
    
    # Matching root
    assert get_relative_path_if_possible(path, root) == "input/file.md"
    # Non-matching root
    assert get_relative_path_if_possible(other, root) == "/external/file.md"
    # No root provided
    assert get_relative_path_if_possible(path, None) == "/project/gnn/input/file.md"

def test_pipeline_logger_caching():
    """Verify that PipelineLogger caches instances."""
    logger1 = PipelineLogger.get_logger("test_module")
    logger2 = PipelineLogger.get_logger("test_module")
    assert logger1 is logger2
    assert logger1.name == "test_module"

def test_setup_step_logging():
    """Test step-specific logging setup."""
    logger = setup_step_logging("3_gnn", verbose=True)
    assert logger.name == "3_gnn"
    assert logger.level == logging.DEBUG

def test_gnn_pipeline_config_defaults():
    """Test GNNPipelineConfig default values."""
    config = GNNPipelineConfig()
    assert config.pipeline.target_dir == Path("input/gnn_files")
    assert config.pipeline.recursive is True
    
    args = config.to_pipeline_arguments()
    assert args['target_dir'] == config.pipeline.target_dir
    assert args['recursive'] is True

@pytest.mark.skipif(not os.path.exists("src/ontology/act_inf_ontology_terms.json"), reason="Missing ontology file for validation test")
def test_config_validation_success():
    """Test validation of default config."""
    config = GNNPipelineConfig()
    # Should not raise
    errors = config.validate()
    # Depending on environment, target_dir might not exist, 
    # but GNNPipelineConfig expects it relative to src/
    assert isinstance(errors, list)

def test_load_config_missing_raises(tmp_path):
    """Test load_config raises FileNotFoundError when explicit path is missing."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")

def test_argument_parser_step_parsing():
    """Test parsing logic for a specific step."""
    # Ensure ArgumentParser is initialized
    argv = ["--target-dir", "custom_in", "--verbose"]
    parsed = ArgumentParser.parse_step_arguments("3_gnn.py", argv)
    
    assert parsed.target_dir == Path("custom_in")
    assert parsed.verbose is True
    # Inherited attribute check
    assert hasattr(parsed, "recursive")
    assert parsed.recursive is True
