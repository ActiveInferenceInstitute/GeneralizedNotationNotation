# This file marks the directory as a Python package.
# It can also be used to define package-level exports or initialization code for tests.

# This file makes tests a package 

"""
GNN Processing Pipeline - Comprehensive Test Suite

This test package provides comprehensive, modular, and safe-to-fail testing
for all GNN pipeline components and functionalities. It is designed to be
standalone, safe to run in any environment, and thoroughly document all 
method usage and expected behaviors.

Test Categories:
- Unit tests: Individual component testing with mocked dependencies
- Integration tests: Cross-component testing with controlled environments
- Environment tests: Setup and dependency validation
- Pipeline tests: End-to-end workflow testing with mock operations
- Safe-to-fail tests: Tests designed to fail gracefully with mocked dependencies

Testing Architecture:
    conftest.py - Pytest configuration, fixtures, and safety mechanisms
    test_environment.py - Environment validation and setup testing
    test_pipeline_steps.py - Testing for all 14 numbered pipeline steps
    test_main_orchestrator.py - Main pipeline orchestrator testing
    test_utilities.py - Utility function and module testing

Usage Examples:
    # Run all tests safely (default mode)
    pytest src/tests/
    
    # Run specific test categories
    pytest src/tests/ -m "unit"
    pytest src/tests/ -m "integration" 
    pytest src/tests/ -m "environment"
    pytest src/tests/ -m "pipeline"
    pytest src/tests/ -m "safe_to_fail"
    
    # Run with detailed coverage reporting
    pytest src/tests/ --cov=src --cov-report=html --cov-report=term-missing
    
    # Run environment validation (critical for new setups)
    pytest src/tests/test_environment.py -v
    
    # Run pipeline step tests with verbose output
    pytest src/tests/test_pipeline_steps.py -v --tb=short
    
    # Run main orchestrator tests
    pytest src/tests/test_main_orchestrator.py -v
    
    # Run utilities tests
    pytest src/tests/test_utilities.py -v
    
    # Run slow tests (longer execution time)
    pytest src/tests/ -m "slow" --tb=line
    
    # Skip tests requiring external APIs
    pytest src/tests/ -m "not requires_api"
    
    # Run only fast, safe-to-fail tests
    pytest src/tests/ -m "safe_to_fail and not slow"

Safety and Design Principles:
    1. Safe-to-fail: All tests use extensive mocking to prevent destructive operations
    2. Modular: Tests can run independently without dependencies on each other
    3. Comprehensive: Coverage of all 14 pipeline steps plus utilities and orchestration
    4. Environment validation: Thorough checking of setup requirements
    5. Graceful degradation: Tests handle missing dependencies appropriately
    6. Documentation: Each test documents expected usage and method behavior
    7. Isolation: Tests don't interfere with each other or production environment
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import tempfile
import shutil
import json
import logging

# Ensure src directory is in Python path
TEST_DIR = Path(__file__).parent
SRC_DIR = TEST_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Test configuration with comprehensive safe-to-fail settings
TEST_CONFIG = {
    # Safety settings
    "safe_mode": True,  # Enable safe-to-fail mode by default
    "mock_external_deps": True,  # Mock external dependencies by default
    "mock_subprocess": True,  # Mock subprocess calls by default
    "mock_file_operations": False,  # Mock destructive file operations
    "mock_network_calls": True,  # Mock network/API calls
    "isolation_mode": True,  # Isolate tests from each other
    
    # Directory settings
    "temp_output_dir": TEST_DIR / "temp_outputs",
    "test_data_dir": TEST_DIR / "data",
    "sample_gnn_dir": TEST_DIR / "sample_gnn_files",
    "mock_project_root": TEST_DIR / "mock_project",
    
    # Execution settings
    "timeout_seconds": 60,  # Default timeout for tests
    "max_test_files": 10,  # Maximum GNN files to process in tests
    "enable_performance_tracking": True,  # Track test performance
    "verbose_logging": False,  # Verbose logging in tests
    
    # Coverage and quality settings
    "coverage_threshold": 80.0,  # Minimum coverage percentage
    "max_memory_mb": 500,  # Maximum memory usage per test
    "max_execution_time_s": 30,  # Maximum execution time per test
    
    # Component-specific settings
    "gnn_processing": {
        "max_files_per_test": 5,
        "mock_file_parsing": True,
        "validate_syntax": True
    },
    "pipeline_execution": {
        "mock_subprocess_calls": True,
        "simulate_step_failures": True,
        "track_execution_order": True
    },
    "visualization": {
        "mock_plotting": True,
        "generate_test_images": False,
        "validate_output_formats": True
    },
    "export": {
        "mock_file_writes": True,
        "validate_output_schemas": True,
        "test_format_conversions": True
    }
}

# Comprehensive pytest markers for test categorization
PYTEST_MARKERS = {
    # Primary categories
    "unit": "Unit tests for individual components with full mocking",
    "integration": "Integration tests for component interactions with controlled environments", 
    "environment": "Environment setup and dependency validation tests",
    "pipeline": "End-to-end pipeline workflow tests with mock operations",
    "safe_to_fail": "Tests that are designed to fail safely with mocked dependencies",
    
    # Execution characteristics  
    "slow": "Tests that take longer to execute (>5 seconds)",
    "fast": "Tests that execute quickly (<1 second)",
    "requires_api": "Tests that require external API access (usually mocked)",
    "requires_gpu": "Tests that require GPU/CUDA availability",
    "requires_internet": "Tests that require internet connectivity",
    "destructive": "Tests that modify the file system (run with extreme caution)",
    
    # Component-specific markers
    "gnn_processing": "Tests for GNN file processing functionality",
    "type_checking": "Tests for GNN type checking and validation",
    "visualization": "Tests for visualization generation",
    "export": "Tests for export functionality",
    "llm": "Tests for LLM integration",
    "mcp": "Tests for Model Context Protocol",
    "discopy": "Tests for DisCoPy integration",
    "utilities": "Tests for utility functions",
    "main_orchestrator": "Tests for main pipeline orchestrator",
    "setup": "Tests for environment setup and dependency management",
    "execution": "Tests for script execution and coordination",
    "ontology": "Tests for ontology processing",
    "rendering": "Tests for code generation and rendering",
    "sapf": "Tests for SAPF (Sound As Pure Form) audio generation functionality",
    
    # Quality and behavior markers
    "regression": "Regression tests to prevent breaking changes",
    "performance": "Performance and benchmarking tests",
    "memory": "Memory usage and leak detection tests",
    "concurrent": "Tests for concurrent execution scenarios"
}

def get_test_config() -> Dict[str, Any]:
    """
    Get the current test configuration.
    
    Returns:
        Dict containing all test configuration settings
        
    Example:
        config = get_test_config()
        assert config["safe_mode"] is True
        assert config["timeout_seconds"] == 60
    """
    return TEST_CONFIG.copy()

def update_test_config(**kwargs) -> None:
    """
    Update test configuration with new values.
    
    Args:
        **kwargs: Configuration key-value pairs to update
        
    Example:
        update_test_config(verbose_logging=True, timeout_seconds=120)
        config = get_test_config()
        assert config["verbose_logging"] is True
        assert config["timeout_seconds"] == 120
    """
    TEST_CONFIG.update(kwargs)

def is_safe_mode() -> bool:
    """
    Check if tests are running in safe mode.
    
    Returns:
        True if safe mode is enabled, False otherwise
        
    Example:
        if is_safe_mode():
            # Use mocked dependencies
            mock_subprocess = True
        else:
            # Use real dependencies (dangerous)
            mock_subprocess = False
    """
    return TEST_CONFIG.get("safe_mode", True)

def setup_test_environment() -> None:
    """
    Set up the test environment with necessary directories and configuration.
    
    This function:
    1. Creates all necessary temporary directories
    2. Sets environment variables for test mode
    3. Ensures safe execution context
    4. Initializes logging for tests
    
    Example:
        # This is typically called automatically by conftest.py
        setup_test_environment()
        assert os.environ.get("GNN_TEST_MODE") == "true"
    """
    # Create temp directories with error handling
    for dir_key in ["temp_output_dir", "test_data_dir", "sample_gnn_dir", "mock_project_root"]:
        dir_path = TEST_CONFIG.get(dir_key)
        if dir_path:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.warning(f"Failed to create test directory {dir_path}: {e}")
    
    # Set environment variables for testing
    os.environ["GNN_TEST_MODE"] = "true"
    os.environ["GNN_SAFE_MODE"] = str(TEST_CONFIG["safe_mode"]).lower()
    os.environ["GNN_MOCK_EXTERNAL_DEPS"] = str(TEST_CONFIG["mock_external_deps"]).lower()
    os.environ["GNN_MOCK_SUBPROCESS"] = str(TEST_CONFIG["mock_subprocess"]).lower()
    os.environ["GNN_MOCK_NETWORK_CALLS"] = str(TEST_CONFIG["mock_network_calls"]).lower()
    
    # Set test-specific paths
    if TEST_CONFIG.get("temp_output_dir"):
        os.environ["GNN_TEST_OUTPUT_DIR"] = str(TEST_CONFIG["temp_output_dir"])

def cleanup_test_environment() -> None:
    """
    Clean up test environment after test execution.
    
    This function:
    1. Removes temporary directories created during testing
    2. Cleans up environment variables
    3. Restores original system state
    
    Example:
        # This is typically called automatically by conftest.py
        cleanup_test_environment()
        assert "GNN_TEST_MODE" not in os.environ
    """
    # Clean up temporary directories with error handling
    for dir_key in ["temp_output_dir", "test_data_dir", "sample_gnn_dir", "mock_project_root"]:
        temp_dir = TEST_CONFIG.get(dir_key)
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logging.debug(f"Failed to clean up {temp_dir}: {e}")
    
    # Clean up environment variables
    test_env_vars = [
        "GNN_TEST_MODE", "GNN_SAFE_MODE", "GNN_MOCK_EXTERNAL_DEPS",
        "GNN_MOCK_SUBPROCESS", "GNN_MOCK_NETWORK_CALLS", "GNN_TEST_OUTPUT_DIR"
    ]
    for env_var in test_env_vars:
        os.environ.pop(env_var, None)

def create_sample_gnn_content() -> Dict[str, str]:
    """
    Create sample GNN file content for testing various scenarios.
    
    Returns:
        Dict mapping scenario names to GNN file content
        
    Example:
        samples = create_sample_gnn_content()
        valid_content = samples["valid_basic"]
        invalid_content = samples["invalid_syntax"]
        
        # Use in tests to create temporary GNN files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md') as f:
            f.write(valid_content)
            f.flush()
            # Test with f.name
    """
    return {
        "valid_basic": """# GNN Example: Basic Valid Model
## GNNVersionAndFlags
GNN v1

## ModelName
BasicValidModel

## StateSpaceBlock
x[2,type=float]      # Observable variable
y[3,type=float]      # Hidden variable

## Connections
x-y                  # Bidirectional connection

## InitialParameterization
x={0.0,0.0}
y={1.0,1.0,1.0}

## Equations
x = f(y)

## Time
Static

## Footer
BasicValidModel

## Signature
Test
""",
        
        "valid_complex": """# GNN Example: Complex Valid Model
## GNNVersionAndFlags
GNN v1

## ModelName
ComplexValidModel

## StateSpaceBlock
s_f0[4,type=categorical]     # Hidden state factor 0
s_f1[3,type=categorical]     # Hidden state factor 1
o_m0[2,type=categorical]     # Observation modality 0
u_c0[2,type=categorical]     # Control factor 0

## Connections
s_f0>o_m0               # State factor 0 influences observation modality 0
s_f1>o_m0               # State factor 1 influences observation modality 0
u_c0>s_f0               # Control factor 0 influences state factor 0
s_f0>s_f1               # State factor 0 influences state factor 1

## InitialParameterization
### Observation model (A matrix)
A_m0 = uniform([2,4,3])

### Transition model (B matrix)
B_f0 = uniform([4,4,2])
B_f1 = uniform([3,3])

### Prior preferences (C vector)
C_m0 = [0.5, -0.5]

### Prior beliefs (D vector)
D_f0 = uniform([4])
D_f1 = uniform([3])

## Equations
o_m0 ~ Cat(A_m0[:, s_f0, s_f1])
s_f0' ~ Cat(B_f0[:, s_f0, u_c0])
s_f1' ~ Cat(B_f1[:, s_f1])

## Time
Dynamic
DiscreteTime
ModelTimeHorizon=10

## Footer
ComplexValidModel

## Signature
Test
""",
        
        "invalid_syntax": """# GNN Example: Invalid Syntax
## GNNVersionAndFlags
InvalidVersion

## ModelName
InvalidSyntaxModel

## StateSpaceBlock
x[invalid_dimension]    # Invalid dimension specification
y                       # Missing dimension specification

## Connections
x>undefined_var         # Connection to undefined variable
invalid_connection      # Invalid connection format

## InitialParameterization
x = undefined_function() # Invalid function call
y = {1, 2, 3, 4, 5}     # Mismatched dimensions

## Equations
x = y +                 # Incomplete equation

## Time
InvalidTimeSpec

## Footer
InvalidSyntaxModel

## Signature
Test
""",
        
        "missing_sections": """# GNN Example: Missing Required Sections
## GNNVersionAndFlags
GNN v1

## ModelName
MissingSectionsModel

## StateSpaceBlock
x[2,type=float]

## Signature
Test
""",
        
        "empty_file": "",
        
        "malformed_headers": """# GNN Example: Malformed Headers
### GNNVersionAndFlags (Wrong level)
GNN v1

# ModelName (Wrong level)
MalformedModel

## StateSpaceBlock
x[2,type=float]

## Signature
Test
"""
    }

def create_test_gnn_files() -> Dict[str, Path]:
    """
    Create temporary GNN files for testing purposes.
    
    Returns:
        Dict mapping scenario names to file paths
        
    Example:
        test_files = create_test_gnn_files()
        valid_file = test_files["valid_basic"]
        invalid_file = test_files["invalid_syntax"]
        
        # Files are automatically cleaned up when test session ends
        assert valid_file.exists()
        assert invalid_file.exists()
    """
    sample_content = create_sample_gnn_content()
    test_files = {}
    
    sample_dir = TEST_CONFIG["sample_gnn_dir"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for scenario_name, content in sample_content.items():
        file_path = sample_dir / f"{scenario_name}.md"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            test_files[scenario_name] = file_path
        except Exception as e:
            logging.warning(f"Failed to create test file {file_path}: {e}")
    
    return test_files

def get_sample_pipeline_arguments() -> Dict[str, Any]:
    """
    Get sample pipeline arguments for testing.
    
    Returns:
        Dict containing sample arguments for pipeline execution
        
    Example:
        args = get_sample_pipeline_arguments()
        assert args["target_dir"] is not None
        assert args["output_dir"] is not None
        assert args["verbose"] is False  # Safe default
    """
    return {
        "target_dir": str(TEST_CONFIG["sample_gnn_dir"]),
        "output_dir": str(TEST_CONFIG["temp_output_dir"]),
        "recursive": False,
        "verbose": TEST_CONFIG["verbose_logging"],
        "strict": True,
        "estimate_resources": False,  # Safer for testing
        "skip_steps": [],
        "only_steps": [],
        "llm_tasks": ["summarize"],  # Safe default
        "llm_timeout": 30,  # Short timeout for tests
        "recreate_venv": False,  # Don't recreate in tests
        "dev": False  # Don't install dev dependencies in tests
    }

def validate_test_environment() -> Tuple[bool, List[str]]:
    """
    Validate that the test environment is properly configured.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
        
    Example:
        is_valid, issues = validate_test_environment()
        if not is_valid:
            pytest.skip(f"Test environment not ready: {issues}")
    """
    issues = []
    
    # Check that we're in test mode
    if not os.environ.get("GNN_TEST_MODE"):
        issues.append("GNN_TEST_MODE environment variable not set")
    
    # Check that safe mode is enabled
    if not is_safe_mode():
        issues.append("Safe mode is not enabled - tests may be destructive")
    
    # Check required directories exist
    for dir_key in ["temp_output_dir", "test_data_dir", "sample_gnn_dir"]:
        dir_path = TEST_CONFIG.get(dir_key)
        if not dir_path or not Path(dir_path).exists():
            issues.append(f"Required test directory missing: {dir_key}")
    
    # Check Python path includes src
    if str(SRC_DIR) not in sys.path:
        issues.append("Source directory not in Python path")
    
    return len(issues) == 0, issues

def get_mock_filesystem_structure() -> Dict[str, Any]:
    """
    Get a mock filesystem structure for testing filesystem operations.
    
    Returns:
        Dict representing a mock filesystem structure
        
    Example:
        mock_fs = get_mock_filesystem_structure()
        assert "src" in mock_fs
        assert "gnn" in mock_fs["src"]
        assert "examples" in mock_fs["src"]["gnn"]
    """
    return {
        "src": {
            "__init__.py": "",
            "main.py": "# Mock main.py content",
            "1_gnn.py": "# Mock pipeline step 1",
            "2_setup.py": "# Mock pipeline step 2",
            "gnn": {
                "__init__.py": "",
                "examples": {
                    "valid_model.md": create_sample_gnn_content()["valid_basic"],
                    "invalid_model.md": create_sample_gnn_content()["invalid_syntax"]
                }
            },
            "utils": {
                "__init__.py": "",
                "logging_utils.py": "# Mock logging utilities",
                "argument_utils.py": "# Mock argument utilities"
            },
            "tests": {
                "__init__.py": "",
                "conftest.py": "# Mock conftest",
                "test_environment.py": "# Mock environment tests"
            }
        },
        "output": {},
        "doc": {
            "README.md": "# Mock documentation"
        },
        "requirements.txt": "pytest\nmock\n"
    }

# Utility functions for test execution
def assert_safe_execution(func, *args, **kwargs):
    """
    Assert that a function executes safely without side effects.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
        
    Raises:
        AssertionError: If function appears to have unsafe side effects
        
    Example:
        def dangerous_function():
            os.system("rm -rf /")  # This would be caught
            
        # This would raise AssertionError
        assert_safe_execution(dangerous_function)
    """
    if not is_safe_mode():
        raise AssertionError("Cannot assert safe execution when not in safe mode")
    
    # Record initial system state
    initial_cwd = os.getcwd()
    initial_env = dict(os.environ)
    
    try:
        result = func(*args, **kwargs)
        
        # Check that system state wasn't modified unsafely
        if os.getcwd() != initial_cwd:
            raise AssertionError("Function changed working directory")
        
        # Check for new environment variables that suggest unsafe operations
        unsafe_env_changes = []
        for key, value in os.environ.items():
            if key not in initial_env and key.startswith(('PATH', 'LD_', 'DYLD_')):
                unsafe_env_changes.append(key)
        
        if unsafe_env_changes:
            raise AssertionError(f"Function set unsafe environment variables: {unsafe_env_changes}")
        
        return result
        
    except Exception as e:
        # Log the exception but don't re-raise unless it's our assertion
        if isinstance(e, AssertionError):
            raise
        logging.debug(f"Function {func.__name__} raised exception: {e}")
        raise

# Export key functions and classes for easy import
__all__ = [
    "TEST_CONFIG",
    "PYTEST_MARKERS", 
    "get_test_config",
    "update_test_config",
    "is_safe_mode",
    "setup_test_environment",
    "cleanup_test_environment",
    "create_sample_gnn_content",
    "create_test_gnn_files",
    "get_sample_pipeline_arguments",
    "validate_test_environment",
    "get_mock_filesystem_structure",
    "assert_safe_execution",
    "TEST_DIR",
    "SRC_DIR", 
    "PROJECT_ROOT"
] 