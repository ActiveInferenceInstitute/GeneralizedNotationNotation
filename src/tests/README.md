# Test Infrastructure Documentation

## Overview

The test infrastructure for the GNN Processing Pipeline provides comprehensive testing capabilities with a clean, module-based organization. All test files follow the `test_MODULENAME_area.py` naming convention for easy discovery and maintenance.

## Test File Structure

### Module-Based Naming Convention

All test files follow the pattern:
- `test_MODULENAME_overall.py` - Comprehensive module coverage
- `test_MODULENAME_area1.py` - Specific module areas
- `test_MODULENAME_area2.py` - Additional specialized areas

### Current Test Files

#### Core Module Tests
- `test_gnn_overall.py` - Comprehensive GNN module testing
- `test_gnn_parsing.py` - GNN parsing and discovery tests
- `test_gnn_validation.py` - GNN validation and type checking
- `test_gnn_processing.py` - GNN processing and transformation
- `test_gnn_integration.py` - GNN integration with other modules

- `test_render_overall.py` - Complete render module testing
- `test_render_integration.py` - Render integration tests
- `test_render_performance.py` - Render performance tests

- `test_mcp_overall.py` - Comprehensive MCP module testing
- `test_mcp_transport.py` - MCP transport layer tests
- `test_mcp_tools.py` - MCP tool registration and execution
- `test_mcp_integration.py` - MCP integration with other modules
- `test_mcp_performance.py` - MCP performance tests

- `test_audio_overall.py` - Comprehensive audio module testing
- `test_audio_sapf.py` - SAPF-specific tests
- `test_audio_generation.py` - Audio generation tests
- `test_audio_integration.py` - Audio integration tests

- `test_visualization_overall.py` - Comprehensive visualization testing
- `test_visualization_matrices.py` - Matrix visualization tests
- `test_visualization_ontology.py` - Ontology visualization tests

#### Infrastructure Module Tests
- `test_pipeline_overall.py` - Comprehensive pipeline testing
- `test_pipeline_steps.py` - Individual pipeline step tests
- `test_pipeline_orchestration.py` - Pipeline orchestration tests
- `test_pipeline_integration.py` - Pipeline integration tests

- `test_export_overall.py` - Comprehensive export testing

- `test_report_overall.py` - Comprehensive report testing
- `test_report_formats.py` - Report format tests
- `test_report_generation.py` - Report generation tests
- `test_report_integration.py` - Report integration tests

#### Environment and Supporting Tests
- `test_environment_overall.py` - Comprehensive environment testing
- `test_environment_python.py` - Python environment tests
- `test_environment_dependencies.py` - Dependency tests
- `test_environment_system.py` - System resource tests
- `test_environment_integration.py` - Environment integration tests

- `test_performance_overall.py` - Comprehensive performance testing
- `test_coverage_overall.py` - Code coverage tests
- `test_unit_overall.py` - Basic unit tests

## Test Utilities

### Centralized Test Utilities

All test utilities are centralized in `src/utils/test_utils.py` to maintain clean naming conventions and avoid circular imports.

#### Key Utilities Available

**Configuration and Setup:**
- `TEST_CONFIG` - Central test configuration
- `TEST_CATEGORIES` - Test category definitions
- `TEST_STAGES` - Test stage definitions
- `COVERAGE_TARGETS` - Coverage target definitions

**Environment Management:**
- `setup_test_environment()` - Initialize test environment
- `cleanup_test_environment()` - Clean up test environment
- `validate_test_environment()` - Validate test environment
- `is_safe_mode()` - Check if running in safe mode

**Test Data Generation:**
- `create_sample_gnn_content()` - Generate sample GNN content
- `create_test_gnn_files()` - Create test GNN files
- `create_test_files()` - Create generic test files
- `get_mock_filesystem_structure()` - Get mock filesystem structure

**Performance Tracking:**
- `performance_tracker()` - Performance tracking decorator
- `get_memory_usage()` - Get current memory usage
- `track_peak_memory()` - Track peak memory usage
- `with_resource_limits()` - Resource limit context manager

**Validation Functions:**
- `assert_file_exists()` - Assert file exists
- `assert_valid_json()` - Assert valid JSON
- `assert_directory_structure()` - Assert directory structure

**Report Generation:**
- `validate_report_data()` - Validate report data
- `generate_html_report_file()` - Generate HTML report
- `generate_markdown_report_file()` - Generate Markdown report
- `generate_json_report_file()` - Generate JSON report

## Test Execution

### Running Tests

#### Individual Test Files
```bash
# Run a specific test file
python -m pytest src/tests/test_gnn_overall.py

# Run with verbose output
python -m pytest src/tests/test_gnn_overall.py -v

# Run a specific test method
python -m pytest src/tests/test_gnn_overall.py::TestGNNCoreProcessor::test_core_processor_imports
```

#### Module-Based Testing
```bash
# Run all GNN tests
python -m pytest src/tests/test_gnn_*.py

# Run all render tests
python -m pytest src/tests/test_render_*.py

# Run all pipeline tests
python -m pytest src/tests/test_pipeline_*.py
```

#### Test Categories
```bash
# Run fast tests only
python -m pytest -m fast

# Run slow tests only
python -m pytest -m slow

# Run unit tests only
python -m pytest -m unit

# Run integration tests only
python -m pytest -m integration

# Run performance tests only
python -m pytest -m performance
```

#### Full Test Suite
```bash
# Run all tests
python -m pytest src/tests/

# Run with coverage
python -m pytest src/tests/ --cov=src

# Run with parallel execution
python -m pytest src/tests/ -n auto
```

### Test Runner

The test runner (`src/tests/runner.py`) provides orchestrated test execution with:

- **Modular Categories**: Organized test categories by module
- **Parallel Execution**: Support for parallel test execution
- **Timeout Management**: Configurable timeouts per test category
- **Failure Limits**: Maximum failure limits per category
- **Performance Tracking**: Built-in performance monitoring

#### Runner Configuration

The `MODULAR_TEST_CATEGORIES` dictionary defines test categories:

```python
MODULAR_TEST_CATEGORIES = {
    "gnn": {
        "name": "GNN Module Tests",
        "description": "GNN processing and validation tests",
        "files": ["test_gnn_overall.py", "test_gnn_parsing.py", "test_gnn_validation.py", 
                  "test_gnn_processing.py", "test_gnn_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    # ... additional categories
}
```

## Test Fixtures

### Core Fixtures

**Session-Level Fixtures:**
- `test_config` - Test configuration for entire session
- `project_root` - Project root directory
- `src_dir` - Source directory
- `test_dir` - Test directory

**Utility Fixtures:**
- `safe_filesystem` - Safe filesystem operations
- `mock_subprocess` - Mock subprocess execution
- `mock_imports` - Safe module imports
- `mock_logger` - Mock logger for capturing logs

**Test Data Fixtures:**
- `sample_gnn_files` - Sample GNN files for testing
- `isolated_temp_dir` - Isolated temporary directory
- `pipeline_arguments` - Standard pipeline arguments
- `comprehensive_test_data` - Complete test data set

### Using Fixtures

```python
def test_gnn_processing(sample_gnn_files, isolated_temp_dir):
    """Test GNN processing with sample files."""
    # Use sample_gnn_files for input
    # Use isolated_temp_dir for output
    pass

def test_pipeline_execution(mock_subprocess, mock_logger):
    """Test pipeline execution with mocks."""
    # Use mock_subprocess to avoid real subprocess calls
    # Use mock_logger to capture and verify logs
    pass
```

## Test Markers

### Available Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.fast` - Fast tests
- `@pytest.mark.safe_to_fail` - Tests safe to fail
- `@pytest.mark.destructive` - Tests that may modify system state
- `@pytest.mark.external` - Tests requiring external dependencies
- `@pytest.mark.core` - Core module tests
- `@pytest.mark.pipeline` - Pipeline infrastructure tests
- `@pytest.mark.recovery` - Pipeline recovery tests
- `@pytest.mark.utilities` - Utility function tests
- `@pytest.mark.environment` - Environment validation tests
- `@pytest.mark.render` - Rendering and code generation tests
- `@pytest.mark.export` - Export functionality tests
- `@pytest.mark.parsers` - Parser and format tests
- `@pytest.mark.main_orchestrator` - Main orchestrator tests
- `@pytest.mark.type_checking` - Type checking tests
- `@pytest.mark.mcp` - Model Context Protocol tests
- `@pytest.mark.sapf` - SAPF audio generation tests
- `@pytest.mark.visualization` - Visualization tests

### Using Markers

```python
@pytest.mark.unit
def test_basic_functionality():
    """Basic unit test."""
    pass

@pytest.mark.integration
def test_module_integration():
    """Integration test."""
    pass

@pytest.mark.performance
def test_performance_characteristics():
    """Performance test."""
    pass

@pytest.mark.slow
def test_comprehensive_workflow():
    """Slow comprehensive test."""
    pass
```

## Test Configuration

### Environment Variables

- `PYTHONPATH=src` - Add src to Python path for imports
- `PYTEST_ADDOPTS` - Additional pytest options
- `TEST_SAFE_MODE` - Enable safe mode for tests

### Configuration Files

- `conftest.py` - Pytest configuration and fixtures
- `runner.py` - Test runner configuration
- `utils/test_utils.py` - Centralized test utilities

## Best Practices

### Test Organization

1. **Module-Based Structure**: Organize tests by module with clear naming
2. **Comprehensive Coverage**: Use `_overall.py` files for complete module coverage
3. **Specialized Areas**: Use `_area.py` files for specific testing areas
4. **Clean Imports**: Import from `utils.test_utils` for shared utilities

### Test Writing

1. **Descriptive Names**: Use clear, descriptive test and method names
2. **Proper Fixtures**: Use appropriate fixtures for test data and mocks
3. **Safe Execution**: Use safe mocks to avoid side effects
4. **Clear Assertions**: Use clear, specific assertions
5. **Error Handling**: Test both success and failure scenarios

### Performance Considerations

1. **Fast Tests**: Mark quick tests with `@pytest.mark.fast`
2. **Slow Tests**: Mark time-consuming tests with `@pytest.mark.slow`
3. **Parallel Execution**: Use parallel execution for independent tests
4. **Resource Limits**: Use resource limits for memory-intensive tests

### Maintenance

1. **Regular Updates**: Keep test utilities and fixtures up to date
2. **Documentation**: Document complex test scenarios and utilities
3. **Coverage**: Maintain good test coverage across all modules
4. **Validation**: Regularly validate test environment and dependencies

## Troubleshooting

### Common Issues

**Import Errors:**
- Ensure `PYTHONPATH=src` is set
- Check that modules exist in the expected locations
- Verify import paths in test files

**Test Discovery Issues:**
- Check file naming follows `test_*.py` pattern
- Verify test methods follow `test_*` naming
- Ensure proper class inheritance from `unittest.TestCase`

**Fixture Issues:**
- Check fixture dependencies are available
- Verify fixture scope matches usage
- Ensure proper cleanup in fixtures

**Performance Issues:**
- Use `@pytest.mark.slow` for time-consuming tests
- Implement proper mocking for external dependencies
- Use parallel execution where appropriate

### Debugging

```bash
# Run with verbose output
python -m pytest -v

# Run with debug output
python -m pytest --tb=long

# Run with print statements
python -m pytest -s

# Run specific test with full output
python -m pytest test_file.py::test_method -v -s
```

## Contributing

### Adding New Tests

1. **Follow Naming Convention**: Use `test_MODULENAME_area.py` pattern
2. **Import Utilities**: Import from `utils.test_utils` for shared functionality
3. **Use Appropriate Fixtures**: Leverage existing fixtures for common patterns
4. **Add Markers**: Use appropriate markers for test categorization
5. **Update Documentation**: Document new test patterns and utilities

### Adding New Fixtures

1. **Place in conftest.py**: Add shared fixtures to `conftest.py`
2. **Use Proper Scope**: Choose appropriate fixture scope (function, class, module, session)
3. **Provide Documentation**: Document fixture purpose and usage
4. **Handle Cleanup**: Ensure proper cleanup for resource-intensive fixtures

### Adding New Utilities

1. **Place in utils/test_utils.py**: Add shared utilities to the centralized location
2. **Follow Naming Convention**: Use clear, descriptive names
3. **Add Type Hints**: Include proper type hints for better IDE support
4. **Provide Documentation**: Document utility purpose and parameters
5. **Update Exports**: Add to `__init__.py` exports if needed

## Summary

The test infrastructure provides:

- **Clean Organization**: Module-based file structure with clear naming
- **Comprehensive Coverage**: Complete test coverage across all modules
- **Centralized Utilities**: Shared test utilities in `utils/test_utils.py`
- **Flexible Execution**: Multiple ways to run tests (individual, module, category)
- **Robust Fixtures**: Comprehensive fixture system for test data and mocks
- **Performance Optimization**: Parallel execution and resource management
- **Maintainable Architecture**: Clear patterns and documentation

This infrastructure supports the full testing lifecycle from development through deployment, ensuring code quality and reliability across the GNN Processing Pipeline. 