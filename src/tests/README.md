# GNN Pipeline Test Architecture

This document provides an overview of the comprehensive test architecture for the GeneralizedNotationNotation pipeline.

## Test Framework Architecture

The test framework is organized as a modular, maintainable system with these key components:

### Core Components

- **Test Utils (`test_utils.py`)**: Central utility functions shared across all test modules
- **Configuration (`conftest.py`)**: Pytest fixtures and configuration
- **Test Categories**: Fast, Standard, Slow, and Performance tests
- **Modular Organization**: Tests are organized by module and functionality

### Test Categories

Tests are organized into categories that can be run separately or together:

1. **Fast Tests** (`-m fast`): Quick validation tests that take < 1s each
2. **Standard Tests** (`-m standard` or `-m "not slow and not performance"`): Integration and validation tests that take < 10s each
3. **Slow Tests** (`-m slow`): Complex tests that may take longer to run
4. **Performance Tests** (`-m performance`): Resource usage and scalability tests

### Key Principles

- **Real Implementations**: No mock methods - all tests use real implementations or gracefully degrade
- **Safe-to-Fail**: Tests are designed to gracefully handle failures
- **Resource Monitoring**: CPU, memory, and disk usage are monitored during tests
- **Comprehensive Coverage**: Tests cover all pipeline components
- **Modular Design**: Test infrastructure is modular and easy to extend

## Test Infrastructure

### Test Utilities

The central `test_utils.py` module provides:

- Constants and configuration
- GNN file and test data generation
- Validation utilities
- Report generation for tests
- MCP integration test helpers

### Test Fixtures

Comprehensive fixtures are available in `conftest.py`:

- **Environment Fixtures**: Project paths, directories, isolation
- **Mock Components**: Safe subprocess, filesystem, logger implementations
- **Real Components**: GNN parser, type checker, visualization, export functionality
- **Test Data**: Sample GNN files, pipeline data, comprehensive test datasets
- **MCP Integration**: Mock MCP tools for testing MCP integration

### Resource Monitoring

Tests include comprehensive resource monitoring:

- Memory usage tracking
- CPU utilization monitoring
- Disk usage monitoring
- Execution time tracking
- Graceful timeout handling

## Running Tests

### Basic Test Execution

```bash
# Run fast tests only
python -m pytest src/tests/ -m fast

# Run standard tests (excluding slow and performance)
python -m pytest src/tests/ -m "not slow and not performance"

# Run all tests including slow ones
python -m pytest src/tests/ -m "not performance"

# Run specific test module
python -m pytest src/tests/test_gnn_core_modules.py

# Run with coverage
python -m pytest src/tests/ --cov=src/
```

### Using the Pipeline Runner

The test suite can also be executed through the pipeline's test runner:

```bash
# Run the test step via the pipeline
python src/main.py --only-steps 2

# Run tests with specific configuration
python src/2_tests.py --include-slow --verbose
```

## Test Categories and File Structure

### Core Module Tests

- `test_gnn_core_modules.py`: Tests for GNN core functionality
- `test_report_comprehensive.py`: Tests for report generation
- `test_pipeline_infrastructure.py`: Tests for pipeline infrastructure

### Integration Tests

- `test_mcp_integration_comprehensive.py`: Tests for MCP integration
- `test_pipeline_steps.py`: Tests for pipeline step functionality
- `test_integration_tests.py`: Cross-module integration tests

### Performance Tests

- `test_performance_tests.py`: Real performance regression tests
- `test_pipeline_performance.py`: Pipeline performance tests

### Environment and Setup Tests

- `test_environment.py`: Testing environment validation
- `test_setup.py`: Setup process validation

## Extending the Test Suite

To add new tests:

1. Choose the appropriate test file or create a new one
2. Use pytest fixtures from `conftest.py` to access test data and components
3. Import utility functions from `test_utils.py`
4. Mark tests appropriately (`@pytest.mark.fast`, `@pytest.mark.standard`, etc.)
5. Follow the existing patterns for assertions and setup

## Continuous Integration

The test suite is designed to work with CI/CD systems:

- Fast tests run on every commit
- Standard tests run on pull requests
- Slow and performance tests run nightly
- Test reports are generated and stored as artifacts

## Performance Monitoring

Performance regression testing includes:

- Memory usage tracking
- CPU utilization monitoring 
- Execution time tracking
- Import time analysis
- Algorithm performance benchmarking

## Best Practices

1. Always use real implementations rather than mocks
2. Add proper markers to tests based on execution time
3. Use fixtures for test data generation
4. Monitor resource usage in performance-critical tests
5. Ensure all tests can run independently
6. Follow the modular organization pattern 