# GNN Pipeline Test Suite

## Overview

This test suite provides comprehensive testing for the GNN (Generalized Notation Notation) processing pipeline. The test suite is designed to be:

- **Comprehensive**: Covers all 14 pipeline steps plus utilities and orchestration
- **Safe-to-fail**: Uses extensive mocking to prevent destructive operations
- **Fast by default**: Runs all tests except slow ones in normal execution
- **Well-organized**: Clear categorization with pytest markers
- **Maintainable**: Modular design with reusable fixtures

## Test Execution

### Default Behavior (Recommended)
```bash
# Run all tests except slow ones (default)
python -m pytest src/tests/

# Run via pipeline step 3
python src/3_tests.py --target-dir input/gnn_files --output-dir output/
```

### Test Categories

#### Fast Tests (145 tests)
```bash
# Run only fast tests
python -m pytest src/tests/ -m "fast"

# Run fast tests via pipeline
python src/3_tests.py --target-dir input/gnn_files --output-dir output/ --fast-only
```

#### All Tests Including Slow (429 tests)
```bash
# Run all tests including slow ones
python -m pytest src/tests/ -m "not slow" --include-slow

# Run all tests via pipeline
python src/3_tests.py --target-dir input/gnn_files --output-dir output/ --include-slow
```

#### Specific Test Categories
```bash
# Environment and setup tests
python -m pytest src/tests/ -m "environment"

# Core module tests
python -m pytest src/tests/ -m "core"

# Utility function tests
python -m pytest src/tests/ -m "utilities"

# Pipeline step tests
python -m pytest src/tests/ -m "pipeline"

# Integration tests
python -m pytest src/tests/ -m "integration"
```

## Test Architecture

### Test Files Structure
```
src/tests/
├── conftest.py              # Pytest configuration and fixtures
├── runner.py                # Test execution orchestration
├── __init__.py              # Test utilities and configuration
├── test_fast_suite.py       # Fast test suite (30 tests)
├── test_environment.py      # Environment validation (fast)
├── test_utilities.py        # Utility function tests (fast)
├── test_core_modules.py     # Core module tests (fast)
├── test_export.py           # Export functionality tests (fast)
├── test_gnn_type_checker.py # Type checker tests (fast)
├── test_pipeline_steps.py   # Pipeline step tests
├── test_main_orchestrator.py # Main orchestrator tests
├── test_mcp_comprehensive.py # MCP integration tests
├── test_render.py           # Code generation tests
├── test_sapf.py             # Audio generation tests
└── test_comprehensive_api.py # API comprehensive tests
```

### Test Markers

#### Execution Categories
- `@pytest.mark.fast`: Tests that execute quickly (<1 second)
- `@pytest.mark.slow`: Tests that take longer to execute (>5 seconds)
- `@pytest.mark.safe_to_fail`: Tests designed to fail gracefully with mocked dependencies

#### Component Categories
- `@pytest.mark.environment`: Environment setup and validation tests
- `@pytest.mark.core`: Core module functionality tests
- `@pytest.mark.utilities`: Utility function tests
- `@pytest.mark.pipeline`: Pipeline step execution tests
- `@pytest.mark.integration`: Cross-component integration tests
- `@pytest.mark.export`: Export functionality tests
- `@pytest.mark.type_checking`: Type checking and validation tests
- `@pytest.mark.rendering`: Code generation tests
- `@pytest.mark.sapf`: Audio generation tests
- `@pytest.mark.mcp`: Model Context Protocol tests

#### Quality Categories
- `@pytest.mark.unit`: Unit tests for individual components
- `@pytest.mark.regression`: Regression tests to prevent breaking changes
- `@pytest.mark.performance`: Performance and benchmarking tests

## Test Configuration

### Default Settings
```python
TEST_CONFIG = {
    "safe_mode": True,                    # Enable safe-to-fail mode
    "mock_external_deps": True,           # Mock external dependencies
    "timeout_seconds": 30,                # Test timeout
    "max_test_files": 5,                  # Max GNN files per test
    "enable_performance_tracking": False, # Disable for speed
    "verbose_logging": False,             # Quiet logging
    "coverage_threshold": 60.0,           # Coverage target
}
```

### Environment Variables
- `GNN_TEST_MODE`: Set to "true" during testing
- `GNN_SAFE_MODE`: Controls safe-to-fail behavior
- `GNN_MOCK_EXTERNAL_DEPS`: Controls dependency mocking
- `GNN_MOCK_SUBPROCESS`: Controls subprocess mocking

## Fixtures

### Core Fixtures
- `isolated_temp_dir`: Provides isolated temporary directory
- `sample_gnn_files`: Creates sample GNN files for testing
- `safe_filesystem`: Safe filesystem operations
- `mock_subprocess`: Mocked subprocess execution
- `mock_llm_provider`: Mocked LLM provider
- `test_logger`: Test logger instance

### Component Fixtures
- `real_gnn_parser`: Real GNN parser for testing
- `real_type_checker`: Real type checker for testing
- `real_visualization`: Real visualization components
- `real_export`: Real export functionality

## Test Development Guidelines

### Adding New Tests

1. **Use appropriate markers**:
   ```python
   @pytest.mark.unit
   @pytest.mark.safe_to_fail
   @pytest.mark.fast  # If test is fast
   def test_new_functionality():
       # Test implementation
   ```

2. **Follow naming conventions**:
   - Test classes: `Test<Component>Comprehensive`
   - Test methods: `test_<functionality>_<scenario>`

3. **Use fixtures for dependencies**:
   ```python
   def test_with_dependencies(isolated_temp_dir, sample_gnn_files):
       # Test implementation
   ```

4. **Handle missing dependencies gracefully**:
   ```python
   try:
       from some_module import some_function
   except ImportError:
       pytest.skip("some_module not available")
   ```

### Test Categories

#### Unit Tests
- Test individual functions and methods
- Use extensive mocking for dependencies
- Focus on specific functionality
- Should be fast and isolated

#### Integration Tests
- Test component interactions
- Use controlled test environments
- Focus on data flow between components
- May be slower but more comprehensive

#### Environment Tests
- Test system setup and configuration
- Validate dependency availability
- Check resource requirements
- Essential for deployment validation

## Performance Considerations

### Test Execution Times
- **Fast tests**: <1 second each
- **Regular tests**: 1-5 seconds each
- **Slow tests**: >5 seconds each

### Optimization Strategies
- Use `@pytest.mark.fast` for quick tests
- Mock expensive operations
- Use `isolated_temp_dir` for file operations
- Disable performance tracking in tests

## Coverage and Quality

### Coverage Targets
- **Overall coverage**: 60% minimum
- **Core modules**: 80% minimum
- **Utilities**: 90% minimum

### Quality Metrics
- All tests must be safe-to-fail
- No destructive operations without mocking
- Comprehensive error handling
- Clear test documentation

## Troubleshooting

### Common Issues

#### JAX Logging Errors
```
ValueError: I/O operation on closed file
```
- **Solution**: JAX cleanup is patched in conftest.py
- **Prevention**: Use `patch_jax_cleanup` fixture

#### Import Errors
```
ImportError: No module named 'some_module'
```
- **Solution**: Use try/except with pytest.skip()
- **Prevention**: Mock dependencies or skip gracefully

#### Test Timeouts
```
pytest.TimeoutError: Test execution timed out
```
- **Solution**: Increase timeout or optimize test
- **Prevention**: Use fast markers for quick tests

### Debug Mode
```bash
# Run with verbose output
python -m pytest src/tests/ -v

# Run with detailed error reporting
python -m pytest src/tests/ --tb=long

# Run specific test with debugging
python -m pytest src/tests/test_specific.py::TestClass::test_method -v -s
```

## Recent Improvements

### Test Execution Defaults
- **Before**: Only 30 fast tests ran by default (7% coverage)
- **After**: 418 tests run by default (97% coverage)
- **Improvement**: 13x increase in default test coverage

### Test Marking
- Added `@pytest.mark.fast` to 115 additional tests
- Improved test categorization and organization
- Enhanced test discovery and execution

### JAX Compatibility
- Fixed JAX logging conflicts during test collection
- Added graceful handling for missing JAX
- Improved test environment stability

### Configuration Updates
- Changed default `fast_only=False` in test runner
- Updated test selection logic for better coverage
- Improved timeout handling for different test types

## Future Improvements

### Planned Enhancements
1. **Parallel test execution** for faster feedback
2. **Test result caching** to avoid redundant work
3. **Automated test categorization** based on execution time
4. **Enhanced coverage reporting** with detailed metrics
5. **Test performance profiling** to identify bottlenecks

### Test Expansion
1. **More integration tests** for complex workflows
2. **Performance regression tests** for critical paths
3. **Stress tests** for large-scale operations
4. **Compatibility tests** for different environments

## Contributing

When contributing to the test suite:

1. **Follow existing patterns** and conventions
2. **Add appropriate markers** for test categorization
3. **Ensure tests are safe-to-fail** with proper mocking
4. **Document test purpose** and expected behavior
5. **Maintain test isolation** and independence
6. **Update this documentation** for significant changes

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [GNN Pipeline Architecture](../pipeline/PIPELINE_ARCHITECTURE.md)
- [Test Configuration](../testing/README.md)
- [Development Guidelines](../development/README.md) 