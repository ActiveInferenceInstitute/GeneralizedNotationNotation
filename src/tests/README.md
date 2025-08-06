# GNN Testing Framework

## Overview

This directory contains a comprehensive testing framework for the GNN (Generalized Notation Notation) Processing Pipeline. The framework includes unit tests, integration tests, performance tests, and coverage analysis.

## Current Status

### ✅ Working Components
- **Test Infrastructure**: Basic test discovery and execution is working
- **Test Utilities**: `test_utils.py` provides shared utilities and fixtures
- **Test Configuration**: `conftest.py` provides pytest fixtures and configuration
- **Core Test Files**: Several test files are functional and can be run individually

### ⚠️ Known Issues
- **Import Path Issues**: Many test files use incorrect import paths (e.g., `from gnn import` instead of `from src.gnn import`)
- **Plugin Conflicts**: pytest plugins (pytest-sugar, pytest-randomly) cause recursion errors
- **Module Dependencies**: Some tests depend on modules that may not be fully implemented

## Test Categories

### 1. Core Module Tests (`test_core_modules.py`)
Tests for essential GNN functionality:
- GNN module imports and basic functionality
- Render module for code generation
- Execute module for script execution
- LLM module for AI-assisted analysis
- MCP module for Model Context Protocol
- Ontology module for semantic processing
- Website module for HTML generation
- SAPF module for audio generation

### 2. GNN Core Modules (`test_gnn_core_modules.py`)
Tests for GNN-specific functionality:
- Core processor for GNN file processing
- Discovery module for file finding
- Reporting module for result generation
- Validation module for structure checking
- Simple validator for basic validation
- Parsers and serializers for format handling

### 3. Pipeline Infrastructure (`test_pipeline_infrastructure.py`)
Tests for pipeline orchestration:
- Pipeline discovery and validation
- Step template generation
- Resource management
- Script validation

### 4. Utilities (`test_utilities.py`)
Tests for shared utilities:
- Argument parsing
- Logging setup
- Path utilities
- Performance tracking
- Dependency validation

### 5. Fast Suite (`test_fast_suite.py`)
Quick validation tests for rapid feedback:
- Environment validation
- Basic module imports
- Core functionality checks

## Running Tests

### Individual Test Files
```bash
# Run a specific test file
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/tests/test_utilities.py -v -p no:randomly -p no:sugar -p no:cacheprovider

# Run a specific test class
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/tests/test_utilities.py::TestArgumentParsing -v -p no:randomly -p no:sugar -p no:cacheprovider

# Run a specific test method
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/tests/test_utilities.py::TestArgumentParsing::test_enhanced_argument_parser_import -v -p no:randomly -p no:sugar -p no:cacheprovider
```

### Using the Test Runner
```bash
# Run all tests through the pipeline test runner
python3 src/2_tests.py --target-dir input/gnn_files --output-dir output --verbose

# Run only fast tests
python3 src/2_tests.py --target-dir input/gnn_files --output-dir output --fast-only --verbose
```

### Test Categories
The test runner organizes tests into categories:
- **Core**: Essential GNN functionality tests
- **Pipeline**: Pipeline infrastructure tests
- **Validation**: Data validation tests
- **Utilities**: Shared utility tests
- **Reporting**: Report generation tests
- **Fast Suite**: Quick validation tests
- **Comprehensive**: Full integration tests

## Test Configuration

### Environment Variables
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: Disables problematic pytest plugins
- `PYTHONPATH`: Should include `src/` directory

### Pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
addopts = -p no:randomly -p no:sugar -p no:cacheprovider
testpaths = src/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests for individual components
    integration: Integration tests for component interactions
    performance: Performance and resource usage tests
    slow: Tests that take significant time to complete
    fast: Quick tests for rapid feedback
    safe_to_fail: Tests safe to run without side effects
```

## Test Fixtures

### Core Fixtures (conftest.py)
- `test_config`: Test configuration for the session
- `project_root`: Path to project root
- `src_dir`: Path to source directory
- `test_dir`: Path to test directory
- `isolated_temp_dir`: Temporary directory for tests
- `sample_gnn_files`: Sample GNN files for testing
- `comprehensive_test_data`: Comprehensive test data

### Mock Fixtures
- `mock_subprocess`: Safe subprocess execution
- `mock_filesystem`: Safe filesystem operations
- `mock_imports`: Safe module imports
- `mock_dangerous_operations`: Safe dangerous operations
- `mock_llm_provider`: Mock LLM provider
- `mock_logger`: Mock logging

## Test Utilities

### Core Functions (test_utils.py)
- `is_safe_mode()`: Check if running in safe mode
- `validate_test_environment()`: Validate test environment
- `create_test_gnn_files()`: Create test GNN files
- `run_all_tests()`: Run comprehensive test suite
- `performance_tracker()`: Track performance metrics
- `with_resource_limits()`: Apply resource limits

### Test Categories
```python
TEST_CATEGORIES = {
    "fast": "Quick validation tests for core functionality",
    "standard": "Integration tests and moderate complexity", 
    "slow": "Complex scenarios and benchmarks",
    "performance": "Resource usage and scalability tests",
    "safe_to_fail": "Tests with graceful degradation"
}
```

## Fixing Import Issues

### Current Import Problems
Many test files use incorrect import paths. For example:
```python
# ❌ Incorrect
from gnn import discover_gnn_files
from render import render_gnn_to_pymdp

# ✅ Correct
from src.gnn import discover_gnn_files
from src.render import render_gnn_to_pymdp
```

### Files Needing Import Fixes
1. `test_core_modules.py` - ✅ Fixed
2. `test_gnn_core_modules.py` - ✅ Fixed
3. `test_pipeline_scripts.py` - Needs fixing
4. `test_pipeline_performance.py` - Needs fixing
5. `test_mcp_integration_comprehensive.py` - Needs fixing
6. `test_environment.py` - Needs fixing
7. `test_fast_suite.py` - Needs fixing
8. `test_main_orchestrator.py` - Needs fixing
9. `test_utility_modules.py` - Needs fixing
10. `test_sapf.py` - Needs fixing

### Systematic Fix Process
1. Find all `from gnn import` statements
2. Replace with `from src.gnn import`
3. Find all `from render import` statements
4. Replace with `from src.render import`
5. Continue for all module imports
6. Test each file individually
7. Run comprehensive test suite

## Test Markers

### Available Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.fast`: Fast tests
- `@pytest.mark.safe_to_fail`: Safe to fail tests
- `@pytest.mark.type_checking`: Type checking tests
- `@pytest.mark.mcp`: MCP tests
- `@pytest.mark.sapf`: SAPF tests
- `@pytest.mark.visualization`: Visualization tests

### Running Tests by Marker
```bash
# Run only unit tests
python3 -m pytest -m unit

# Run only fast tests
python3 -m pytest -m fast

# Run integration tests
python3 -m pytest -m integration

# Exclude slow tests
python3 -m pytest -m "not slow"
```

## Performance Testing

### Memory Tracking
```python
from src.tests.test_utils import track_peak_memory, get_memory_usage

@track_peak_memory
def test_memory_intensive_operation():
    # Test code here
    pass
```

### Resource Limits
```python
from src.tests.test_utils import with_resource_limits

@with_resource_limits(max_memory_mb=100, max_cpu_percent=50)
def test_with_limits():
    # Test code here
    pass
```

## Coverage Analysis

### Running Coverage
```bash
# Run with coverage
python3 -m pytest --cov=src --cov-report=html --cov-report=term

# Generate coverage report
coverage run -m pytest src/tests/
coverage report
coverage html
```

### Coverage Targets
- Overall: 85%
- Unit tests: 90%
- Integration tests: 80%
- Performance tests: 70%

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Check import paths (should use `src.module` not `module`)
   - Ensure `src/` is in Python path
   - Verify module exists and is properly structured

2. **Plugin Conflicts**
   - Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
   - Disable problematic plugins: `-p no:randomly -p no:sugar -p no:cacheprovider`

3. **Recursion Errors**
   - Usually caused by pytest plugins
   - Use `--tb=short` for shorter tracebacks
   - Run individual tests instead of full suite

4. **Memory Issues**
   - Use `@with_resource_limits` decorator
   - Monitor memory usage with `track_peak_memory`
   - Clean up resources in test teardown

### Debugging Tests
```bash
# Run with maximum verbosity
python3 -m pytest -vvv --tb=long

# Run with print statements
python3 -m pytest -s

# Run with debugger
python3 -m pytest --pdb
```

## Next Steps

### Immediate Actions
1. ✅ Fix import paths in `test_core_modules.py`
2. ✅ Fix import paths in `test_gnn_core_modules.py`
3. 🔄 Fix import paths in remaining test files
4. 🔄 Test each file individually
5. 🔄 Run comprehensive test suite
6. 🔄 Document any remaining issues

### Long-term Improvements
1. Add more comprehensive test coverage
2. Implement automated test discovery
3. Add performance regression testing
4. Implement continuous integration
5. Add test result reporting
6. Implement test parallelization

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Use appropriate test markers
3. Include proper error handling
4. Add comprehensive docstrings
5. Use the provided fixtures
6. Follow the safe-to-fail pattern for integration tests

## Contact

For issues with the testing framework, check the test logs and refer to this documentation. The testing framework is designed to be robust and provide clear feedback about what's working and what needs attention. 