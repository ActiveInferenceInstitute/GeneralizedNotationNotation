# Tests Module - Agent Scaffolding

## Module Overview

**Purpose**: Comprehensive test suite execution and validation for the GNN processing pipeline

**Pipeline Step**: Step 2: Test suite execution (2_tests.py)

**Category**: Testing / Quality Assurance

---

## Core Functionality

### Primary Responsibilities
1. Comprehensive test suite execution
2. Test result collection and analysis
3. Coverage analysis and reporting
4. Performance testing and benchmarking
5. Test environment management and validation

### Key Capabilities
- Multi-level test execution (unit, integration, performance)
- Comprehensive test reporting and analysis
- Coverage analysis and optimization
- Performance benchmarking and profiling
- Test environment validation and setup

---

## API Reference

### Public Functions

#### `run_tests(logger, output_dir, verbose=False, **kwargs) -> bool`
**Description**: Run comprehensive test suite

**Parameters**:
- `logger`: Logger instance
- `output_dir`: Output directory for test results
- `verbose`: Enable verbose output
- `**kwargs`: Additional test options

**Returns**: `True` if tests passed

#### `run_fast_tests(target_dir, output_dir, verbose) -> bool`
**Description**: Run fast test suite only

**Parameters**:
- `target_dir`: Target directory
- `output_dir`: Output directory
- `verbose`: Enable verbose output

**Returns**: `True` if fast tests passed

#### `run_standard_tests(target_dir, output_dir, verbose) -> bool`
**Description**: Run standard test suite

**Parameters**:
- `target_dir`: Target directory
- `output_dir`: Output directory
- `verbose`: Enable verbose output

**Returns**: `True` if standard tests passed

---

## Dependencies

### Required Dependencies
- `pytest` - Test framework
- `pytest-cov` - Coverage analysis
- `pathlib` - Path manipulation

### Optional Dependencies
- `pytest-xdist` - Parallel test execution
- `pytest-benchmark` - Performance benchmarking
- `pytest-html` - HTML test reports

### Internal Dependencies
- `utils.pipeline_template` - Pipeline utilities

---

## Configuration

### Test Settings
```python
TEST_CONFIG = {
    'fast_tests': True,
    'standard_tests': True,
    'slow_tests': False,
    'performance_tests': False,
    'coverage_analysis': True,
    'parallel_execution': True,
    'timeout': 300
}
```

### Test Categories
```python
TEST_CATEGORIES = {
    'unit': ['test_*unit*.py'],
    'integration': ['test_*integration*.py'],
    'performance': ['test_*performance*.py'],
    'slow': ['test_*slow*.py']
}
```

---

## Usage Examples

### Run Test Suite
```python
from tests.runner import run_tests

success = run_tests(
    logger=logger,
    output_dir=Path("output/2_tests_output"),
    verbose=True,
    comprehensive=True
)
```

### Run Fast Tests Only
```python
from tests.runner import run_fast_tests

success = run_fast_tests(
    target_dir="src/",
    output_dir="output/tests_fast",
    verbose=True
)
```

### Run Specific Test Category
```python
from tests.runner import run_test_category

success = run_test_category(
    category="unit",
    target_dir="src/",
    output_dir="output/tests_unit",
    verbose=True
)
```

---

## Output Specification

### Output Products
- `test_results.json` - Test execution results
- `coverage.xml` - Coverage analysis report
- `test_report.html` - HTML test report
- `performance_report.json` - Performance analysis
- `test_summary.md` - Human-readable test summary

### Output Directory Structure
```
output/2_tests_output/
├── test_results.json
├── coverage.xml
├── test_report.html
├── performance_report.json
├── test_summary.md
└── test_details/
    ├── unit_tests/
    ├── integration_tests/
    └── performance_tests/
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: ~5-15 minutes for comprehensive suite
- **Memory**: ~100-300MB during test execution
- **Status**: ✅ Production Ready

### Expected Performance
- **Fast Tests**: 1-3 minutes
- **Standard Tests**: 3-8 minutes
- **Slow Tests**: 5-15 minutes
- **Performance Tests**: 10-30 minutes

---

## Error Handling

### Test Errors
1. **Test Failures**: Individual test case failures
2. **Setup Errors**: Test environment setup failures
3. **Dependency Errors**: Missing test dependencies
4. **Timeout Errors**: Test execution timeouts
5. **Coverage Errors**: Coverage analysis failures

### Recovery Strategies
- **Test Isolation**: Run tests in isolation on failure
- **Environment Reset**: Reset test environment
- **Dependency Installation**: Install missing dependencies
- **Timeout Adjustment**: Adjust test timeouts
- **Error Reporting**: Comprehensive error documentation

---

## Integration Points

### Orchestrated By
- **Script**: `2_tests.py` (Step 2)
- **Function**: `run_tests()`

### Imports From
- `utils.pipeline_template` - Pipeline utilities

### Imported By
- `main.py` - Pipeline orchestration
- `tests.test_*` - Individual test modules

### Data Flow
```
Test Discovery → Environment Setup → Test Execution → Result Collection → Report Generation
```

---

## Testing

### Test Files
- `src/tests/test_runner.py` - Test runner functionality
- `src/tests/test_conftest.py` - Test configuration
- Various test files for individual modules

### Test Coverage
- **Current**: 95%
- **Target**: 98%+

### Key Test Scenarios
1. Test suite execution and management
2. Coverage analysis and reporting
3. Performance testing and benchmarking
4. Error handling and recovery

---

## MCP Integration

### Tools Registered
- `tests.run_suite` - Run test suite
- `tests.run_fast` - Run fast tests
- `tests.get_coverage` - Get coverage report
- `tests.get_performance` - Get performance metrics

### Tool Endpoints
```python
@mcp_tool("tests.run_suite")
def run_test_suite_tool(output_dir):
    """Run comprehensive test suite"""
    # Implementation
```

---

**Last Updated: October 28, 2025
**Status**: ✅ Production Ready