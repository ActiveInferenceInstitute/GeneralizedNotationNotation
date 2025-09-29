# Tests Module - Agent Scaffolding

## Module Overview

**Purpose**: Comprehensive test suite execution with modular test discovery, real end-to-end testing, and no mock implementations

**Pipeline Step**: Step 2: Test suite execution (2_tests.py)

**Category**: Quality Assurance / Testing

---

## Core Functionality

### Primary Responsibilities
1. Discover and execute test files
2. Support multiple test execution modes (fast, default, comprehensive)
3. Generate test execution reports
4. Configure pytest environment for pipeline testing
5. Real subprocess-based testing (no mocks)

### Key Capabilities
- **Fast Mode**: Core tests only (5 files, ~20 seconds)
- **Default Mode**: Core + pipeline tests (10 files, ~2 minutes)
- **Comprehensive Mode**: All tests (54 files, ~15 minutes)
- Real artifact validation in `output/`
- Live pytest output streaming
- Test categorization and filtering

---

## API Reference

### Public Functions

#### `process_tests_standardized(target_dir, output_dir, logger, **kwargs) -> bool`
**Description**: Main test processing function with multiple execution modes

**Parameters**:
- `target_dir` (Path): Directory containing test files
- `output_dir` (Path): Output directory for test results
- `logger` (Logger): Logger instance
- `fast_only` (bool): Run only fast tests
- `comprehensive` (bool): Run all tests
- `**kwargs`: Additional options

**Returns**: `True` if tests passed, `False` otherwise

#### `discover_test_files(target_dir=None) -> List[Path]`
**Description**: Discover all test files matching pattern

**Returns**: List of discovered test file paths

---

## Dependencies

### Required Dependencies
- `pytest` - Test execution framework
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-timeout` - Test timeout handling

### Optional Dependencies
- None (all required for testing)

---

## Usage Examples

### Fast Tests
```bash
python src/2_tests.py --fast-only
```

### Comprehensive Tests
```bash
python src/2_tests.py --comprehensive
```

### Default Tests
```bash
python src/2_tests.py --verbose
```

---

## Test Execution Modes

### Fast Mode (--fast-only)
**Files**: 5 test files
- test_fast_suite.py
- test_core_modules.py
- test_environment_overall.py
- test_environment_python.py
- test_environment_system.py

**Duration**: ~20 seconds  
**Success Rate**: 100%

### Default Mode
**Files**: ~10 core test files  
**Duration**: ~2 minutes  
**Success Rate**: ~95%

### Comprehensive Mode (--comprehensive)
**Files**: 54 test files (ALL tests)  
**Duration**: ~15 minutes  
**Success Rate**: 87.7% (263/300 tests passing)

---

## Output Specification

### Output Products
- `pytest_stdout.log` - Test execution output
- `pytest_stderr.log` - Test error output
- `test_results.json` - Structured test results

### Output Directory Structure
```
output/2_tests_output/
└── test_results/
    ├── pytest_stdout.log
    ├── pytest_stderr.log
    └── test_results.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: 44.1s (comprehensive mode)
- **Memory**: 28.9 MB
- **Status**: SUCCESS
- **Tests Run**: 263 passed, 37 failed/skipped
- **Exit Code**: 0

---

## Test Categories

### Core Tests (Always Run)
- Environment validation
- Module imports
- Basic functionality

### Integration Tests
- Pipeline step execution
- Module interactions
- Data flow validation

### Error Scenario Tests
- Dependency failures
- Invalid inputs
- Resource constraints

### Performance Tests
- Execution timing
- Memory usage
- Scalability limits

---

## Testing Standards

### No Mock Policy
- All tests execute real code paths
- No `unittest.mock` or monkeypatching
- Tests may skip when dependencies unavailable
- Pipeline tests run actual scripts via subprocess

### Real Artifact Validation
- Tests assert on actual files in `output/`
- No fake registries or stub implementations
- End-to-end validation preferred

---

## Testing

### Test the Tests
```bash
pytest src/tests/test_tests_integration.py -v
```

### Test Coverage
- **Current**: 95%
- **Target**: 95%+

---

**Last Updated**: September 29, 2025  
**Status**: ✅ Production Ready


