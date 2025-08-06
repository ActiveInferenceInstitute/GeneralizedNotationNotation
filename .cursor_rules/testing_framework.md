# Testing Framework Guidelines

## Overview

This document provides comprehensive guidelines for the GNN testing framework, including import patterns, test organization, and best practices for maintaining a robust test suite.

## Import Patterns

### Correct Import Patterns
All test files must use the following import patterns:

```python
# ✅ CORRECT - Use src.module pattern
from src.gnn import discover_gnn_files, parse_gnn_file
from src.render import render_gnn_to_pymdp
from src.execute import execute_script_safely
from src.llm import analyze_gnn_model
from src.utils import setup_step_logging, EnhancedArgumentParser
from src.pipeline import get_pipeline_config
from src.visualization import create_graph_visualization
from src.export import export_to_json
from src.ontology import process_ontology
from src.website import generate_website
from src.audio import generate_sapf_audio
from src.type_checker import validate_gnn_files
from src.validation import validate_gnn_structure
from src.report import generate_report
from src.setup import validate_environment
```

### Incorrect Import Patterns
```python
# ❌ INCORRECT - These cause import errors
from gnn import discover_gnn_files
from render import render_gnn_to_pymdp
from utils import setup_step_logging
from pipeline import get_pipeline_config
```

## Test File Organization

### Test Categories
Tests are organized into the following categories:

1. **Core Module Tests** (`test_core_modules.py`)
   - GNN module functionality
   - Render module functionality
   - Execute module functionality
   - LLM module functionality
   - MCP module functionality
   - Ontology module functionality
   - Website module functionality
   - SAPF module functionality

2. **GNN Core Tests** (`test_gnn_core_modules.py`)
   - Core processor functionality
   - Discovery functionality
   - Reporting functionality
   - Validation functionality
   - Simple validator functionality
   - Parsers and serializers

3. **Pipeline Infrastructure Tests** (`test_pipeline_infrastructure.py`)
   - Pipeline discovery
   - Step template functionality
   - Pipeline validation
   - Resource management
   - Script validation

4. **Utility Tests** (`test_utilities.py`)
   - Argument parsing
   - Logging setup
   - Path utilities
   - Performance tracking
   - Dependency validation

5. **Fast Suite Tests** (`test_fast_suite.py`)
   - Quick validation tests
   - Environment checks
   - Basic functionality tests

6. **Integration Tests** (`integration_tests.py`)
   - Cross-module integration
   - End-to-end workflows
   - Pipeline coordination

7. **Performance Tests** (`test_pipeline_performance.py`)
   - Memory usage patterns
   - Execution time tests
   - Resource scaling tests

8. **MCP Integration Tests** (`test_mcp_integration_comprehensive.py`)
   - MCP tool registration
   - MCP tool execution
   - MCP resource management

## Test Execution Patterns

### Individual Test Execution
```bash
# Run a specific test file
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/tests/test_utilities.py -v -p no:randomly -p no:sugar -p no:cacheprovider

# Run a specific test class
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/tests/test_utilities.py::TestArgumentParsing -v -p no:randomly -p no:sugar -p no:cacheprovider

# Run a specific test method
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/tests/test_utilities.py::TestArgumentParsing::test_enhanced_argument_parser_import -v -p no:randomly -p no:sugar -p no:cacheprovider
```

### Test Runner Execution
```bash
# Run all tests through the pipeline test runner
python3 src/2_tests.py --target-dir input/gnn_files --output-dir output --verbose

# Run only fast tests
python3 src/2_tests.py --target-dir input/gnn_files --output-dir output --fast-only --verbose
```

## Test Markers

### Available Markers
```python
@pytest.mark.unit              # Unit tests for individual components
@pytest.mark.integration       # Integration tests for component interactions
@pytest.mark.performance       # Performance and resource usage tests
@pytest.mark.slow             # Tests that take significant time to complete
@pytest.mark.fast             # Quick tests for rapid feedback
@pytest.mark.safe_to_fail     # Tests safe to run without side effects
@pytest.mark.type_checking    # Type checking tests
@pytest.mark.mcp              # MCP tests
@pytest.mark.sapf             # SAPF tests
@pytest.mark.visualization    # Visualization tests
```

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

## Test Fixtures

### Core Fixtures (conftest.py)
```python
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Test configuration for the session."""

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Path to project root."""

@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Path to source directory."""

@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Path to test directory."""

@pytest.fixture
def isolated_temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for tests."""

@pytest.fixture
def sample_gnn_files(safe_filesystem) -> Dict[str, Path]:
    """Sample GNN files for testing."""

@pytest.fixture
def comprehensive_test_data(isolated_temp_dir) -> Dict[str, Any]:
    """Comprehensive test data."""
```

### Mock Fixtures
```python
@pytest.fixture
def mock_subprocess(safe_subprocess):
    """Safe subprocess execution."""

@pytest.fixture
def mock_filesystem(safe_filesystem):
    """Safe filesystem operations."""

@pytest.fixture
def mock_imports(real_imports):
    """Safe module imports."""

@pytest.fixture
def mock_dangerous_operations():
    """Safe dangerous operations."""

@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider."""

@pytest.fixture
def mock_logger():
    """Mock logging."""
```

## Test Utilities

### Core Functions (test_utils.py)
```python
def is_safe_mode() -> bool:
    """Check if running in safe mode."""

def validate_test_environment() -> bool:
    """Validate test environment."""

def create_test_gnn_files() -> Dict[str, Path]:
    """Create test GNN files."""

def run_all_tests() -> Dict[str, Any]:
    """Run comprehensive test suite."""

def performance_tracker() -> ContextManager:
    """Track performance metrics."""

def with_resource_limits(max_memory_mb: int, max_cpu_percent: int) -> Callable:
    """Apply resource limits."""
```

## Error Handling Patterns

### Safe-to-Fail Pattern
```python
@pytest.mark.safe_to_fail
def test_module_imports(self):
    """Test that module can be imported."""
    try:
        from src.gnn import discover_gnn_files
        assert callable(discover_gnn_files)
        logging.info("Module imports validated")
    except ImportError as e:
        pytest.skip(f"Module not available: {e}")
    except Exception as e:
        logging.warning(f"Module import failed: {e}")
        pytest.skip(f"Module import failed: {e}")
```

### Graceful Degradation Pattern
```python
def test_optional_functionality(self):
    """Test optional functionality with graceful degradation."""
    try:
        from src.optional_module import optional_function
        result = optional_function()
        assert result is not None
    except ImportError:
        # Functionality not available, test passes
        pytest.skip("Optional functionality not available")
    except Exception as e:
        # Functionality available but failed, test fails
        pytest.fail(f"Optional functionality failed: {e}")
```

## Performance Testing

### Memory Tracking
```python
from src.tests.test_utils import track_peak_memory, get_memory_usage

@track_peak_memory
def test_memory_intensive_operation():
    """Test memory-intensive operation."""
    # Test code here
    pass
```

### Resource Limits
```python
from src.tests.test_utils import with_resource_limits

@with_resource_limits(max_memory_mb=100, max_cpu_percent=50)
def test_with_limits():
    """Test with resource limits."""
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

## Best Practices

1. **Always use real implementations** rather than mocks
2. **Add proper markers** to tests based on execution time
3. **Use fixtures** for test data generation
4. **Monitor resource usage** in performance-critical tests
5. **Ensure all tests can run independently**
6. **Follow the modular organization pattern**
7. **Use the thin orchestrator pattern** for pipeline scripts
8. **Implement comprehensive error handling and reporting**
9. **Maintain separation of concerns** between orchestration and core functionality
10. **Use safe-to-fail patterns** for integration tests

## Test File Naming Conventions

- `test_*.py`: Individual test files
- `*_tests.py`: Test suite files
- `conftest.py`: Pytest configuration and fixtures
- `test_utils.py`: Test utilities and helpers
- `runner.py`: Test runner implementation

## Test Structure

### Test Class Structure
```python
class TestModuleName:
    """Test cases for ModuleName functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_module_imports(self):
        """Test that module can be imported."""
        try:
            from src.module import function_name
            assert callable(function_name)
        except ImportError as e:
            pytest.skip(f"Module not available: {e}")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_basic_functionality(self, sample_gnn_files):
        """Test basic functionality."""
        # Test implementation
        pass
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_integration_workflow(self, sample_gnn_files, isolated_temp_dir):
        """Test integration workflow."""
        # Integration test implementation
        pass
```

### Test Function Structure
```python
def test_function_name():
    """Test description."""
    # Arrange
    test_data = create_test_data()
    
    # Act
    result = function_under_test(test_data)
    
    # Assert
    assert result is not None
    assert isinstance(result, expected_type)
    assert result.has_expected_property()
```

## Continuous Integration

The test suite is designed to work with CI/CD systems:

- **Fast tests** run on every commit
- **Standard tests** run on pull requests
- **Slow and performance tests** run nightly
- **Test reports** are generated and stored as artifacts
- **Comprehensive error categorization** and reporting

## Future Enhancements

1. **Additional test categories** for new modules
2. **Enhanced performance monitoring**
3. **Integration with external CI/CD systems**
4. **Advanced test result visualization**
5. **Automated test generation** for new modules
6. **Parallel test execution** implementation
7. **Test result caching** for faster feedback
8. **Advanced coverage analysis** with branch coverage
9. **Performance regression testing** with baselines
10. **Automated test maintenance** tools 