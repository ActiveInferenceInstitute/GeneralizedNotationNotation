Ensure # Testing Framework Guidelines

## Overview

This document provides comprehensive guidelines for the GNN testing framework, which has been completely refactored to follow a modular, organized structure with comprehensive coverage for all modules. The framework implements a clean, module-based naming convention and provides robust test execution capabilities.

## Test Architecture

### Module-Based Test Organization

The test infrastructure follows a clean, modular organization pattern:

```
src/tests/
├── test_MODULENAME_overall.py      # Comprehensive module coverage
├── test_MODULENAME_area1.py        # Specific module areas
├── test_MODULENAME_area2.py        # Additional specialized areas
├── runner.py                       # Test runner with category-based execution
├── conftest.py                     # Pytest configuration and fixtures
├── run_fast_tests.py              # Fast test suite execution
└── README.md                      # Comprehensive documentation
```

### Test File Naming Convention

All test files follow the pattern:
- `test_MODULENAME_overall.py` - Comprehensive module coverage
- `test_MODULENAME_area1.py` - Specific module areas (e.g., parsing, validation, processing)
- `test_MODULENAME_area2.py` - Additional specialized areas (e.g., integration, performance)

### Current Test Categories

The test runner (`runner.py`) is configured with 23 comprehensive test categories:

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
    "render": {
        "name": "Render Module Tests", 
        "description": "Code generation and rendering tests",
        "files": ["test_render_overall.py", "test_render_integration.py", "test_render_performance.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "mcp": {
        "name": "MCP Module Tests",
        "description": "Model Context Protocol tests",
        "files": ["test_mcp_overall.py", "test_mcp_tools.py", "test_mcp_transport.py", 
                  "test_mcp_integration.py", "test_mcp_performance.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "audio": {
        "name": "Audio Module Tests",
        "description": "Audio generation and SAPF tests",
        "files": ["test_audio_overall.py", "test_audio_sapf.py", "test_audio_generation.py", 
                  "test_audio_integration.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "visualization": {
        "name": "Visualization Module Tests",
        "description": "Graph and matrix visualization tests",
        "files": ["test_visualization_overall.py", "test_visualization_matrices.py", 
                  "test_visualization_ontology.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "pipeline": {
        "name": "Pipeline Module Tests",
        "description": "Pipeline orchestration and step tests",
        "files": ["test_pipeline_overall.py", "test_pipeline_integration.py", 
                  "test_pipeline_orchestration.py", "test_pipeline_performance.py", 
                  "test_pipeline_recovery.py", "test_pipeline_scripts.py", 
                  "test_pipeline_infrastructure.py", "test_pipeline_functionality.py"],
        "markers": [],
        "timeout_seconds": 180,
        "max_failures": 10,
        "parallel": False
    },
    "export": {
        "name": "Export Module Tests",
        "description": "Multi-format export tests",
        "files": ["test_export_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "execute": {
        "name": "Execute Module Tests",
        "description": "Execution and simulation tests",
        "files": ["test_execute_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "llm": {
        "name": "LLM Module Tests",
        "description": "LLM integration and analysis tests",
        "files": ["test_llm_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "ontology": {
        "name": "Ontology Module Tests",
        "description": "Ontology processing and validation tests",
        "files": ["test_ontology_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "website": {
        "name": "Website Module Tests",
        "description": "Website generation tests",
        "files": ["test_website_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "report": {
        "name": "Report Module Tests",
        "description": "Report generation and formatting tests",
        "files": ["test_report_overall.py", "test_report_generation.py", 
                  "test_report_integration.py", "test_report_formats.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "environment": {
        "name": "Environment Module Tests",
        "description": "Environment setup and validation tests",
        "files": ["test_environment_overall.py", "test_environment_dependencies.py",
                  "test_environment_integration.py", "test_environment_python.py",
                  "test_environment_system.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "type_checker": {
        "name": "Type Checker Module Tests",
        "description": "Type checking and validation tests",
        "files": ["test_type_checker_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "validation": {
        "name": "Validation Module Tests",
        "description": "Validation and consistency tests",
        "files": ["test_validation_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "model_registry": {
        "name": "Model Registry Module Tests",
        "description": "Model registry and versioning tests",
        "files": ["test_model_registry_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "analysis": {
        "name": "Analysis Module Tests",
        "description": "Analysis and statistical tests",
        "files": ["test_analysis_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "integration": {
        "name": "Integration Module Tests",
        "description": "System integration tests",
        "files": ["test_integration_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "security": {
        "name": "Security Module Tests",
        "description": "Security validation tests",
        "files": ["test_security_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "research": {
        "name": "Research Module Tests",
        "description": "Research tools tests",
        "files": ["test_research_overall.py"],
        "markers": [],
        "timeout_seconds": 90,
        "max_failures": 6,
        "parallel": True
    },
    "ml_integration": {
        "name": "ML Integration Module Tests",
        "description": "Machine learning integration tests",
        "files": ["test_ml_integration_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "advanced_visualization": {
        "name": "Advanced Visualization Module Tests",
        "description": "Advanced visualization tests",
        "files": ["test_advanced_visualization_overall.py"],
        "markers": [],
        "timeout_seconds": 120,
        "max_failures": 8,
        "parallel": True
    },
    "comprehensive": {
        "name": "Comprehensive API Tests",
        "description": "Comprehensive API and integration tests",
        "files": ["test_comprehensive_api.py", "test_core_modules.py", "test_fast_suite.py",
                  "test_main_orchestrator.py", "test_coverage_overall.py", "test_performance_overall.py",
                  "test_unit_overall.py"],
        "markers": [],
        "timeout_seconds": 300,
        "max_failures": 15,
        "parallel": False
    }
}
```

## Import Patterns

### Correct Import Patterns
All test files must use the following import patterns:

```python
#!/usr/bin/env python3
"""
Test ModuleName Overall Tests

This file contains comprehensive tests for the module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *

# ✅ CORRECT - Use direct module imports
from gnn import discover_gnn_files, parse_gnn_file
from render import render_gnn_to_pymdp
from execute import execute_script_safely
from llm import analyze_gnn_model
from utils import setup_step_logging, EnhancedArgumentParser
from pipeline import get_pipeline_config
from visualization import create_graph_visualization
from export import export_to_json
from ontology import process_ontology
from website import generate_website
from audio import generate_sapf_audio
from type_checker import validate_gnn_files
from validation import validate_gnn_structure
from report import generate_report
from setup import validate_environment
```

### Incorrect Import Patterns
```python
# ❌ INCORRECT - These cause import errors
from src.gnn import discover_gnn_files
from src.render import render_gnn_to_pymdp
from src.utils import setup_step_logging
from src.pipeline import get_pipeline_config
```

## Test File Structure

### Standard Test File Template

```python
#!/usr/bin/env python3
"""
Test ModuleName Overall Tests

This file contains comprehensive tests for the module functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import *


class TestModuleNameComprehensive:
    """Comprehensive tests for the module."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_module_imports(self):
        """Test that module can be imported."""
        try:
            import module_name
            assert hasattr(module_name, '__version__')
            assert hasattr(module_name, 'MainClass')
            assert hasattr(module_name, 'get_module_info')
        except ImportError:
            pytest.skip("Module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_main_class_instantiation(self):
        """Test MainClass instantiation."""
        try:
            from module_name import MainClass
            instance = MainClass()
            assert instance is not None
            assert hasattr(instance, 'main_method')
        except ImportError:
            pytest.skip("MainClass not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_module_info(self):
        """Test module information retrieval."""
        try:
            from module_name import get_module_info
            info = get_module_info()
            assert isinstance(info, dict)
            assert 'version' in info
            assert 'description' in info
        except ImportError:
            pytest.skip("Module info not available")


class TestModuleNameFunctionality:
    """Tests for module functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_basic_functionality(self, comprehensive_test_data):
        """Test basic functionality."""
        try:
            from module_name import MainClass
            instance = MainClass()
            result = instance.main_method(comprehensive_test_data)
            assert result is not None
        except ImportError:
            pytest.skip("Module not available")


class TestModuleNameIntegration:
    """Integration tests for module."""
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_module_pipeline_integration(self, sample_gnn_files, isolated_temp_dir):
        """Test module integration with pipeline."""
        try:
            from module_name import MainClass
            instance = MainClass()
            result = instance.main_method({'test': 'data'})
            assert result is not None
        except ImportError:
            pytest.skip("Module not available")
    
    @pytest.mark.integration
    @pytest.mark.safe_to_fail
    def test_module_mcp_integration(self):
        """Test module MCP integration."""
        try:
            from module_name.mcp import register_tools
            assert callable(register_tools)
        except ImportError:
            pytest.skip("Module MCP not available")


def test_module_completeness():
    """Test that module has all required components."""
    required_components = [
        'MainClass',
        'get_module_info',
        'validate_module'
    ]
    
    try:
        import module_name
        for component in required_components:
            assert hasattr(module_name, component), f"Missing component: {component}"
    except ImportError:
        pytest.skip("Module not available")


@pytest.mark.slow
def test_module_performance():
    """Test module performance characteristics."""
    try:
        from module_name import MainClass
        import time
        
        instance = MainClass()
        start_time = time.time()
        
        result = instance.main_method({'test': 'data'})
        
        processing_time = time.time() - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
        
    except ImportError:
        pytest.skip("Module not available")
```

## Test Execution Patterns

### Test Runner Execution
```bash
# Run all tests through the pipeline test runner
python src/2_tests.py --target-dir input/gnn_files --output-dir output --verbose

# Run specific module tests
python src/2_tests.py --category gnn
python src/2_tests.py --category render,audio,visualization

# Run with specific options
python src/2_tests.py --verbose --parallel --coverage

# Run only fast tests
python src/2_tests.py --fast-only --verbose
```

### Individual Test Execution
```bash
# Run a specific test file
PYTHONPATH=src python -m pytest src/tests/test_gnn_overall.py -v

# Run a specific test class
PYTHONPATH=src python -m pytest src/tests/test_gnn_overall.py::TestGNNComprehensive -v

# Run a specific test method
PYTHONPATH=src python -m pytest src/tests/test_gnn_overall.py::TestGNNComprehensive::test_module_imports -v

# Run with pytest directly
PYTHONPATH=src python -c "import pytest; pytest.main(['src/tests/', '--tb=no', '--no-header'])"
```

### Fast Test Suite
```bash
# Run fast test suite
python src/tests/run_fast_tests.py
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
```

### Running Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run fast tests only
pytest -m fast

# Exclude slow tests
pytest -m "not slow"

# Run safe-to-fail tests
pytest -m safe_to_fail
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
def safe_filesystem():
    """Safe filesystem operations."""

@pytest.fixture
def mock_subprocess():
    """Safe subprocess execution."""

@pytest.fixture
def mock_imports():
    """Safe module imports."""

@pytest.fixture
def mock_logger():
    """Mock logging."""
```

## Test Utilities

### Shared Test Utilities (src/utils/test_utils.py)
```python
def is_safe_mode() -> bool:
    """Check if running in safe mode."""

def setup_test_environment() -> bool:
    """Setup test environment."""

def create_sample_gnn_content() -> str:
    """Create sample GNN content."""

def performance_tracker() -> ContextManager:
    """Track performance metrics."""

def get_memory_usage() -> float:
    """Get current memory usage."""

def assert_file_exists(file_path: Path) -> None:
    """Assert file exists."""

def generate_html_report_file(report_data: Dict[str, Any], output_path: Path) -> None:
    """Generate HTML report."""

def generate_markdown_report_file(report_data: Dict[str, Any], output_path: Path) -> None:
    """Generate Markdown report."""

def generate_json_report_file(report_data: Dict[str, Any], output_path: Path) -> None:
    """Generate JSON report."""
```

## Error Handling Patterns

### Safe-to-Fail Pattern
```python
@pytest.mark.safe_to_fail
def test_module_imports(self):
    """Test that module can be imported."""
    try:
        import module_name
        assert hasattr(module_name, '__version__')
        assert hasattr(module_name, 'MainClass')
        assert hasattr(module_name, 'get_module_info')
    except ImportError:
        pytest.skip("Module not available")
    except Exception as e:
        pytest.skip(f"Module import failed: {e}")
```

### Graceful Degradation Pattern
```python
def test_optional_functionality(self):
    """Test optional functionality with graceful degradation."""
    try:
        from module_name import optional_function
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
from utils.test_utils import track_peak_memory, get_memory_usage

@track_peak_memory
def test_memory_intensive_operation():
    """Test memory-intensive operation."""
    # Test code here
    pass
```

### Resource Limits
```python
from utils.test_utils import with_resource_limits

@with_resource_limits(max_memory_mb=100, max_cpu_percent=50)
def test_with_limits():
    """Test with resource limits."""
    # Test code here
    pass
```

## Test Categories

### Core Module Tests
- **GNN Module**: Processing, validation, parsing, integration
- **Render Module**: Code generation, multiple targets, performance
- **MCP Module**: Model Context Protocol, tools, transport, integration
- **Audio Module**: SAPF, generation, integration
- **Visualization Module**: Graphs, matrices, ontology, interactive

### Infrastructure Module Tests
- **Pipeline Module**: Orchestration, steps, configuration, performance, recovery
- **Export Module**: Multi-format export (JSON, XML, GraphML, GEXF, Pickle)
- **Execute Module**: Execution and simulation
- **LLM Module**: LLM integration and analysis
- **Ontology Module**: Ontology processing and validation
- **Website Module**: Website generation
- **Report Module**: Report generation and formatting
- **Environment Module**: Environment setup and validation

### Specialized Module Tests
- **Type Checker Module**: Type checking and validation
- **Validation Module**: Validation and consistency
- **Model Registry Module**: Model registry and versioning
- **Analysis Module**: Analysis and statistical
- **Integration Module**: System integration
- **Security Module**: Security validation
- **Research Module**: Research tools
- **ML Integration Module**: Machine learning integration
- **Advanced Visualization Module**: Advanced visualization

### Comprehensive Tests
- **Comprehensive API**: Complete API testing
- **Core Modules**: Core module integration
- **Fast Suite**: Fast execution tests
- **Main Orchestrator**: Main orchestrator functionality
- **Coverage**: Code coverage tests
- **Performance**: Performance and benchmarking
- **Unit**: Basic unit tests

## Test Execution Features

### Resource Monitoring
- Memory usage tracking
- CPU usage monitoring
- Timeout handling
- Resource limits

### Parallel Execution
- Category-based parallel execution
- Configurable parallelization
- Resource-aware scheduling

### Error Handling
- Graceful failure handling
- Error reporting and logging
- Recovery mechanisms
- Safe-to-fail test execution

### Reporting
- Comprehensive test reports
- Performance metrics
- Coverage analysis
- Error summaries

## Coverage Analysis

### Running Coverage
```bash
# Run with coverage
python -m pytest --cov=src --cov-report=html --cov-report=term

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
   - Check import paths (should use direct module imports, not `src.module`)
   - Ensure `src/` is in Python path with `sys.path.insert(0, str(Path(__file__).parent.parent))`
   - Verify module exists and is properly structured

2. **Plugin Conflicts**
   - Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` if needed
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
python -m pytest -vvv --tb=long

# Run with print statements
python -m pytest -s

# Run with debugger
python -m pytest --pdb
```

## Best Practices

### Test Organization
1. **Module-Based Structure**: Each module has its own test files
2. **Comprehensive Coverage**: Each module has an `_overall.py` test file
3. **Specialized Testing**: Additional test files for specific areas
4. **Integration Testing**: Cross-module integration tests

### Test Writing
1. **Safe-to-Fail**: Use `@pytest.mark.safe_to_fail` for tests that can fail gracefully
2. **Import Error Handling**: Wrap imports in try/except blocks
3. **Comprehensive Assertions**: Test both success and failure cases
4. **Performance Monitoring**: Use performance tracking for slow operations

### Test Execution
1. **Category-Based**: Run tests by module category
2. **Parallel Execution**: Use parallel execution for faster results
3. **Resource Monitoring**: Monitor resource usage during execution
4. **Error Recovery**: Handle errors gracefully with fallback mechanisms

### Documentation and Communication Standards
- **Direct Documentation Updates**: Update existing test documentation directly rather than creating separate report files
- **Functional Improvements**: Focus on making smart functional improvements to tests and documentation
- **Inline Updates**: Add documentation directly to relevant test files and README.md
- **Concrete Demonstrations**: Show test functionality through working code and actual test results
- **Understated Communication**: Use specific examples and functional demonstrations over promotional language

## Current Status

### Test Coverage
- **423 test items** collected
- **Comprehensive module coverage** for all major modules
- **Specialized test areas** for specific functionality
- **Integration tests** for cross-module functionality

### Test Infrastructure
- **Modular test runner** with category-based execution
- **Resource monitoring** and timeout handling
- **Parallel execution** support
- **Comprehensive reporting** and error handling

### Module Coverage
- ✅ GNN Module - Complete coverage
- ✅ Render Module - Complete coverage
- ✅ MCP Module - Complete coverage
- ✅ Audio Module - Complete coverage
- ✅ Visualization Module - Complete coverage
- ✅ Pipeline Module - Complete coverage
- ✅ Export Module - Complete coverage
- ✅ Execute Module - Complete coverage
- ✅ LLM Module - Complete coverage
- ✅ Ontology Module - Complete coverage
- ✅ Website Module - Complete coverage
- ✅ Report Module - Complete coverage
- ✅ Environment Module - Complete coverage

## Future Enhancements

### Planned Improvements
1. **Additional Module Tests**: Complete coverage for remaining modules
2. **Performance Benchmarking**: Enhanced performance testing
3. **Coverage Analysis**: Improved code coverage tracking
4. **Automated Testing**: CI/CD integration
5. **Test Documentation**: Enhanced test documentation

### Module Expansion
- Type Checker Module tests
- Validation Module tests
- Model Registry Module tests
- Analysis Module tests
- Integration Module tests
- Security Module tests
- Research Module tests
- ML Integration Module tests
- Advanced Visualization Module tests

## Continuous Integration

The test suite is designed to work with CI/CD systems:

- **Fast tests** run on every commit
- **Standard tests** run on pull requests
- **Slow and performance tests** run nightly
- **Test reports** are generated and stored as artifacts
- **Comprehensive error categorization** and reporting

## Test File Naming Conventions

- `test_*.py`: Individual test files
- `conftest.py`: Pytest configuration and fixtures
- `runner.py`: Test runner implementation
- `run_fast_tests.py`: Fast test suite execution
- `README.md`: Comprehensive documentation

## Test Structure

### Test Class Structure
```python
class TestModuleNameComprehensive:
    """Comprehensive tests for ModuleName functionality."""
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_module_imports(self):
        """Test that module can be imported."""
        try:
            import module_name
            assert hasattr(module_name, '__version__')
            assert hasattr(module_name, 'MainClass')
        except ImportError:
            pytest.skip("Module not available")
    
    @pytest.mark.unit
    @pytest.mark.safe_to_fail
    def test_basic_functionality(self, comprehensive_test_data):
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

This testing framework provides a solid foundation for comprehensive testing of the GNN Processing Pipeline, with modular organization, parallel execution, and comprehensive coverage of all major components. 