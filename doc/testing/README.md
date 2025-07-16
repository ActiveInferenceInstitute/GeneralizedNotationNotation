# GNN Testing Guide

> **📋 Document Metadata**  
> **Type**: Testing Guide | **Audience**: Developers & QA Engineers | **Complexity**: Intermediate  
> **Last Updated**: June 2025 | **Status**: Production-Ready  
> **Cross-References**: [Development Guide](../development/README.md) | [Pipeline Architecture](../pipeline/README.md)

## Overview
This guide covers the comprehensive testing strategy for GeneralizedNotationNotation (GNN), including unit tests, integration tests, performance tests, and validation procedures.

## Testing Architecture

### Test Organization
```
src/tests/
├── unit/                    # Unit tests for individual components
│   ├── test_gnn_parser.py
│   ├── test_type_checker.py
│   ├── test_export.py
│   └── test_visualization.py
├── integration/             # Integration tests for pipeline steps
│   ├── test_pipeline_flow.py
│   ├── test_backend_rendering.py
│   └── test_mcp_integration.py
├── performance/             # Performance and benchmark tests
│   ├── test_large_models.py
│   └── test_memory_usage.py
├── validation/              # Model validation tests
│   ├── test_syntax_validation.py
│   └── test_semantic_validation.py
├── fixtures/                # Test data and fixtures
│   ├── gnn_models/         # Sample GNN files
│   ├── expected_outputs/   # Expected results
│   └── invalid_models/     # Models with known errors
├── utils/                   # Testing utilities
│   ├── test_helpers.py
│   └── mock_services.py
└── conftest.py             # Pytest configuration
```

## Test Categories

### 1. Unit Tests

#### Core GNN Parser Tests
```python
# src/tests/unit/test_gnn_parser.py
import pytest
from src.gnn import GNNModel, GNNSyntaxError

class TestGNNParser:
    def test_parse_valid_model(self):
        """Test parsing of a valid GNN model"""
        gnn_content = """
        # Test Model
        
        ## GNNVersionAndFlags
        GNN v1.0
        
        ## ModelName
        TestModel
        
        ## StateSpaceBlock
        s_f0[2,1,type=categorical]
        o_m0[2,1,type=categorical]
        
        ## Connections
        s_f0 > o_m0
        """
        
        model = GNNModel.from_string(gnn_content)
        assert model.model_name == "TestModel"
        assert len(model.state_space) == 2
        assert len(model.connections) == 1
    
    def test_parse_invalid_syntax(self):
        """Test that invalid syntax raises appropriate error"""
        invalid_content = """
        ## StateSpaceBlock
        invalid_variable_name[2,1]
        """
        
        with pytest.raises(GNNSyntaxError) as exc_info:
            GNNModel.from_string(invalid_content)
        
        assert "invalid variable name" in str(exc_info.value).lower()
    
    def test_missing_required_sections(self):
        """Test error handling for missing required sections"""
        incomplete_content = """
        ## ModelName
        TestModel
        """
        
        with pytest.raises(GNNSyntaxError) as exc_info:
            GNNModel.from_string(incomplete_content)
        
        assert "missing required section" in str(exc_info.value).lower()

    @pytest.mark.parametrize("variable_def,expected_dims", [
        ("s_f0[2,1,type=categorical]", [2, 1]),
        ("o_m0[3,2,1,type=continuous]", [3, 2, 1]),
        ("u_c0[5,type=binary]", [5]),
    ])
    def test_variable_dimension_parsing(self, variable_def, expected_dims):
        """Test parsing of different variable dimension formats"""
        content = f"""
        ## StateSpaceBlock
        {variable_def}
        """
        # Test implementation here
```

#### Type Checker Tests
```python
# src/tests/unit/test_type_checker.py
import pytest
from src.type_checker import TypeChecker, ValidationResult
from src.gnn import GNNModel

class TestTypeChecker:
    def setup_method(self):
        self.checker = TypeChecker(strict_mode=True)
    
    def test_dimension_compatibility(self):
        """Test matrix dimension compatibility checking"""
        model = GNNModel.from_string("""
        ## StateSpaceBlock
        s_f0[2,1,type=categorical]
        o_m0[3,1,type=categorical]
        
        ## InitialParameterization
        A_m0 = [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]]
        """)
        
        result = self.checker.check_model(model)
        assert result.is_valid
    
    def test_dimension_mismatch_error(self):
        """Test detection of dimension mismatches"""
        model = GNNModel.from_string("""
        ## StateSpaceBlock
        s_f0[2,1,type=categorical]
        o_m0[3,1,type=categorical]
        
        ## InitialParameterization
        A_m0 = [[0.8, 0.2, 0.1], [0.3, 0.7, 0.4]]  # Wrong dimensions
        """)
        
        result = self.checker.check_model(model)
        assert not result.is_valid
        assert any("dimension" in error.message.lower() for error in result.errors)
```

### 2. Integration Tests

#### Pipeline Integration Tests
```python
# src/tests/integration/test_pipeline_flow.py
import pytest
import tempfile
import os
from pathlib import Path
from src.main import run_pipeline

class TestPipelineIntegration:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_basic_model(self):
        """Test complete pipeline run with basic model"""
        # Create test model
        model_path = os.path.join(self.temp_dir, "test_model.md")
        with open(model_path, "w") as f:
            f.write(self.get_basic_model_content())
        
        # Run pipeline steps 1-6 (core functionality)
        result = run_pipeline(
            target_dir=self.temp_dir,
            output_dir=self.output_dir,
            steps=[1, 2, 3, 4, 5, 6]
        )
        
        assert result.success
        assert os.path.exists(os.path.join(self.output_dir, "gnn_exports"))
        assert os.path.exists(os.path.join(self.output_dir, "visualization"))
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid model"""
        # Create invalid model
        model_path = os.path.join(self.temp_dir, "invalid_model.md")
        with open(model_path, "w") as f:
            f.write("## Invalid\nThis is not a valid GNN model")
        
        result = run_pipeline(
            target_dir=self.temp_dir,
            output_dir=self.output_dir,
            steps=[1, 4]
        )
        
        assert not result.success
        assert "syntax error" in result.error_message.lower()
    
    @staticmethod
    def get_basic_model_content():
        return """
        # Test Model
        
        ## GNNVersionAndFlags
        GNN v1.0
        
        ## ModelName
        BasicTestModel
        
        ## StateSpaceBlock
        s_f0[2,1,type=categorical]
        o_m0[2,1,type=categorical]
        
        ## Connections
        s_f0 > o_m0
        
        ## InitialParameterization
        A_m0 = [[0.8, 0.2], [0.3, 0.7]]
        D_f0 = [0.5, 0.5]
        """
```

#### Backend Rendering Tests
```python
# src/tests/integration/test_backend_rendering.py
import pytest
from src.render.pymdp import PyMDPRenderer
from src.render.rxinfer import RxInferRenderer
from src.gnn import GNNModel

class TestBackendRendering:
    def setup_method(self):
        self.test_model = GNNModel.from_string("""
        ## ModelName
        RenderingTestModel
        
        ## StateSpaceBlock
        s_f0[3,1,type=categorical]
        o_m0[2,1,type=categorical]
        u_c0[2,1,type=categorical]
        
        ## Connections
        s_f0 > o_m0
        u_c0 > s_f0
        """)
    
    def test_pymdp_rendering(self):
        """Test PyMDP code generation"""
        renderer = PyMDPRenderer()
        code = renderer.render_model(self.test_model)
        
        # Check that essential PyMDP components are present
        assert "import pymdp" in code
        assert "Agent" in code
        assert "A = " in code  # Likelihood matrix
        assert "B = " in code  # Transition matrix
    
    def test_rxinfer_rendering(self):
        """Test RxInfer.jl code generation"""
        renderer = RxInferRenderer()
        code = renderer.render_model(self.test_model)
        
        # Check that essential RxInfer components are present
        assert "using RxInfer" in code
        assert "@model" in code
        assert "s_f0" in code
        assert "o_m0" in code
    
    def test_rendering_consistency(self):
        """Test that different backends produce consistent results"""
        pymdp_renderer = PyMDPRenderer()
        rxinfer_renderer = RxInferRenderer()
        
        pymdp_code = pymdp_renderer.render_model(self.test_model)
        rxinfer_code = rxinfer_renderer.render_model(self.test_model)
        
        # Both should handle the same variables
        for var in ["s_f0", "o_m0", "u_c0"]:
            assert var in pymdp_code
            assert var in rxinfer_code
```

### 3. Performance Tests

#### Large Model Performance Tests
```python
# src/tests/performance/test_large_models.py
import pytest
import time
import psutil
import os
from src.gnn import GNNModel
from src.type_checker import TypeChecker

class TestPerformance:
    @pytest.mark.slow
    def test_large_model_parsing(self):
        """Test parsing performance with large models"""
        # Generate large model
        large_model_content = self.generate_large_model(
            num_states=100, 
            num_observations=50
        )
        
        start_time = time.time()
        model = GNNModel.from_string(large_model_content)
        parse_time = time.time() - start_time
        
        # Should parse large model in reasonable time
        assert parse_time < 10.0  # seconds
        assert len(model.state_space) == 150  # 100 states + 50 observations
    
    @pytest.mark.slow
    def test_memory_usage_large_model(self):
        """Test memory usage with large models"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large model
        large_model = self.generate_large_model(200, 100)
        model = GNNModel.from_string(large_model)
        checker = TypeChecker()
        result = checker.check_model(model)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for this test)
        assert memory_increase < 500
    
    @staticmethod
    def generate_large_model(num_states, num_observations):
        """Generate a large GNN model for testing"""
        content = """
        ## GNNVersionAndFlags
        GNN v1.0
        
        ## ModelName
        LargeTestModel
        
        ## StateSpaceBlock
        """
        
        # Add state variables
        for i in range(num_states):
            content += f"s_f{i}[3,1,type=categorical]\n"
        
        # Add observation variables
        for i in range(num_observations):
            content += f"o_m{i}[2,1,type=categorical]\n"
        
        content += "\n## Connections\n"
        
        # Add connections (each state to first observation)
        for i in range(num_states):
            content += f"s_f{i} > o_m0\n"
        
        return content
```

### 4. Validation Tests

#### Model Validation Tests
```python
# src/tests/validation/test_syntax_validation.py
import pytest
from src.gnn import GNNModel, GNNSyntaxError

class TestSyntaxValidation:
    @pytest.mark.parametrize("invalid_content,expected_error", [
        (
            "## StateSpaceBlock\ninvalid-name[2]",
            "invalid variable name"
        ),
        (
            "## StateSpaceBlock\ns_f0[type=invalid]",
            "invalid type"
        ),
        (
            "## Connections\ns_f0 -> o_m0",  # Invalid arrow
            "invalid connection syntax"
        ),
    ])
    def test_syntax_errors(self, invalid_content, expected_error):
        """Test detection of various syntax errors"""
        with pytest.raises(GNNSyntaxError) as exc_info:
            GNNModel.from_string(invalid_content)
        
        assert expected_error.lower() in str(exc_info.value).lower()
```

## Test Configuration

### Pytest Configuration
```python
# src/tests/conftest.py
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_directory():
    """Provide a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_gnn_model():
    """Provide a standard test GNN model"""
    return """
    ## GNNVersionAndFlags
    GNN v1.0
    
    ## ModelName
    TestModel
    
    ## StateSpaceBlock
    s_f0[2,1,type=categorical]
    o_m0[2,1,type=categorical]
    
    ## Connections
    s_f0 > o_m0
    
    ## InitialParameterization
    A_m0 = [[0.8, 0.2], [0.3, 0.7]]
    D_f0 = [0.5, 0.5]
    """

@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    class MockLLMService:
        def analyze_model(self, model):
            return {"analysis": "test analysis"}
        
        def suggest_improvements(self, model):
            return ["test suggestion"]
    
    return MockLLMService()

# Test markers
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
```

### pytest.ini Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = src/tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    performance: marks tests as performance tests

addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html:output/coverage
    --cov-fail-under=80

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

## Running Tests

### Basic Test Commands
```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest src/tests/unit/          # Unit tests only
python -m pytest src/tests/integration/   # Integration tests only
python -m pytest -m "not slow"            # Skip slow tests

# Run with coverage
python -m pytest --cov=src --cov-report=html:output/coverage

# Run specific test file
python -m pytest src/tests/unit/test_gnn_parser.py

# Run specific test function
python -m pytest src/tests/unit/test_gnn_parser.py::TestGNNParser::test_parse_valid_model
```

### Advanced Test Commands
```bash
# Parallel test execution
python -m pytest -n auto

# Run tests with verbose output
python -m pytest -v

# Run failed tests only
python -m pytest --lf

# Generate test report
python -m pytest --html=output/test_report.html --self-contained-html

# Profile test performance
python -m pytest --profile

# Run tests with specific fixtures
python -m pytest --fixtures
```

## Continuous Integration

### GitHub Actions Test Workflow
```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y graphviz
    
    - name: Install Julia
      uses: julia-actions/setup-julia@v1
      with:
        version: '1.9'
    
    - name: Setup GNN environment
      run: |
        python src/main.py --only-steps 2
    
    - name: Run tests
      run: |
        source src/.venv/bin/activate
        python -m pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
    -   id: black

-   repo: https://github.com/PyCQA/flake8
    rev: 5.0.4
    hooks:
    -   id: flake8

-   repo: local
    hooks:
    -   id: pytest-check
        name: pytest-check
        entry: python -m pytest src/tests/unit/ -x
        language: system
        pass_filenames: false
        always_run: true
```

## Test Data Management

### Test Fixtures Organization
```
src/tests/fixtures/
├── gnn_models/
│   ├── valid/
│   │   ├── basic_model.md
│   │   ├── complex_model.md
│   │   └── multimodal_model.md
│   ├── invalid/
│   │   ├── syntax_errors.md
│   │   ├── dimension_mismatches.md
│   │   └── missing_sections.md
│   └── edge_cases/
│       ├── minimal_model.md
│       ├── large_model.md
│       └── unicode_model.md
├── expected_outputs/
│   ├── json_exports/
│   ├── visualizations/
│   └── rendered_code/
└── mock_data/
    ├── simulation_results.pkl
    └── llm_responses.json
```

## Best Practices

### 1. Test Design Principles
- **Isolation**: Each test should be independent
- **Deterministic**: Tests should produce consistent results
- **Fast**: Unit tests should run quickly
- **Clear**: Test names and assertions should be descriptive

### 2. Coverage Goals
- **Unit Tests**: 90%+ coverage for core modules
- **Integration Tests**: Cover all pipeline step interactions
- **Edge Cases**: Test boundary conditions and error paths
- **Regression Tests**: Prevent previously fixed bugs

### 3. Performance Testing
- **Benchmarks**: Establish performance baselines
- **Memory Monitoring**: Track memory usage for large models
- **Timeout Limits**: Set reasonable execution time limits
- **Resource Cleanup**: Ensure tests clean up resources

### 4. Test Maintenance
- **Regular Updates**: Keep tests current with code changes
- **Documentation**: Document complex test scenarios
- **Refactoring**: Regularly refactor test code for clarity
- **Review Process**: Include test reviews in code review process

This comprehensive testing strategy ensures the reliability, performance, and maintainability of the GNN system across all its components and use cases. 