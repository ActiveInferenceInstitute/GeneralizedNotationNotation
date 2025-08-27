# Test Infrastructure

This directory contains the comprehensive test suite for the GNN Processing Pipeline. The test infrastructure has been completely refactored to follow a modular, organized structure that provides comprehensive coverage for all modules.

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
- `test_gnn_validation.py` - GNN validation and consistency tests
- `test_gnn_processing.py` - GNN processing and serialization tests
- `test_gnn_integration.py` - GNN integration tests

#### Render Module Tests
- `test_render_overall.py` - Comprehensive render module testing
- `test_render_integration.py` - Render integration tests
- `test_render_performance.py` - Render performance tests

#### MCP Module Tests
- `test_mcp_overall.py` - Comprehensive MCP module testing
- `test_mcp_tools.py` - MCP tool execution tests
- `test_mcp_transport.py` - MCP transport layer tests
- `test_mcp_integration.py` - MCP integration tests
- `test_mcp_performance.py` - MCP performance tests

#### Audio Module Tests
- `test_audio_overall.py` - Comprehensive audio module testing
- `test_audio_sapf.py` - SAPF audio generation tests
- `test_audio_generation.py` - Audio generation tests
- `test_audio_integration.py` - Audio integration tests

#### Visualization Module Tests
- `test_visualization_overall.py` - Comprehensive visualization module testing
- `test_visualization_matrices.py` - Matrix visualization tests
- `test_visualization_ontology.py` - Ontology visualization tests

#### Pipeline Module Tests
- `test_pipeline_overall.py` - Comprehensive pipeline module testing
- `test_pipeline_integration.py` - Pipeline integration tests
- `test_pipeline_orchestration.py` - Pipeline orchestration tests
- `test_pipeline_performance.py` - Pipeline performance tests
- `test_pipeline_recovery.py` - Pipeline recovery tests
- `test_pipeline_scripts.py` - Pipeline script tests
- `test_pipeline_infrastructure.py` - Pipeline infrastructure tests
- `test_pipeline_functionality.py` - Pipeline functionality tests

#### Export Module Tests
- `test_export_overall.py` - Comprehensive export module testing

#### Execute Module Tests
- `test_execute_overall.py` - Comprehensive execute module testing

#### LLM Module Tests
- `test_llm_overall.py` - Comprehensive LLM module testing

#### Ontology Module Tests
- `test_ontology_overall.py` - Comprehensive ontology module testing

#### Website Module Tests
- `test_website_overall.py` - Comprehensive website module testing

#### Report Module Tests
- `test_report_overall.py` - Comprehensive report module testing
- `test_report_generation.py` - Report generation tests
- `test_report_integration.py` - Report integration tests
- `test_report_formats.py` - Report format tests

#### Environment Module Tests
- `test_environment_overall.py` - Comprehensive environment module testing
- `test_environment_dependencies.py` - Environment dependency tests
- `test_environment_integration.py` - Environment integration tests
- `test_environment_python.py` - Python environment tests
- `test_environment_system.py` - System environment tests

#### Comprehensive Tests
- `test_comprehensive_api.py` - Comprehensive API testing
- `test_core_modules.py` - Core module integration tests
- `test_fast_suite.py` - Fast test suite
- `test_main_orchestrator.py` - Main orchestrator tests
- `test_coverage_overall.py` - Coverage tests
- `test_performance_overall.py` - Performance tests
- `test_unit_overall.py` - Unit tests

## Test Runner Configuration

The test runner (`runner.py`) is configured with comprehensive test categories:

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
    # ... additional categories for all modules
}
```

## Test Execution

### Running All Tests
```bash
# Run all tests
python src/2_tests.py

# Run with specific options
python src/2_tests.py --verbose --parallel --coverage
```

### Running Module-Specific Tests
```bash
# Run GNN tests only
python src/2_tests.py --category gnn

# Run multiple categories
python src/2_tests.py --category gnn,render,audio
```

### Running Fast Tests
```bash
# Run fast test suite
python src/tests/run_fast_tests.py
```

## Test Utilities

### Shared Test Utilities (`src/utils/test_utils.py`)
- `TEST_CATEGORIES` - Test category definitions
- `TEST_STAGES` - Test execution stages
- `TEST_CONFIG` - Test configuration
- `is_safe_mode()` - Safe mode detection
- `setup_test_environment()` - Test environment setup
- `create_sample_gnn_content()` - Sample GNN content creation
- `performance_tracker()` - Performance tracking decorator
- `get_memory_usage()` - Memory usage monitoring
- `assert_file_exists()` - File existence assertions
- Report generation functions

### Test Fixtures (`conftest.py`)
- `project_root` - Project root directory
- `src_dir` - Source directory
- `test_dir` - Test directory
- `safe_filesystem` - Safe filesystem operations
- `sample_gnn_files` - Sample GNN files
- `isolated_temp_dir` - Isolated temporary directory
- `comprehensive_test_data` - Comprehensive test data

## Test Markers

### Available Markers
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests
- `@pytest.mark.safe_to_fail` - Tests that can safely fail
- `@pytest.mark.fast` - Fast tests

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

## Best Practices

### Test Organization
1. **Module-Based Structure**: Each module has its own test files
2. **Comprehensive Coverage**: Each module has an `_overall.py` test file
3. **Specialized Testing**: Additional test files for specific areas
4. **Integration Testing**: Cross-module integration tests

### Test Writing
1. **No Mocks**: Do not use mocking frameworks or monkeypatches to simulate behavior. Execute real methods and code paths.
2. **Import Error Handling**: Wrap imports in try/except blocks; skip if optional deps missing.
3. **Comprehensive Assertions**: Test both success and failure cases against real artifacts.
4. **Performance Monitoring**: Use performance tracking for slow operations

### Test Execution
1. **Category-Based**: Run tests by module category
2. **Parallel Execution**: Use parallel execution for faster results
3. **Resource Monitoring**: Monitor resource usage during execution
4. **Error Recovery**: Handle errors gracefully with fallback mechanisms

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

This test infrastructure provides a solid foundation for comprehensive testing of the GNN Processing Pipeline, with modular organization, parallel execution, and comprehensive coverage of all major components. 

## References

- Project overview: ../../README.md
- Comprehensive docs: ../../DOCS.md
- Architecture guide: ../../ARCHITECTURE.md
- Pipeline details: ../../doc/pipeline/README.md