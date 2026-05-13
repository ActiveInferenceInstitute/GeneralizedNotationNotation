# GNN Pipeline Test Suite - Comprehensive Summary

**Last Updated**: 2026-05-13  
**Status**: ✅ Production Ready  
**Test Infrastructure Version**: 2.0.1

---

## Executive Summary

The GNN Processing Pipeline test suite provides comprehensive coverage across all 25 pipeline steps and the maintained source modules under `src/`. Tests prioritize real code paths, representative data, and explicit skip behavior for unavailable optional services.

### Key Metrics

- **Total Test Files**: 166 `test_*.py` files
- **Directory Layout**: 33 first-level directories under `src/tests/`; 31 contain direct test files
- **Root-Level Tests**: 28 `test_*.py` files at `src/tests/`
- **Subdirectory Tests**: 138 `test_*.py` files under module/helper directories
- **Collected Tests**: 2,271 with `uv run pytest src/tests/ --collect-only -q --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py` (2026-05-13)
- **Execution Time**: depends on optional backends and selected markers; use current run output for pass/skip counts

---

## Test Infrastructure Architecture

### Component Structure

```
src/tests/
├── ../2_tests.py           # Thin orchestrator (CLI entry point)
├── runner.py               # Core test execution logic
├── conftest.py             # Pytest fixtures and configuration
├── __init__.py             # Module exports and utilities
├── README.md               # Comprehensive documentation
├── AGENTS.md               # Technical API documentation
├── <module>/test_*.py      # 138 module/helper test files
└── test_*.py               # 28 cross-cutting root test files
```

### Execution Modes

1. **Fast Tests** (`--fast-only`): Default for pipeline
   - Duration: 1-3 minutes
   - Coverage: Essential functionality
   - Use case: Quick validation during development

2. **Comprehensive Tests** (`--comprehensive`): Full suite
   - Duration: 5-15 minutes
   - Coverage: All tests including slow/performance
   - Use case: Pre-release validation, CI/CD

3. **Reliable Tests** (recovery): Essential only
   - Duration: ~90 seconds
   - Coverage: Critical path validation
   - Use case: Emergency validation

---

## Test Quality Standards

### No Simulated Usage Policy

All tests follow strict "no mocks" policy:
- No `unittest.mock`-style replacement of production behavior for core paths
- Real code paths executed in module and integration tests
- Real data or representative fixtures used throughout
- Real dependencies are used when available; optional integrations skip cleanly when unavailable
- File-based assertions on real artifacts

### Real Implementation Testing

- **Real Methods**: All tests execute actual module functions
- **Real Data**: Tests use representative GNN files and data structures
- **Real Dependencies**: Tests validate actual API responses and service interactions
- **Real File I/O**: Tests assert on real file outputs in `output/` directories
- **Real Subprocesses**: Pipeline tests run actual numbered scripts via subprocess

### Comprehensive Error Handling

- **Error Scenarios**: Tests cover all error conditions with real failure modes
- **Graceful Degradation**: Tests validate recovery behavior when dependencies unavailable
- **Recovery Testing**: Tests verify error recovery mechanisms
- **Timeout Handling**: Tests include timeout scenarios and resource limits

---

## Coverage Layout

The suite mirrors the source tree: module-focused tests live in `src/tests/<module>/`, while root-level `src/tests/test_*.py` files cover cross-cutting environment, coverage, and runner behavior. Mechanical AGENTS/README coverage for maintained test subdirectories is enforced by `doc/development/docs_audit.py --strict`.

---

## Test Categories and Markers

### Test Categories (20+)

Organized by module and functionality:
- `gnn` - GNN processing and validation
- `render` - Code generation for simulation frameworks
- `mcp` - Model Context Protocol integration
- `audio` - Audio generation and sonification
- `visualization` - Graph and matrix visualization
- `pipeline` - Pipeline orchestration and infrastructure
- `export` - Multi-format export
- `execute` - Simulation execution
- `llm` - LLM-enhanced analysis
- `ontology` - Active Inference ontology processing
- And 10+ more categories...

### Test Markers (25+)

For selective execution:
- `@pytest.mark.fast` - Fast tests (< 1 second)
- `@pytest.mark.slow` - Slow tests (minutes)
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.safe_to_fail` - Safe-to-fail tests (336 functions)
- `@pytest.mark.requires_gpu` - GPU-required tests
- `@pytest.mark.requires_network` - Network-required tests
- And 15+ more markers...

---

## Test Execution Examples

### Run Fast Tests (Default)
```bash
python src/2_tests.py --fast-only --verbose
```

### Run Comprehensive Test Suite
```bash
python src/2_tests.py --comprehensive --verbose
```

### Run Specific Module Tests
```bash
pytest src/tests/test_gnn_overall.py -v
pytest src/tests/test_render_overall.py -v
```

### Run by Marker
```bash
pytest -m fast  # Run only fast tests
pytest -m integration  # Run only integration tests
pytest -m "not slow"  # Exclude slow tests
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

---

## Test Results and Reporting

### Output Structure

```
output/2_tests_output/
├── test_execution_report.json      # Structured test results
├── pytest_comprehensive_output.txt # Full pytest output
├── test_summary.md                 # Human-readable summary
├── coverage.xml                    # Coverage data (if enabled)
└── test_details/                   # Detailed per-category results
    ├── gnn_tests.json
    ├── render_tests.json
    └── ...
```

### Report Formats

- **JSON**: Machine-readable structured data
- **Markdown**: Human-readable summaries
- **HTML**: Coverage reports (if enabled)
- **Text**: Full pytest output for debugging

---

## Performance Characteristics

### Execution Times

- **Fast Tests**: 1-3 minutes (default)
- **Comprehensive Tests**: 5-15 minutes
- **Reliable Tests**: ~90 seconds (recovery)
- **Individual Test**: < 1 second average

### Resource Usage

- **Memory**: ~100-300MB during execution
- **CPU**: Parallel execution supported
- **Disk**: Temporary test artifacts cleaned up automatically

### Scalability

- **Test Discovery**: Handles 54 files efficiently
- **Parallel Execution**: Category-based parallelization
- **Timeout Handling**: Per-category timeouts prevent hangs
- **Resource Monitoring**: Built-in memory and CPU tracking

---

## Best Practices

### Writing Tests

1. **No Mocks**: Always use real implementations
2. **Real Data**: Use representative test data
3. **Comprehensive Assertions**: Test both success and failure cases
4. **Error Handling**: Wrap optional dependencies in try/except
5. **Documentation**: Include clear docstrings explaining test purpose

### Test Organization

1. **Module-Based**: One test file per module (`test_MODULE_overall.py`)
2. **Specialized**: Additional files for specific areas
3. **Integration**: Separate files for cross-module testing
4. **Naming**: Follow `test_*.py` convention

### Test Execution

1. **Fast First**: Run fast tests during development
2. **Comprehensive Before Release**: Run full suite before releases
3. **Selective Execution**: Use markers to run specific test subsets
4. **Parallel Execution**: Use parallel execution for faster results

---

## Troubleshooting

### Common Issues

1. **Collection Errors**: Check for missing dependencies or syntax errors
2. **Import Errors**: Verify `src/` is in Python path
3. **Timeout Errors**: Increase timeout or run fast tests only
4. **Memory Errors**: Run tests sequentially or reduce test scope
5. **Test Failures**: Review `pytest_comprehensive_output.txt` for details

### Getting Help

1. Check test output files in `output/2_tests_output/`
2. Review `test_execution_report.json` for execution summary
3. Check `pytest_comprehensive_output.txt` for detailed error messages
4. Verify environment variables are set correctly
5. Ensure all dependencies are installed

---

## Future Enhancements

### Planned Improvements

1. **Coverage Analysis**: Enhanced code coverage tracking and reporting
2. **Performance Benchmarking**: Automated performance regression detection
3. **CI/CD Integration**: Automated test execution in CI/CD pipelines
4. **Test Data Management**: Centralized test data fixtures
5. **Visual Test Reports**: Enhanced HTML reporting with visualizations

### Module Expansion

- Additional tests for new modules as they're added
- Enhanced integration tests for cross-module workflows
- Expanded performance tests for scalability validation
- Additional error recovery scenarios

---

## References

- **Test Documentation**: `src/tests/README.md`
- **API Documentation**: `src/tests/AGENTS.md`
- **Pipeline Documentation**: `README.md`
- **Architecture Guide**: `ARCHITECTURE.md`
- **Project Overview**: `DOCS.md`

---

## Conclusion

The GNN Processing Pipeline test suite provides comprehensive, production-ready testing infrastructure with:

- 166 test files across root and module-specific directories
- 2,271 collected tests in the current command-of-record collect pass with Ollama integration tests ignored
- Real data and real implementations throughout core paths
- Comprehensive error handling and recovery testing
- Module coverage for all 25 pipeline steps
- AGENTS/README documentation coverage for maintained test subdirectories

The test infrastructure is mature, comprehensive, and ready for production use.
