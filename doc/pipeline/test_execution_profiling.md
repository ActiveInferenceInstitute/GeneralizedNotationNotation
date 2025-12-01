# Test Execution Profiling Guide

**Date**: November 19, 2025  
**Status**: 📋 PLANNING (Priority 2)

## Overview

This guide documents strategies for profiling test execution, identifying bottlenecks, and optimizing test suite performance for the GNN pipeline.

## Current Performance Baseline

### Test Suite Execution Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Timeout** | 600s | ✅ Improved |
| **Expected Pass Rate** | 95%+ | ✅ Target |
| **Peak Memory** | ~100-300MB | ✅ Efficient |
| **Pipeline Share** | ~30-40% | ⏳ Target |
| **Per-test Timeout** | 120s | ✅ Adequate |

## Profiling Strategy

### 1. Test Categorization

**Goal**: Identify which tests consume the most time

**Implementation**:
```python
# Add pytest markers to classify tests
@pytest.mark.fast       # < 1 second
@pytest.mark.standard   # 1-10 seconds
@pytest.mark.slow       # 10-60 seconds
@pytest.mark.very_slow  # > 60 seconds
```

**Usage**:
```bash
# Run only fast tests (< 1s each)
pytest -m fast

# Run standard + fast tests
pytest -m "fast or standard"

# Skip very slow tests
pytest -m "not very_slow"
```

### 2. Execution Time Tracking

**Goal**: Measure individual test execution times

**Tool**: pytest-benchmark or built-in timing

**Configuration**:
```bash
# Show slowest 20 tests
pytest --durations=20

# Save timing data
pytest --json-report --json-report-file=test_timings.json
```

### 3. Test Count Estimation

**Goal**: Estimate test count before execution

**Approach**:
```python
import subprocess
result = subprocess.run(
    ["pytest", "--collect-only", "-q"],
    capture_output=True
)
test_count = len([line for line in result.stdout.split('\n') if '::test_' in line])
print(f"Estimated tests: {test_count}")
```

### 4. Performance Regression Detection

**Goal**: Alert when tests become slower

**Baseline Tracking**:
```json
{
    "timestamp": "2025-11-19",
    "total_duration": 300,
    "test_count": 592,
    "average_per_test": 0.507,
    "slowest_tests": [
        {"name": "test_pipeline_main", "duration": 120},
        {"name": "test_llm_integration", "duration": 85}
    ]
}
```

## Implementation Plan

### Phase 1: Baseline Measurement (Week 1)

1. Run comprehensive test suite with full timing
2. Record baseline metrics
3. Identify top 10 slowest tests
4. Document slow test categories

### Phase 2: Test Profiling (Week 2-3)

1. Add pytest markers for test categories
2. Implement automatic timing collection
3. Generate profiling reports
4. Analyze time distribution

### Phase 3: Optimization (Week 4-5)

1. Identify optimization opportunities
2. Refactor slow tests
3. Implement parallel execution
4. Measure improvement

### Phase 4: Continuous Monitoring (Week 6+)

1. Automated performance regression detection
2. CI/CD integration
3. Performance dashboards
4. Alert on regressions

## Quick Start: Manual Profiling

### Find Slowest Tests
```bash
# Show 20 slowest tests
pytest --durations=20 src/tests

# Save to file
pytest --durations=0 src/tests > test_timings.txt
```

### Profile Specific Category
```bash
# Profile GNN tests
time pytest src/tests/test_gnn_overall.py

# Profile render tests with durations
pytest --durations=10 src/tests/test_render_overall.py
```

### Estimate Test Suite Size
```bash
# Count all tests
pytest --collect-only -q src/tests | wc -l

# Count by category
pytest --collect-only -q src/tests/test_pipeline_*.py | wc -l

# Count marked as slow
pytest --collect-only -m slow -q src/tests | wc -l
```

## Test Optimization Techniques

### 1. Fixture Optimization

**Before**:
```python
@pytest.fixture
def large_dataset():
    return load_large_file()  # Runs for every test
```

**After**:
```python
@pytest.fixture(scope="module")
def large_dataset():
    return load_large_file()  # Runs once per module
```

### 2. Parallel Execution

**Installation**:
```bash
pip install pytest-xdist
```

**Usage**:
```bash
# Run tests in parallel (auto-detects CPU count)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

### 3. Test Isolation

**Strategy**: Run related tests together to share setup

```python
# Group related tests in a class
class TestGNNProcessing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.gnn = load_gnn()  # Shared setup
    
    def test_parsing(self):
        # Uses self.gnn
        pass
    
    def test_validation(self):
        # Uses self.gnn  
        pass
```

### 4. Test Skipping Strategies

**Mark slow tests as optional in CI**:
```python
@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skip in CI for speed"
)
def test_slow_operation():
    pass
```

**Skip by marker in pipeline**:
```bash
# Skip very slow tests
export SKIP_MARKERS="very_slow,benchmark"
pytest -m "not very_slow and not benchmark"
```

## Monitoring and Alerts

### Performance Thresholds

| Threshold | Action |
|-----------|--------|
| Single test > 120s | Investigate - may timeout |
| Test suite > 600s | Warn - approaching timeout |
| Test suite > 1200s | Alert - exceeds recommended |
| Average per-test > 1s | Review for optimization |

### Alert Configuration

```python
# In test runner
if total_time > 600:
    logger.warning(f"⚠️  Test suite exceeding timeout budget: {total_time}s")
    if total_time > 1200:
        logger.error(f"❌ Test suite exceeded safe limit: {total_time}s")
```

## Troubleshooting Slow Tests

### Identify Problematic Tests
```bash
# Find tests taking > 10 seconds
pytest --durations=0 | grep -E "10\.[0-9]+ |[0-9]{2}\.[0-9]+ "

# Get test execution order
pytest -v --collect-only | grep "::test_"
```

### Debug Specific Test
```bash
# Run with verbose output
pytest -vv tests/test_slow.py::test_name

# Run with profiling
pytest --profile tests/test_slow.py::test_name

# Run with pdb debugger
pytest --pdb tests/test_slow.py::test_name
```

### Common Slow Test Causes
1. **Large data loading**: Use fixtures with module scope
2. **Network calls**: Mock or use responses library
3. **File I/O**: Use temporary directories
4. **Database operations**: Use in-memory databases
5. **Sleep calls**: Replace with time mocking

## Performance Targets

### For Different Environments

| Environment | Target Timeout | Test Count | Avg per-test |
|-------------|----------------|-----------|--------------|
| Development | 600s | 500+ | <1.2s |
| CI/CD Fast | 300s | 100-150 | <2s |
| CI/CD Full | 1800s | 500+ | <3.6s |
| Local | 900s | 500+ | <1.8s |

## Reporting

### Generate Performance Report
```bash
# Collect metrics
pytest --duration=0 > test_timings.txt
pytest --cov --cov-report=term > coverage.txt

# Create summary
echo "## Test Execution Summary" > report.md
echo "- Tests: $(grep -c '::test_' test_timings.txt)" >> report.md
echo "- Duration: $(tail -1 test_timings.txt)" >> report.md
echo "- Coverage: $(grep 'TOTAL' coverage.txt)" >> report.md
```

## References

- **Related Files**: `src/tests/runner.py`, `src/tests/conftest.py`
- **Previous Work**: `doc/pipeline/test_timeout_improvements.md`
- **Pytest Docs**: https://docs.pytest.org/en/stable/index.html

## Next Steps

1. ✅ Implement timeout improvements (DONE)
2. ⏳ Add test categorization with markers
3. ⏳ Implement timing collection
4. ⏳ Create performance baselines
5. ⏳ Set up regression detection
6. ⏳ Optimize slowest tests






