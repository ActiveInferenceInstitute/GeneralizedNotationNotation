# Test Execution Report

**Generated**: 2025-07-16 08:43:43
**Status**: ❌ FAILED
**Exit Code**: 2

## Test Configuration

- **Test Mode**: Regular Tests
- **Verbose Output**: True
- **Parallel Execution**: False
- **Coverage Enabled**: True

## Test Dependencies

- ✅ **pytest** (v8.4.1)
- ✅ **pytest-cov** (v6.2.1)
- ❌ **pytest-json-report**
- ❌ **pytest-xdist**

## Generated Reports

- **Markdown Report**: `../output/test_reports/test_report.md`

## Execution Details

- **Command**: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --verbose --tb=short --junitxml=../output/test_reports/pytest_report.xml --maxfail=20 --durations=15 --disable-warnings --strict-markers -c/Users/4d/Documents/GitHub/GeneralizedNotationNotation/pytest.ini --cov=src --cov-report=html:../output/test_reports/coverage --cov-report=json:../output/test_reports/test_coverage.json --cov-report=term-missing --cov-fail-under=0 -m not slow /Users/4d/Documents/GitHub/GeneralizedNotationNotation/src/tests`
- **Working Directory**: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation`
- **Timeout**: 600 seconds

