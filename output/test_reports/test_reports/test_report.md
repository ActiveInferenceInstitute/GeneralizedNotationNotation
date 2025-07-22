# Test Execution Report

**Generated**: 2025-07-22 05:42:42
**Status**: ❌ FAILED
**Exit Code**: 2

## Test Configuration

- **Test Mode**: Regular Tests
- **Verbose Output**: True
- **Parallel Execution**: False
- **Coverage Enabled**: True

## Test Statistics

- **Total Tests**: 350
- **Passed**: ✅ 0
- **Failed**: ❌ 0
- **Skipped**: ⏭️ 0
- **Errors**: 🚨 4
- **Warnings**: ⚠️ 0
- **Xfailed**: 0
- **Xpassed**: 0
- **Deselected**: 11

**Success Rate**: 0.0%
**Failure Rate**: 1.1%
**Execution Time**: 0.9 seconds

## Test Dependencies

- ✅ **pytest** (v8.4.1)
- ✅ **pytest-cov** (v6.2.1)
- ❌ **pytest-json-report** (vN/A)
- ❌ **pytest-xdist** (vN/A)
- ✅ **coverage** (v7.9.2)
- ❌ **mock** (vN/A)
- ✅ **psutil** (v7.0.0)

## Generated Reports

- **Xml Report**: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/pytest_report.xml`
- **Markdown Report**: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/test_report.md`

## Execution Details

- **Command**: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation/.venv/bin/python -m pytest --verbose --tb=short --junitxml=/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/pytest_report.xml --maxfail=20 --durations=15 --disable-warnings -c/Users/4d/Documents/GitHub/GeneralizedNotationNotation/pytest.ini --cov=src/gnn --cov=src/pipeline --cov=src/utils --cov-report=html:/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/coverage --cov-report=json:/Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/test_coverage.json --cov-report=term-missing --cov-fail-under=0 --cov-config=/Users/4d/Documents/GitHub/GeneralizedNotationNotation/.coveragerc --cov-branch -m not slow /Users/4d/Documents/GitHub/GeneralizedNotationNotation/src/tests`
- **Working Directory**: `/Users/4d/Documents/GitHub/GeneralizedNotationNotation`
- **Timeout**: 600 seconds

## Raw Output Preview

### Standard Output (last 50 lines)
```
<frozen posixpath>:169: in basename
    ???
E   RecursionError: maximum recursion depth exceeded
__________________ ERROR collecting src/tests/test_parsers.py __________________
src/tests/test_parsers.py:14: in <module>
    from gnn.parsers.markdown_parser import MarkdownGNNParser
src/gnn/__init__.py:59: in <module>
    from .mcp import (
src/gnn/mcp.py:36: in <module>
    from .processors import (
src/gnn/processors.py:23: in <module>
    from utils.path_utils import get_relative_path_if_possible
src/utils/__init__.py:24: in <module>
    from .logging_utils import (
src/utils/logging_utils.py:26: in <module>
    from .performance_tracker import PerformanceTracker, performance_tracker
src/utils/performance_tracker.py:17: in <module>
    import psutil
.venv/lib/python3.13/site-packages/psutil/__init__.py:39: in <module>
    from . import _common
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1322: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:1262: in _find_spec
    ???
.venv/lib/python3.13/site-packages/_pytest/assertion/rewrite.py:100: in find_spec
    if self._early_rewrite_bailout(name, state):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.13/site-packages/_pytest/assertion/rewrite.py:211: in _early_rewrite_bailout
    path = PurePath(*parts).with_suffix(".py")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/lib/python3.13/pathlib/_abc.py:227: in with_suffix
    stem = self.stem
           ^^^^^^^^^
/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/lib/python3.13/pathlib/_abc.py:197: in stem
    name = self.name
           ^^^^^^^^^
/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/lib/python3.13/pathlib/_local.py:349: in name
    tail = self._tail
           ^^^^^^^^^^
E   RecursionError: maximum recursion depth exceeded
- generated xml file: /Users/4d/Documents/GitHub/GeneralizedNotationNotation/output/test_reports/test_reports/pytest_report.xml -
=========================== short test summary info ============================
ERROR src/tests/test_comprehensive_api.py - RecursionError: maximum recursion...
ERROR src/tests/test_export.py - RecursionError: maximum recursion depth exce...
ERROR src/tests/test_gnn_type_checker.py - RecursionError: maximum recursion ...
ERROR src/tests/test_parsers.py - RecursionError: maximum recursion depth exc...
!!!!!!!!!!!!!!!!!!! Interrupted: 4 errors during collection !!!!!!!!!!!!!!!!!!!!
================= 11 deselected, 1 warning, 4 errors in 0.94s ==================
```

### Standard Error (last 20 lines)
```

```
