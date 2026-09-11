# `gnn/utils/testing/` — agent contract

Concern-package home for the test-harness family, extracted from
`testing_utils.py` in S2-33 Step 1 (design:
[`docs/development/utils_split_design.md`](../../../../docs/development/utils_split_design.md)
§6). The old module path `gnn/utils/testing_utils.py` remains as a
`DeprecationWarning` facade for the deprecation window; new code imports
from this package.

## Layout

| Module | Owns |
|---|---|
| [`constants.py`](constants.py) | `PROJECT_ROOT` / `SRC_DIR` / `TEST_DIR`, `TEST_CATEGORIES`, `TEST_STAGES`, `COVERAGE_TARGETS`, `TEST_CONFIG` |
| [`runner.py`](runner.py) | `TestRunner`/`TestResult`/`TestCategory`/`TestStage`/`CoverageTarget`, `run_tests`/`run_test_category`/`run_test_stage`, `run_all_tests`, `run_fast/standard/slow/performance/coverage_tests` |
| [`fixtures.py`](fixtures.py) | `get_test_args`, `get_sample_pipeline_arguments`, `get_step_metadata_dict`, the `create_*`/`get_sample_*` families, `get_test_filesystem_structure` |
| [`reports.py`](reports.py) | the `get_test_*` accessor family, `validate_report_data`, `run_all_tests_mcp`, `generate_html/markdown/json_report_file`, `generate_comprehensive_report` |
| [`environment.py`](environment.py) | `validate/setup/cleanup_test_environment`, coverage + dependency helpers, test configuration accessors |
| [`perf.py`](perf.py) | `performance_tracker`, `track_peak_memory`, `with_resource_limits` (`_PerformanceTracker` stays private — no facade alias, design §4.3.4) |
| [`assertions.py`](assertions.py) | `assert_file_exists`, `assert_valid_json`, `assert_directory_structure` |

## Invariants

- The family `__init__.py` re-exports all public names eagerly (real
  objects, design §4.3.1) and must never import psutil or matplotlib at
  module scope (`tests/tests/test_light_import.py` is the gate).
- `_EXPORT_MAP` keys in `gnn/utils/__init__.py` are frozen (113 keys,
  golden-asserted); this package's leaf modules are the *values* for the
  35 `testing.*` entries.
- Cross-family imports go through leaf modules (`pipeline_arguments`,
  `resource_manager`), never the `gnn.utils` facade (design I5).
- `_PerformanceTracker` is private to `perf.py` (§4.3.4).

## Tests

- [`tests/tests/test_infrastructure_exports.py`](../../../../tests/tests/test_infrastructure_exports.py)
  asserts the frozen `_EXPORT_MAP` surface and resolves every export.
- [`tests/tests/test_light_import.py`](../../../../tests/tests/test_light_import.py)
  asserts `import gnn.utils` stays light (no psutil/matplotlib).
- [`tests/utils/test_shared_helpers.py`](../../../../tests/utils/test_shared_helpers.py)
  pins the old-path facade's delegation.
