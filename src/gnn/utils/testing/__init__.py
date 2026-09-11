"""Test-harness concern package (S2-33 Step 1): canonical home of the code that
used to live in the ``gnn/utils/testing_utils.py`` grab-bag.

Re-exports the family's public names as real objects (not lazy — design
§4.3.1), so intra-family and test-suite reads use
``from gnn.utils.testing import TestRunner``. Import-weight note (I1): this
package is NOT imported by ``import gnn.utils`` — the top-level facade stays
lazy through its PEP 562 map (guarded by tests/tests/test_light_import.py).
Importing this package eagerly imports every leaf, which is the same cost the
old ``import gnn.utils.testing_utils`` paid.

Leaf inventory:
- constants: path anchors (PROJECT_ROOT/SRC_DIR/TEST_DIR) and the
  category/stage/coverage/configuration tables
- runner: ``TestRunner``/``TestResult``/... classes and the ``run_*`` entry points
- fixtures: ``get_test_args``, the ``create_*``/``get_sample_*`` families
- reports: the ``get_test_*`` accessor family + report generators
- environment: validate/setup/cleanup environment, coverage/dependency/config checks
- perf: ``performance_tracker``/``track_peak_memory``/``with_resource_limits``
  (``_PerformanceTracker`` stays private; design §4.3.4)
- assertions: the ``assert_*`` helpers

Cross-family imports go through leaf modules, never through any facade (I5).
"""

from gnn.utils.testing.assertions import (
    assert_directory_structure,
    assert_file_exists,
    assert_valid_json,
)
from gnn.utils.testing.constants import (
    COVERAGE_TARGETS,
    PROJECT_ROOT,
    SRC_DIR,
    TEST_CATEGORIES,
    TEST_CONFIG,
    TEST_DIR,
    TEST_STAGES,
)
from gnn.utils.testing.environment import (
    cleanup_test_environment,
    get_test_configuration,
    get_test_coverage,
    get_test_dependencies,
    get_test_environment,
    install_test_dependencies,
    setup_test_environment,
    validate_coverage_targets,
    validate_test_configuration,
    validate_test_dependencies,
    validate_test_environment,
)
from gnn.utils.testing.fixtures import (
    create_missing_test_files,
    create_sample_config,
    create_sample_gnn_content,
    create_sample_ontology,
    create_test_files,
    create_test_gnn_files,
    get_sample_pipeline_arguments,
    get_step_metadata_dict,
    get_test_args,
    get_test_filesystem_structure,
    is_safe_mode,
)
from gnn.utils.testing.perf import (
    performance_tracker,
    track_peak_memory,
    with_resource_limits,
)
from gnn.utils.testing.reports import (
    generate_comprehensive_report,
    generate_html_report_file,
    generate_json_report_file,
    generate_markdown_report_file,
    generate_test_report,
    get_test_artifacts,
    get_test_duration,
    get_test_logs,
    get_test_metadata,
    get_test_performance,
    get_test_progress,
    get_test_results,
    get_test_statistics,
    get_test_status,
    get_test_summary,
    get_test_timestamps,
    run_all_tests_mcp,
    validate_report_data,
)
from gnn.utils.testing.runner import (
    CoverageTarget,
    TestCategory,
    TestResult,
    TestRunner,
    TestStage,
    run_all_tests,
    run_coverage_tests,
    run_fast_tests,
    run_performance_tests,
    run_slow_tests,
    run_standard_tests,
    run_test_category,
    run_test_stage,
    run_tests,
)

__all__: list[str] = [
    "COVERAGE_TARGETS",
    "CoverageTarget",
    "PROJECT_ROOT",
    "SRC_DIR",
    "TEST_CATEGORIES",
    "TEST_CONFIG",
    "TEST_DIR",
    "TEST_STAGES",
    "TestCategory",
    "TestResult",
    "TestRunner",
    "TestStage",
    "assert_directory_structure",
    "assert_file_exists",
    "assert_valid_json",
    "cleanup_test_environment",
    "create_missing_test_files",
    "create_sample_config",
    "create_sample_gnn_content",
    "create_sample_ontology",
    "create_test_files",
    "create_test_gnn_files",
    "generate_comprehensive_report",
    "generate_html_report_file",
    "generate_json_report_file",
    "generate_markdown_report_file",
    "generate_test_report",
    "get_sample_pipeline_arguments",
    "get_step_metadata_dict",
    "get_test_args",
    "get_test_artifacts",
    "get_test_configuration",
    "get_test_coverage",
    "get_test_dependencies",
    "get_test_duration",
    "get_test_environment",
    "get_test_filesystem_structure",
    "get_test_logs",
    "get_test_metadata",
    "get_test_performance",
    "get_test_progress",
    "get_test_results",
    "get_test_statistics",
    "get_test_status",
    "get_test_summary",
    "get_test_timestamps",
    "install_test_dependencies",
    "is_safe_mode",
    "performance_tracker",
    "run_all_tests",
    "run_all_tests_mcp",
    "run_coverage_tests",
    "run_fast_tests",
    "run_performance_tests",
    "run_slow_tests",
    "run_standard_tests",
    "run_test_category",
    "run_test_stage",
    "run_tests",
    "setup_test_environment",
    "track_peak_memory",
    "validate_coverage_targets",
    "validate_report_data",
    "validate_test_configuration",
    "validate_test_dependencies",
    "validate_test_environment",
    "with_resource_limits",
]
