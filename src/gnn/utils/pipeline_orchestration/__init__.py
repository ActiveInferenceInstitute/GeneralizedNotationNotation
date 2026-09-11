"""Pipeline-orchestration concern package (S2-33 Step 3): canonical home of
the ``pipeline_*`` family plus ``base_processor``/``execution_utils`` that
used to live in the ``gnn/utils`` top-level grab-bag.

Re-exports the family's public names as real objects (not lazy — design
§4.3.1). Import-weight note (I1): this package is NOT imported by
``import gnn.utils`` — the top-level facade stays lazy through its PEP 562
map (guarded by tests/tests/test_light_import.py). Importing this package
eagerly imports every leaf, which is the same cost the old
``import gnn.utils.pipeline_monitor`` paid.

The package is deliberately NOT named ``pipeline/`` (risk R1: top-level
``gnn.pipeline`` already exists).

Leaf inventory:
- pipeline_monitor: ``PipelineMonitor``/``Alert``/``PipelineHealth`` and the
  health-report entry points (the ``pipeline_monitor`` module-level singleton
  instance is intentionally NOT re-exported here: binding it as a package
  attribute would shadow the ``pipeline_monitor`` submodule for
  ``from ... import pipeline_monitor`` module-object imports — I6 guard)
- pipeline_dependencies: ``PipelineDependencyManager`` and dependency probes
- base_processor: ``BaseProcessor``/``ProcessingResult``/``create_processor``
- pipeline_validator: pre-execution prerequisite/step-sequence validation
- pipeline_template: ``create_standardized_pipeline_script`` (+ the
  ``log_step_*`` re-export surface and ``UTILS_AVAILABLE`` flag)
- pipeline_step_dependencies: step/script dependency tables and resolvers
- execution_utils: ``execute_command_streaming``

Cross-family imports go through leaf modules, never through any facade (I5).
The old top-level paths (``gnn/utils/pipeline_monitor.py`` etc.) are
deprecation facades over this package; ``gnn/utils/pipeline.py`` remains the
self-declared compat entry and now delegates its argument imports here.
"""

from gnn.utils.pipeline_orchestration.base_processor import (
    BaseProcessor,
    ProcessingResult,
    create_processor,
)
from gnn.utils.pipeline_orchestration.execution_utils import (
    execute_command_streaming,
)
from gnn.utils.pipeline_orchestration.pipeline_dependencies import (
    DependencyResult,
    PipelineDependencyManager,
    StepDependencyInfo,
    get_pipeline_dependency_manager,
)
from gnn.utils.pipeline_orchestration.pipeline_monitor import (
    Alert,
    AlertLevel,
    HealthStatus,
    PipelineHealth,
    PipelineMonitor,
    StepMetrics,
    generate_pipeline_health_report,
    get_pipeline_health_status,
    record_step_execution,
    start_pipeline_monitoring,
    stop_pipeline_monitoring,
)
from gnn.utils.pipeline_orchestration.pipeline_step_dependencies import (
    PIPELINE_SCRIPT_STEPS,
    PIPELINE_STEP_DEPENDENCIES,
    PIPELINE_STEP_SCRIPTS,
    dependency_scripts_for_script,
    dependency_steps_for_step,
    normalize_script_name,
    resolve_step_dependencies,
    step_number_for_script,
)
from gnn.utils.pipeline_orchestration.pipeline_template import (
    UTILS_AVAILABLE,
    create_standardized_pipeline_script,
)
from gnn.utils.pipeline_orchestration.pipeline_validator import (
    check_pipeline_readiness,
    validate_pipeline_step_sequence,
    validate_step_outputs,
    validate_step_prerequisites,
)

__all__: list[str] = [
    "Alert",
    "AlertLevel",
    "BaseProcessor",
    "DependencyResult",
    "HealthStatus",
    "PIPELINE_SCRIPT_STEPS",
    "PIPELINE_STEP_DEPENDENCIES",
    "PIPELINE_STEP_SCRIPTS",
    "PipelineDependencyManager",
    "PipelineHealth",
    "PipelineMonitor",
    "ProcessingResult",
    "StepDependencyInfo",
    "StepMetrics",
    "UTILS_AVAILABLE",
    "check_pipeline_readiness",
    "create_processor",
    "create_standardized_pipeline_script",
    "dependency_scripts_for_script",
    "dependency_steps_for_step",
    "execute_command_streaming",
    "generate_pipeline_health_report",
    "get_pipeline_dependency_manager",
    "get_pipeline_health_status",
    "normalize_script_name",
    "record_step_execution",
    "resolve_step_dependencies",
    "start_pipeline_monitoring",
    "step_number_for_script",
    "stop_pipeline_monitoring",
    "validate_pipeline_step_sequence",
    "validate_step_outputs",
    "validate_step_prerequisites",
]
