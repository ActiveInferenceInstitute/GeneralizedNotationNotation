"""Real-behavior tests for ``src/gnn/pipeline/pipeline_validator.py``.

The runtime integration tester is the only consumer chain behind
``gnn.pipeline.health_check`` (its constructor is invoked by the health
check's integration probe), so its pure decision logic is pinned here.
The subprocess-driven surfaces (``test_pipeline_execution``) are
pipeline-marked behavior exercised by the pipeline suite, not unit
tests.
"""

from __future__ import annotations

import pytest

from gnn.pipeline.pipeline_validator import PipelineValidator

pytestmark = pytest.mark.filterwarnings("default")


def test_validator_constructs_without_subprocess() -> None:
    """Construction must succeed quietly and prepare state (health_check
    calls ``PipelineValidator(verbose=False)`` during integration probe).
    """
    validator = PipelineValidator(verbose=False)

    assert validator.validation_results == {}
    assert validator.dependency_manager is not None


def test_calculate_overall_health_transitions() -> None:
    """Empty -> unknown; all healthy -> healthy; any failed -> failed;
    a mix without failures -> degraded.
    """
    validator = PipelineValidator(verbose=False)

    assert validator._calculate_overall_health({}) == "unknown"
    assert (
        validator._calculate_overall_health(
            {"a": {"status": "healthy"}, "b": {"status": "healthy"}}
        )
        == "healthy"
    )
    assert (
        validator._calculate_overall_health(
            {"a": {"status": "healthy"}, "b": {"status": "failed"}}
        )
        == "failed"
    )
    assert (
        validator._calculate_overall_health(
            {"a": {"status": "healthy"}, "b": {"status": "degraded"}}
        )
        == "degraded"
    )
