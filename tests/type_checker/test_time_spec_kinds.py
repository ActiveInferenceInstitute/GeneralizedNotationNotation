#!/usr/bin/env python3
"""Typed ``## Time`` classification (Static | TimeVarying | RegimeSwitched).

Covers the typed time-spec semantics: ``classify_time_spec_kind`` returns the
``TimeSpecKind`` member, ``classify_time_spec`` keeps the legacy
Static/Dynamic/Hierarchical string contract (both dynamic kinds project to
``"Dynamic"``), and ``detect_time_dynamics`` shares one marker set with the
classification so the two can never disagree.
"""

from __future__ import annotations

from pathlib import Path

from gnn.type_checker.checking.sections import (
    TimeSpecKind,
    classify_time_spec,
    classify_time_spec_kind,
    detect_time_dynamics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCRETE_DIR = REPO_ROOT / "input" / "gnn_files" / "discrete"


def test_static_spec_classifies_static() -> None:
    assert classify_time_spec_kind("## Time\nStatic\n") is TimeSpecKind.STATIC
    assert classify_time_spec_kind("no time section") is TimeSpecKind.STATIC


def test_dynamic_marker_classifies_time_varying() -> None:
    assert (
        classify_time_spec_kind("## Time\nDynamic\nDiscreteTime=t\n")
        is TimeSpecKind.TIME_VARYING
    )
    assert (
        classify_time_spec_kind("## Time\ncontinuous-time\n")
        is TimeSpecKind.TIME_VARYING
    )
    assert (
        classify_time_spec_kind("## Time\ntime-varying\n") is TimeSpecKind.TIME_VARYING
    )


def test_regime_marker_classifies_regime_switched() -> None:
    content = "## Time\nRegimeSwitched\nDiscreteTime=t\nModelTimeHorizon=8\n"
    assert classify_time_spec_kind(content) is TimeSpecKind.REGIME_SWITCHED


def test_regime_wins_over_plain_dynamic_markers() -> None:
    content = "## Time\nDynamic\nRegimeSwitched\n"
    assert classify_time_spec_kind(content) is TimeSpecKind.REGIME_SWITCHED


def test_hierarchical_is_orthogonal_to_typed_kinds() -> None:
    """Typed kinds classify the time variation; the legacy string keeps the
    Hierarchical label winning."""
    content = "## Time\nHierarchical\nDynamic\n"
    assert classify_time_spec_kind(content) is TimeSpecKind.TIME_VARYING
    assert classify_time_spec(content) == "Hierarchical"


def test_classify_time_spec_legacy_string_contract() -> None:
    """Existing callers keep their exact Static/Dynamic/Hierarchical strings."""
    assert classify_time_spec("## Time\nStatic\n") == "Static"
    assert classify_time_spec("## Time\nDynamic\n") == "Dynamic"
    assert classify_time_spec("## Time\ncontinuous-time\n") == "Dynamic"
    assert classify_time_spec("## Time\nHierarchical\n") == "Hierarchical"
    assert classify_time_spec("## Time\nRegimeSwitched\n") == "Dynamic"
    assert classify_time_spec("no time section") == "Static"


def test_detect_time_dynamics_covers_regime_switched() -> None:
    """One marker set: regime-switched content is dynamic for both APIs."""
    assert detect_time_dynamics("## Time\nRegimeSwitched\n") is True
    assert detect_time_dynamics("## Time\nStatic\n") is False
    assert detect_time_dynamics("## Time\nStatic\n## Notes\ndynamic talk\n") is False


def test_classify_time_spec_kind_agrees_with_legacy_for_regime() -> None:
    """The classify/detect agreement invariant extends to regime content."""
    content = "## Time\nRegimeSwitched\nDiscreteTime=t\n"
    assert (classify_time_spec(content) != "Static") == detect_time_dynamics(content)


def test_checking_package_reexports_typed_time_spec() -> None:
    from gnn.type_checker.checking import (
        TimeSpecKind as reexported_kind,
    )
    from gnn.type_checker.checking import (
        classify_time_spec_kind as reexported_classifier,
    )

    assert reexported_kind is TimeSpecKind
    assert reexported_classifier is classify_time_spec_kind


def test_exemplar_time_varying_dynamics_classifies_time_varying() -> None:
    content = (DISCRETE_DIR / "time_varying_dynamics.md").read_text(encoding="utf-8")
    assert classify_time_spec_kind(content) is TimeSpecKind.TIME_VARYING
    assert classify_time_spec(content) == "Dynamic"


def test_exemplar_regime_switched_dynamics_classifies_regime_switched() -> None:
    content = (DISCRETE_DIR / "regime_switched_dynamics.md").read_text(encoding="utf-8")
    assert classify_time_spec_kind(content) is TimeSpecKind.REGIME_SWITCHED
    assert classify_time_spec(content) == "Dynamic"
