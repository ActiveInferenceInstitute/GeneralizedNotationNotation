"""Factor-count, role, and provenance-stamping contract for the actinf exemplar.

Contract under test (pinned decisions #4, #5):
- to_dict() gains gnn_version, extraction_schema_version="1.0.0",
  num_state_factors / num_observation_modalities / num_control_factors, and
  dimension_provenance ({name: {"value", "source"}}).
- Descriptor entries gain "role" in {"factor", "bookkeeping"}; the existing
  descriptor lists keep every entry (s/s_prime and pi/u remain, roles differ).
- actinf_pomdp_agent.md: exactly one state factor, one observation modality,
  one control factor (bookkeeping excluded: s_prime, pi; u IS the factor).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gnn.pomdp_extractor import extract_pomdp_from_file

REPO = Path(__file__).resolve().parents[3]
ACTINF_EXEMPLAR = REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"


def _actinf_dict() -> dict[str, Any]:
    """Extract the exemplar in-test (not a fixture: call-phase failures must
    be xfail-able while the contract is landing in parallel)."""
    spec = extract_pomdp_from_file(ACTINF_EXEMPLAR, strict_validation=True)
    assert spec is not None
    assert not isinstance(spec, tuple)
    return spec.to_dict()


def _roles_by_name(descriptors: Any) -> dict[str, Any]:
    assert isinstance(descriptors, list) and descriptors, (
        f"expected non-empty descriptor list, got: {descriptors!r}"
    )
    return {d["name"]: d.get("role") for d in descriptors}


def test_num_state_factors_counts_factors_not_aliases() -> None:
    """s_prime is bookkeeping for s: exactly ONE state factor."""
    d = _actinf_dict()
    assert d["num_state_factors"] == 1


def test_num_observation_modalities() -> None:
    d = _actinf_dict()
    assert d["num_observation_modalities"] == 1


def test_num_control_factors_u_is_the_factor() -> None:
    """pi is policy bookkeeping; u is the single control factor."""
    d = _actinf_dict()
    assert d["num_control_factors"] == 1


def test_state_factor_roles() -> None:
    roles = _roles_by_name(_actinf_dict()["state_factors"])
    assert roles.get("s") == "factor"
    assert roles.get("s_prime") == "bookkeeping"


def test_control_factor_roles() -> None:
    roles = _roles_by_name(_actinf_dict()["control_factors"])
    assert roles.get("u") == "factor"
    assert roles.get("π") == "bookkeeping"


def test_role_values_are_restricted() -> None:
    """Every role value must be 'factor' or 'bookkeeping'."""
    d = _actinf_dict()
    for key in ("state_factors", "observation_modalities", "control_factors"):
        for descriptor in d[key]:
            assert descriptor.get("role") in {"factor", "bookkeeping"}, (
                f"{key} descriptor {descriptor!r} has invalid role"
            )


def test_all_descriptors_kept_not_removed() -> None:
    """Role tagging must not shrink the descriptor lists (non-breaking)."""
    d = _actinf_dict()
    names = {descriptor["name"] for descriptor in d["state_factors"]}
    assert {"s", "s_prime"} <= names
    control_names = {descriptor["name"] for descriptor in d["control_factors"]}
    assert {"u", "π"} <= control_names


def test_to_dict_schema_stamps() -> None:
    d = _actinf_dict()
    assert d["extraction_schema_version"] == "1.0.0"
    assert isinstance(d["gnn_version"], str) and d["gnn_version"]


def test_to_dict_dimension_provenance() -> None:
    provenance = _actinf_dict()["dimension_provenance"]
    assert isinstance(provenance, dict) and provenance
    for name, entry in provenance.items():
        assert "value" in entry, f"dimension_provenance[{name}] missing value"
        assert "source" in entry, f"dimension_provenance[{name}] missing source"
