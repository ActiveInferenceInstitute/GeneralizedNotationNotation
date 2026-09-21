"""GNN-side conformance with the GEO-INFER consumer contracts.

Each assertion mirrors one validation performed by the pinned GEO-INFER
consumer (checkout @ fc5dd1ac, the revision in .github/gnn-pair.json):

- GEO-INFER-ACT/src/geo_infer_act/core/gnn_contract.py
  (schema gnn-geo-infer/1, model_type categorical)
- GEO-INFER-ACT/src/geo_infer_act/core/gnn_gaussian_contract.py
  (schema gnn-geo-infer/2, model_type linear_gaussian)
- GEO-INFER-ACT/src/geo_infer_act/core/gnn_factored_contract.py
  (schema gnn-geo-infer/factored/1, model_type categorical_factored)

The paired validator (GEO-INFER-TEST/validate_gnn_interchange.py) round-trips
these same fixtures through the real consumer classes in CI; these tests pin
the producer side so emission drift fails here first, with the consumer rule
cited per assertion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from gnn.export.geo_infer import build_geo_infer_artifact
from gnn.export.geo_infer_factored import build_geo_infer_factored_artifact
from gnn.export.geo_infer_gaussian import build_geo_infer_gaussian_artifact

ROOT = Path(__file__).resolve().parents[2]
CATEGORICAL_SOURCE = ROOT / "input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md"
GAUSSIAN_SOURCE = ROOT / "tests/export/gaussian_rectangular.md"
FACTORED_SOURCE = ROOT / "tests/export/factored_example.json"

GAUSSIAN_UNITS = {
    "states": ["m", "m/s", "K"],
    "observations": ["m", "m/s"],
    "controls": ["N"],
}


def _assert_finite(values: list) -> None:
    array = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(array)), "consumer rejects non-finite leaves"


def _assert_axis0_stochastic(values: list, *, atol: float = 1e-8) -> None:
    """Consumer: nonnegative and stochastic along axis zero (gnn_contract.py:133-139)."""
    array = np.asarray(values, dtype=float)
    assert np.all(array >= 0)
    np.testing.assert_allclose(array.sum(axis=0), 1.0, rtol=0, atol=atol)


def test_categorical_artifact_matches_consumer_v1() -> None:
    """gnn-geo-infer/1 emission satisfies geo_infer_act.core.gnn_contract."""
    text = CATEGORICAL_SOURCE.read_text()
    artifact = build_geo_infer_artifact(text, step_seconds=60)

    # Exact top-level key set: gnn_contract.py:94-107 (extra/missing keys fatal).
    assert set(artifact) == {
        "schema_version",
        "model_type",
        "model_name",
        "dimensions",
        "matrices",
        "space",
        "time",
        "provenance",
    }
    # Byte-exact literals: gnn_contract.py:108-112.
    assert artifact["schema_version"] == "gnn-geo-infer/1"
    assert artifact["model_type"] == "categorical"
    assert isinstance(artifact["model_name"], str) and artifact["model_name"].strip()

    # Dimensions: exactly {states, observations, actions}, positive ints: 115-118.
    assert artifact["dimensions"] == {
        "states": 9,
        "observations": 9,
        "actions": 5,
    }

    # Matrices: exactly {A,B,C,D,E} with shapes A=(o,s) B=(s,s,u) C=(o,) D=(s,)
    # E=(u,): gnn_contract.py:122-126.
    matrices = artifact["matrices"]
    assert set(matrices) == {"A", "B", "C", "D", "E"}
    s, o, u = 9, 9, 5
    np.testing.assert_array_equal(np.asarray(matrices["A"]).shape, (o, s))
    np.testing.assert_array_equal(np.asarray(matrices["B"]).shape, (s, s, u))
    np.testing.assert_array_equal(np.asarray(matrices["C"]).shape, (o,))
    np.testing.assert_array_equal(np.asarray(matrices["D"]).shape, (s,))
    np.testing.assert_array_equal(np.asarray(matrices["E"]).shape, (u,))
    for name in ("A", "B", "D", "E"):
        _assert_finite(matrices[name])
        _assert_axis0_stochastic(matrices[name])  # gnn_contract.py:130-139
    _assert_finite(matrices["C"])  # C is log preferences: exempt from stochasticity

    # Space: exactly {kind, state_ids}; s unique nonempty strings in matrix
    # order: gnn_contract.py:140-148.
    assert set(artifact["space"]) == {"kind", "state_ids"}
    assert artifact["space"]["kind"] == "categorical"
    state_ids = artifact["space"]["state_ids"]
    assert len(state_ids) == s and len(set(state_ids)) == s
    assert all(isinstance(x, str) and x for x in state_ids)

    # Time: exactly {step_seconds}, finite positive, bool rejected: 162-165.
    assert artifact["time"] == {"step_seconds": 60}

    # Provenance: exactly {producer, source_sha256}; sha256 is lowercase hex
    # of the original source bytes: gnn_contract.py:166-175.
    assert set(artifact["provenance"]) == {"producer", "source_sha256"}
    assert artifact["provenance"]["producer"].strip()
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def test_gaussian_artifact_matches_consumer_v2() -> None:
    """gnn-geo-infer/2 emission satisfies gnn_gaussian_contract."""
    text = GAUSSIAN_SOURCE.read_text()
    artifact = build_geo_infer_gaussian_artifact(
        text, step_seconds=2, units=GAUSSIAN_UNITS
    )

    # Exact key set: gnn_gaussian_contract.py:32-46.
    assert set(artifact) == {
        "schema_version",
        "model_type",
        "model_name",
        "dimensions",
        "matrices",
        "initial_belief",
        "units",
        "time",
        "provenance",
    }
    # Byte-exact literals: gnn_gaussian_contract.py:47-51.
    assert artifact["schema_version"] == "gnn-geo-infer/2"
    assert artifact["model_type"] == "linear_gaussian"

    # Matrices: exactly {F,G,H,Q,R}; F=(n,n) G=(n,k) H=(m,n) Q=(n,n) R=(m,m):
    # gnn_gaussian_contract.py:61-62.
    n, m, k = 3, 2, 1
    assert artifact["dimensions"] == {
        "states": n,
        "observations": m,
        "controls": k,
    }
    matrices = artifact["matrices"]
    assert set(matrices) == {"F", "G", "H", "Q", "R"}
    np.testing.assert_array_equal(np.asarray(matrices["F"]).shape, (n, n))
    np.testing.assert_array_equal(np.asarray(matrices["G"]).shape, (n, k))
    np.testing.assert_array_equal(np.asarray(matrices["H"]).shape, (m, n))
    np.testing.assert_array_equal(np.asarray(matrices["Q"]).shape, (n, n))
    np.testing.assert_array_equal(np.asarray(matrices["R"]).shape, (m, m))
    for name in ("F", "G", "H", "Q", "R"):
        _assert_finite(matrices[name])

    # Covariances: symmetric atol 1e-12; Q positive semidefinite, R positive
    # definite: gnn_gaussian_contract.py:79-92.
    for name, definite in (("Q", "semi"), ("R", "definite")):
        array = np.asarray(matrices[name], dtype=float)
        np.testing.assert_allclose(array, array.T, rtol=0, atol=1e-12)
        spectrum = np.linalg.eigvalsh(array)
        if definite == "semi":
            assert spectrum.min() >= -1e-12
        else:
            assert spectrum.min() > 0

    # initial_belief: exactly {mean, covariance}; mean=(n,), covariance=(n,n)
    # positive definite: gnn_gaussian_contract.py:63-76, 89-92.
    belief = artifact["initial_belief"]
    assert set(belief) == {"mean", "covariance"}
    np.testing.assert_array_equal(np.asarray(belief["mean"]).shape, (n,))
    covariance = np.asarray(belief["covariance"], dtype=float)
    np.testing.assert_array_equal(covariance.shape, (n, n))
    assert np.linalg.eigvalsh(covariance).min() > 0

    # Units: exactly one list per dimension, each of the right size, nonempty
    # strings: gnn_gaussian_contract.py:93-101.
    assert artifact["units"] == GAUSSIAN_UNITS

    # Time: exactly {domain, step_seconds}; domain is literally "discrete":
    # gnn_gaussian_contract.py:102-109.
    assert artifact["time"] == {"domain": "discrete", "step_seconds": 2}

    # Provenance: gnn_gaussian_contract.py:110-120.
    assert set(artifact["provenance"]) == {"producer", "source_sha256"}
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def test_factored_artifact_matches_consumer_factored_1() -> None:
    """gnn-geo-infer/factored/1 emission satisfies gnn_factored_contract."""
    text = FACTORED_SOURCE.read_text()
    artifact = build_geo_infer_factored_artifact(text, step_seconds=60)

    # Exact key set: gnn_factored_contract.py:65-69.
    assert set(artifact) == {
        "schema_version",
        "model_type",
        "model_name",
        "state_factors",
        "control_factors",
        "modalities",
        "transitions",
        "initial_joint",
        "policies",
        "policy_prior",
        "time",
        "provenance",
    }
    # Byte-exact literals: gnn_factored_contract.py:70-74.
    assert artifact["schema_version"] == "gnn-geo-infer/factored/1"
    assert artifact["model_type"] == "categorical_factored"

    # Provenance: exactly {producer, source_kind, source_sha256} and
    # source_kind is literally 'explicit_factored_json': 88-96.
    assert set(artifact["provenance"]) == {
        "producer",
        "source_kind",
        "source_sha256",
    }
    assert artifact["provenance"]["source_kind"] == "explicit_factored_json"
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
    )

    # Factors: 1..8 each; joint state budget <= 256: 103-115.
    state_sizes = [len(f["states"]) for f in artifact["state_factors"]]
    control_sizes = [len(f["actions"]) for f in artifact["control_factors"]]
    assert 1 <= len(state_sizes) <= 8
    assert 1 <= len(control_sizes) <= 8
    joint = int(np.prod(state_sizes))
    assert joint <= 256
    assert len(artifact["initial_joint"]) == joint

    # Policies: 1..256, shared horizon 1..8, every action list covers EVERY
    # control factor with in-range indices: 161-173.
    policies = artifact["policies"]
    assert 1 <= len(policies) <= 256
    horizons = {len(policy) for policy in policies}
    assert len(horizons) == 1
    horizon = horizons.pop()
    assert 1 <= horizon <= 8
    for policy in policies:
        for action in policy:
            assert len(action) == len(control_sizes)
            assert all(0 <= index < size for index, size in zip(action, control_sizes))
    assert len(artifact["policy_prior"]) == len(policies)

    # Stochastic vectors along axis zero, atol 1e-8: 191-199. Preferences are
    # per-outcome vectors and are NOT required to be stochastic (128-133).
    _assert_finite(artifact["initial_joint"])
    _assert_axis0_stochastic(artifact["initial_joint"])
    _assert_finite(artifact["policy_prior"])
    _assert_axis0_stochastic(artifact["policy_prior"])
    for modality in artifact["modalities"]:
        assert set(modality) == {
            "id",
            "outcomes",
            "dependencies",
            "likelihood",
            "preferences",
        }  # gnn_factored_contract.py:125
        _assert_finite(modality["likelihood"])
        _assert_axis0_stochastic(modality["likelihood"])
        _assert_finite(modality["preferences"])
    for transition in artifact["transitions"]:
        assert set(transition) == {"dependencies", "control_factor", "probabilities"}
        _assert_finite(transition["probabilities"])
        _assert_axis0_stochastic(transition["probabilities"])

    # Time: exactly {step_seconds}: 77-86.
    assert artifact["time"] == {"step_seconds": 60}


def test_producer_version_constants_match_the_pinned_consumer() -> None:
    """The producer's advertised schema strings are byte-exact (consumer 108-112
    / 47-51 / 70-74 reject anything else)."""
    from gnn.export import geo_infer, geo_infer_factored, geo_infer_gaussian

    assert geo_infer.CONTRACT_VERSION == "gnn-geo-infer/1"
    assert geo_infer_gaussian.CONTRACT_VERSION == "gnn-geo-infer/2"
    assert geo_infer_factored.CONTRACT_VERSION == "gnn-geo-infer/factored/1"
