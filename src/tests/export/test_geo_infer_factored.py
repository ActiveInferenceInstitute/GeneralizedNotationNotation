"""Explicit factored JSON export rejects missing axes and unsupported repairs."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from export.geo_infer_factored import build_geo_infer_factored_artifact


def _source() -> str:
    return Path(__file__).with_name("factored_example.json").read_text()


def test_factored_export_preserves_dependencies_policy_order_and_source() -> None:
    source = _source()
    artifact = build_geo_infer_factored_artifact(source, step_seconds=60)
    assert artifact["schema_version"] == "gnn-geo-infer/factored/1"
    assert artifact["provenance"]["source_kind"] == "explicit_factored_json"
    assert (
        artifact["provenance"]["source_sha256"]
        == hashlib.sha256(source.encode()).hexdigest()
    )
    for name, value in json.loads(source).items():
        assert artifact[name] == value
    assert artifact["modalities"][1]["dependencies"] == [1, 0]
    assert len(artifact["policies"]) == len(artifact["policy_prior"]) == 4


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda d: d["modalities"][1].pop("dependencies"), "exactly"),
        (lambda d: d["modalities"][1].update(dependencies=[0, 1]), "shape"),
        (lambda d: d.update(policy_prior=[0.5, 0.5]), "shape"),
        (lambda d: d["policies"][0][0].__setitem__(1, 1), "integer"),
        (lambda d: d["initial_joint"].__setitem__(0, -0.1), "stochastic"),
        (lambda d: d["transitions"][0].update(probabilities=[[[0.5]]]), "shape"),
    ],
)
def test_export_rejects_ambiguous_axes_or_missing_policy_metadata(
    mutation: Callable[[dict[str, Any]], object], match: str
) -> None:
    source = json.loads(_source())
    mutation(source)
    with pytest.raises(ValueError, match=match):
        build_geo_infer_factored_artifact(json.dumps(source), step_seconds=60)


def test_export_rejects_duplicate_json_and_markdown_input() -> None:
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        build_geo_infer_factored_artifact(
            '{"model_name":"a","model_name":"b"}', step_seconds=1
        )
    with pytest.raises(ValueError):
        build_geo_infer_factored_artifact(
            "## ModelName\nMarkdown model", step_seconds=1
        )


@pytest.mark.parametrize("seconds", [True, 0, -1, float("inf")])
def test_export_requires_explicit_positive_seconds(seconds: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_geo_infer_factored_artifact(_source(), step_seconds=seconds)
