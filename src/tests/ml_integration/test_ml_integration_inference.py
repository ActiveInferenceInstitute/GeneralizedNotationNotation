#!/usr/bin/env python3
"""Inference error-path and training round-trip tests for ml_integration.

Non-gated tests exercise :class:`ml_integration.InferenceError` on broken or
missing classifier artifacts (sklearn-free). Gated tests
(``pytest.importorskip("sklearn")``) run the full training round trip via
``process_ml_integration`` and score the saved artifacts.
"""

import pickle
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

from ml_integration.inference import InferenceError, load_classifier
from tests.ml_integration.test_classifier_deserialization_security import _Reducer

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_gnn(target_dir: Path, name: str, content: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / name).write_text(textwrap.dedent(content))


POMDP_TEMPLATE = """\
    ## StateSpaceBlock

    B[{b1},{b2},{b3}]
    π[{pi}]
    """

HMM_TEMPLATE = """\
    ## StateSpaceBlock

    B[{b1},{b2}]
    """

UNKNOWN_TEMPLATE = """\
    ## StateSpaceBlock

    s[{dim}]
    """


def _load_results(output_dir: Path) -> dict[str, Any]:
    import json

    with open(output_dir / "ml_integration_results.json") as f:
        return cast("dict[str, Any]", json.load(f))


def test_predict_with_model_missing_path_raises(tmp_path: Path) -> None:
    from ml_integration import InferenceError, predict_with_model

    with pytest.raises(InferenceError):
        predict_with_model(tmp_path / "does_not_exist.pkl", {})


def test_predict_with_model_garbage_artifact_raises(tmp_path: Path) -> None:
    from ml_integration import InferenceError, predict_with_model

    garbage = tmp_path / "garbage.pkl"
    garbage.write_bytes(b"not a pickle")
    with pytest.raises(InferenceError):
        predict_with_model(garbage, {})


def test_predict_batch_garbage_artifact_raises(tmp_path: Path) -> None:
    from ml_integration import InferenceError, predict_batch

    garbage = tmp_path / "garbage.pkl"
    garbage.write_bytes(b"not a pickle")
    with pytest.raises(InferenceError):
        predict_batch(garbage, [{}, {}])


def test_load_classifier_missing_dependency_raises_inference_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpickling an artifact whose classes live in a missing module
    (e.g. a sklearn-pickled model without sklearn installed) must raise
    InferenceError, never a raw ModuleNotFoundError/ImportError."""
    import ml_integration.inference as inference_mod
    from ml_integration import InferenceError

    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"csklearn.tree._classes\nDecisionTreeClassifier\n.")

    def _raise_module_not_found(*args: object, **kwargs: object) -> object:
        raise ModuleNotFoundError("No module named 'sklearn'")

    monkeypatch.setattr(inference_mod, "import_module", _raise_module_not_found)
    with pytest.raises(InferenceError):
        inference_mod.load_classifier(artifact)


def _train_family_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Write 6 GNN files (3 pomdp + 3 hmm), run processing, load results."""
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    out = tmp_path / "out"
    for i, (b1, b2, b3, pi) in enumerate([(2, 3, 4, 2), (3, 4, 5, 3), (2, 4, 6, 4)]):
        _write_gnn(
            target,
            f"pomdp_{i}.md",
            POMDP_TEMPLATE.format(b1=b1, b2=b2, b3=b3, pi=pi),
        )
    for i, (b1, b2) in enumerate([(2, 3), (3, 3), (2, 5)]):
        _write_gnn(target, f"hmm_{i}.md", HMM_TEMPLATE.format(b1=b1, b2=b2))

    assert process_ml_integration(target, out) is True
    return target, out, _load_results(out)


def test_training_roundtrip_family_classification(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from ml_integration import (
        NUMERIC_FEATURE_NAMES,
        extract_gnn_features,
        predict_batch,
        predict_with_model,
    )

    target, out, results = _train_family_fixture(tmp_path)

    assert results["classification_status"] == "trained"
    assert results["classification_task"] == "family_classification"

    models = results["models_trained"]
    assert {m["type"] for m in models} == {"decision_tree", "random_forest"}
    for model in models:
        assert model["feature_names"] == list(NUMERIC_FEATURE_NAMES)
        assert model["n_samples"] == 6
        assert Path(model["artifact_path"]).exists()

    label_names = results["label_names"]

    pomdp_file = target / "pomdp_0.md"
    feats = extract_gnn_features(pomdp_file)

    all_feats = [
        extract_gnn_features(target / name)
        for name in sorted(p.name for p in target.glob("*.md"))
    ]
    for kind in ("decision_tree", "random_forest"):
        artifact = out / f"gnn_{kind}.pkl"
        single = predict_with_model(artifact, feats, label_names)
        assert single in label_names
        batch = predict_batch(artifact, all_feats, label_names)
        assert len(batch) == 6
        assert all(label in label_names for label in batch)


def test_complexity_fallback_classification(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    out = tmp_path / "out"
    for i, dim in enumerate([5, 100, 1000]):
        _write_gnn(target, f"unknown_{i}.md", UNKNOWN_TEMPLATE.format(dim=dim))

    assert process_ml_integration(target, out) is True
    results = _load_results(out)

    assert results["classification_status"] == "trained"
    assert results["classification_task"] == "complexity_classification"
    assert results["label_names"] == ["small", "medium", "large"]
    assert (out / "gnn_decision_tree.pkl").exists()
    assert (out / "gnn_random_forest.pkl").exists()


def test_insufficient_label_variation_skips_training(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from ml_integration import process_ml_integration

    target = tmp_path / "target"
    out = tmp_path / "out"
    # Two unknown-family files with the same total_parameters bucket (small).
    _write_gnn(target, "same_a.md", UNKNOWN_TEMPLATE.format(dim=5))
    _write_gnn(target, "same_b.md", UNKNOWN_TEMPLATE.format(dim=7))

    assert process_ml_integration(target, out) is True
    results = _load_results(out)

    assert results["classification_status"] == "insufficient_label_variation"
    assert results["training_note"]
    assert not list(out.glob("gnn_*.pkl"))


def test_label_decode_out_of_range_raises(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from ml_integration import InferenceError, predict_with_model

    _, out, _ = _train_family_fixture(tmp_path)
    with pytest.raises(InferenceError):
        predict_with_model(out / "gnn_decision_tree.pkl", {}, label_names=[])


@pytest.mark.parametrize("kind", ["decision_tree", "random_forest"])
@pytest.mark.parametrize("protocol", [4, 5])
def test_maintained_classifier_roundtrip(
    tmp_path: Path, kind: str, protocol: int
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    x = np.arange(84, dtype=np.float64).reshape(6, 14)
    y = np.array([0, 0, 0, 1, 1, 1])
    original = (
        DecisionTreeClassifier(max_depth=4, random_state=42)
        if kind == "decision_tree"
        else RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    ).fit(x, y)
    artifact = tmp_path / "classifier.pkl"
    artifact.write_bytes(pickle.dumps(original, protocol=protocol))
    restored = load_classifier(artifact)
    assert type(restored) is type(original)
    np.testing.assert_array_equal(restored.predict(x), original.predict(x))
    np.testing.assert_array_equal(restored.predict_proba(x), original.predict_proba(x))


@pytest.mark.parametrize("kind", ["decision_tree", "random_forest"])
def test_estimator_containing_reducer_never_executes(tmp_path: Path, kind: str) -> None:
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    model = (
        DecisionTreeClassifier(max_depth=4, random_state=42)
        if kind == "decision_tree"
        else RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    ).fit([[0], [1]], [0, 1])
    model.unexpected_payload = _Reducer()
    artifact = tmp_path / "nested.pkl"
    artifact.write_bytes(pickle.dumps(model))
    with pytest.raises(InferenceError, match="refused global"):
        load_classifier(artifact)


def test_malformed_estimator_state_is_wrapped(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    # A permitted class with an invalid (integer) BUILD state must fail through
    # InferenceError even when the underlying error is not UnpicklingError.
    artifact = tmp_path / "invalid-state.pkl"
    artifact.write_bytes(
        b"csklearn.tree._classes\nDecisionTreeClassifier\n)\x81K\x01b."
    )
    with pytest.raises(InferenceError):
        load_classifier(artifact)
