"""Inference utilities for classifiers trained by the ml_integration module.

Step 14 pickles trained scikit-learn classifiers as ``gnn_<model>.pkl``
artifacts next to ``ml_integration_results.json``. This module closes the
loop: it loads those artifacts and scores feature mappings using the exact
canonical vector order defined by
:data:`ml_integration.processor.NUMERIC_FEATURE_NAMES` — the same order used
at training time.

scikit-learn is only required when a classifier is actually loaded; every
function raises :class:`InferenceError` (never ImportError) on failure.
"""

from __future__ import annotations

import io
import logging

# Required for the exact-global restricted unpickler below.
import pickle  # nosec B403

# Opcode inspection does not reconstruct or execute pickle objects.
import pickletools  # nosec B403
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from .processor import feature_vector

logger = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when a trained classifier artifact cannot be used for prediction."""


# Exact globals emitted by Step 14's two maintained estimators (protocols 4/5).
# NumPy 1 and 2 use different names for the same reconstruction primitives.
_CLASSIFIER_GLOBALS = frozenset(
    {
        ("sklearn.tree._classes", "DecisionTreeClassifier"),
        ("sklearn.ensemble._forest", "RandomForestClassifier"),
        ("sklearn.tree._tree", "Tree"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
    }
)


class _ClassifierUnpickler(pickle.Unpickler):
    """Resolve only the globals needed by maintained Step 14 classifiers."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _CLASSIFIER_GLOBALS:
            raise pickle.UnpicklingError(
                f"Classifier unpickler refused global {module}.{name}"
            )
        return getattr(import_module(module), name)


def load_classifier(model_path: Path) -> Any:
    """Load only Step 14 DecisionTreeClassifier/RandomForestClassifier artifacts.

    Unknown globals, pickle extension opcodes, trailing data, and unexpected
    root types are rejected. Optional dependencies remain deferred until load.
    This restricts reconstruction capabilities; it does not authenticate an
    artifact, make sklearn versions compatible, or bound native memory use.

    Raises:
        InferenceError: If reading, reconstruction, or root validation fails.
    """
    try:
        payload = model_path.read_bytes()
        # Use one byte snapshot for inspection and reconstruction. EXT opcodes
        # can bypass find_class when copyreg's global extension cache is warm.
        stop_position: int | None = None
        for opcode, _, position in pickletools.genops(payload):
            if opcode.name in {"EXT1", "EXT2", "EXT4"}:
                raise pickle.UnpicklingError(
                    "Pickle extension globals are not permitted"
                )
            stop_position = position
        if stop_position is None or stop_position + 1 != len(payload):
            raise pickle.UnpicklingError("Trailing data after classifier pickle")
        model = _ClassifierUnpickler(io.BytesIO(payload)).load()
        model_type = type(model)
        root_global = (model_type.__module__, model_type.__name__)
        if root_global not in {
            ("sklearn.tree._classes", "DecisionTreeClassifier"),
            ("sklearn.ensemble._forest", "RandomForestClassifier"),
        } or model_type is not getattr(import_module(root_global[0]), root_global[1]):
            raise pickle.UnpicklingError("Unexpected classifier root type")
        return model
    except Exception as e:
        raise InferenceError(f"Could not load classifier from {model_path}: {e}") from e


def _decode_label(
    prediction: Any, label_names: Sequence[str] | None
) -> str | int | float:
    """Decode a raw model prediction into a human-readable label when possible."""
    if hasattr(prediction, "item"):
        prediction = prediction.item()  # numpy scalar -> python scalar
    if label_names is None:
        if isinstance(prediction, (int, float, str)):
            return prediction
        return str(prediction)
    index = int(prediction)
    if not 0 <= index < len(label_names):
        raise InferenceError(
            f"Predicted class index {index} is outside "
            f"label_names (size {len(label_names)})"
        )
    return label_names[index]


def predict_with_model(
    model_path: Path,
    features: Mapping[str, Any],
    label_names: Sequence[str] | None = None,
) -> str | int | float:
    """Predict a class label for one feature mapping with a saved classifier.

    Args:
        model_path: Path to a ``gnn_<model>.pkl`` artifact written by
            ``process_ml_integration``.
        features: Feature mapping as produced by ``extract_gnn_features``.
            Projected onto :data:`ml_integration.processor.NUMERIC_FEATURE_NAMES`
            before scoring.
        label_names: Ordered class names as recorded in
            ``ml_integration_results.json`` (``label_names``). When given, the
            raw class index is decoded to the corresponding name.

    Returns:
        The decoded label, or the raw prediction when ``label_names`` is omitted.

    Raises:
        InferenceError: If loading or prediction fails, or the predicted
            index is out of range for ``label_names``.
    """
    clf = load_classifier(model_path)
    try:
        vector = feature_vector(features)
        prediction = clf.predict([vector])[0]
    except InferenceError:
        raise
    except Exception as e:
        raise InferenceError(f"Prediction failed with model {model_path}: {e}") from e
    return _decode_label(prediction, label_names)


def predict_batch(
    model_path: Path,
    feature_mappings: Sequence[Mapping[str, Any]],
    label_names: Sequence[str] | None = None,
) -> list[str | int | float]:
    """Predict labels for a sequence of feature mappings.

    Loads the classifier once and scores every mapping; see
    :func:`predict_with_model` for argument semantics.

    Raises:
        InferenceError: If loading or prediction fails for any mapping.
    """
    clf = load_classifier(model_path)
    try:
        vectors = [feature_vector(features) for features in feature_mappings]
        predictions = clf.predict(vectors)
    except InferenceError:
        raise
    except Exception as e:
        raise InferenceError(
            f"Batch prediction failed with model {model_path}: {e}"
        ) from e
    return [_decode_label(p, label_names) for p in predictions]
