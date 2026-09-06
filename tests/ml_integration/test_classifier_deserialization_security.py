"""Restricted Step 14 artifact loading, including reducers that must never run."""

from __future__ import annotations

import copyreg
import pickle
from pathlib import Path
from typing import Any

import pytest

from gnn.ml_integration.inference import InferenceError, load_classifier


def _sentinel() -> None:
    pytest.fail("Forbidden pickle reducer executed")


class _Reducer:
    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return _sentinel, ()


@pytest.mark.parametrize("value", [None, {}, [], "not a classifier"])
def test_unexpected_root_is_rejected(tmp_path: Path, value: object) -> None:
    artifact = tmp_path / "wrong-root.pkl"
    artifact.write_bytes(pickle.dumps(value))
    with pytest.raises(InferenceError, match="classifier root"):
        load_classifier(artifact)


@pytest.mark.parametrize("payload", [b"", b"\x80\x04", b"N", b"not a pickle"])
def test_malformed_artifact_is_wrapped(tmp_path: Path, payload: bytes) -> None:
    artifact = tmp_path / "broken.pkl"
    artifact.write_bytes(payload)
    with pytest.raises(InferenceError):
        load_classifier(artifact)


def test_forbidden_reducer_never_executes(tmp_path: Path) -> None:
    artifact = tmp_path / "reducer.pkl"
    artifact.write_bytes(pickle.dumps(_Reducer()))
    with pytest.raises(InferenceError, match="refused global"):
        load_classifier(artifact)


@pytest.mark.parametrize(
    "opcode, code", [(b"\x82", 231), (b"\x83", 60001), (b"\x84", 1000001)]
)
def test_cached_extension_never_executes(
    tmp_path: Path, opcode: bytes, code: int
) -> None:
    # An already cached extension skips Unpickler.find_class. Never load it
    # unrestricted: register/cache the inert callable directly, then clean up.
    artifact = tmp_path / "extension.pkl"
    key = (__name__, "_sentinel")
    copyreg.add_extension(*key, code)
    cache: dict[int, object] = vars(copyreg)["_extension_cache"]
    cache[code] = _sentinel
    try:
        width = {b"\x82": 1, b"\x83": 2, b"\x84": 4}[opcode]
        artifact.write_bytes(
            b"\x80\x04" + opcode + code.to_bytes(width, "little") + b")R."
        )
        with pytest.raises(InferenceError, match="extension"):
            load_classifier(artifact)
    finally:
        copyreg.remove_extension(*key, code)
        cache.pop(code, None)


def test_other_sklearn_global_is_rejected_before_import(tmp_path: Path) -> None:
    artifact = tmp_path / "other-sklearn.pkl"
    artifact.write_bytes(b"csklearn.utils\nnot_a_permitted_global\n.")
    with pytest.raises(InferenceError, match="refused global"):
        load_classifier(artifact)


def test_numpy_root_is_rejected(tmp_path: Path) -> None:
    import numpy as np

    artifact = tmp_path / "array.pkl"
    artifact.write_bytes(pickle.dumps(np.arange(3)))
    with pytest.raises(InferenceError, match="classifier root"):
        load_classifier(artifact)


def test_forbidden_module_is_not_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gnn.ml_integration.inference as inference

    def forbidden_import(name: str) -> Any:
        pytest.fail(f"Unapproved import attempted: {name}")

    monkeypatch.setattr(inference, "import_module", forbidden_import)
    artifact = tmp_path / "global.pkl"
    artifact.write_bytes(b"cnot_an_approved_package\ncallable\n.")
    with pytest.raises(InferenceError, match="refused global"):
        load_classifier(artifact)


def test_trailing_pickle_data_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "trailing.pkl"
    artifact.write_bytes(pickle.dumps(None) + pickle.dumps(_Reducer()))
    with pytest.raises(InferenceError, match="Trailing data"):
        load_classifier(artifact)
