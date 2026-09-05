"""Export validation must parse records without executing pickle or XML payloads."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from export.processor import validate_export_outputs


def _validate(tmp_path: Path, suffix: str, payload: bytes) -> dict[str, Any]:
    artifact = tmp_path / f"model{suffix}"
    artifact.write_bytes(payload)
    (tmp_path / "export_results.json").write_text(
        json.dumps(
            {
                "files_exported": [
                    {
                        "file_name": "model.md",
                        "exports": {
                            "artifact": {"success": True, "export_file": str(artifact)}
                        },
                    }
                ]
            }
        )
    )
    return validate_export_outputs(tmp_path)


def _sentinel() -> None:
    pytest.fail("Forbidden exported pickle reducer executed")


class _Reducer:
    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return _sentinel, ()


@pytest.mark.parametrize("suffix", [".xml", ".graphml", ".gexf"])
def test_xml_entity_is_invalid(tmp_path: Path, suffix: str) -> None:
    payload = b'<!DOCTYPE gnn [<!ENTITY harmless "expanded">]><gnn>&harmless;</gnn>'
    result = _validate(tmp_path, suffix, payload)
    assert result["success"] is False
    assert len(result["invalid"]) == 1
    assert "Forbidden" in result["invalid"][0]["error"]


@pytest.mark.parametrize("suffix", [".xml", ".graphml", ".gexf"])
def test_external_entity_is_invalid(tmp_path: Path, suffix: str) -> None:
    secret = tmp_path / "local.txt"
    secret.write_text("MUST_NOT_BE_INCLUDED")
    payload = (
        f'<!DOCTYPE gnn [<!ENTITY local SYSTEM "{secret.as_uri()}">]><gnn>&local;</gnn>'
    )
    result = _validate(tmp_path, suffix, payload.encode())
    assert result["success"] is False
    assert "Forbidden" in result["invalid"][0]["error"]
    assert "MUST_NOT_BE_INCLUDED" not in str(result)


@pytest.mark.parametrize("suffix", [".xml", ".graphml", ".gexf"])
def test_plain_xml_is_valid(tmp_path: Path, suffix: str) -> None:
    assert _validate(tmp_path, suffix, b"<gnn><name>model</name></gnn>")["success"]


@pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
def test_exported_record_roundtrip(tmp_path: Path, suffix: str) -> None:
    record = {"name": "gnn", "variables": [{"name": "s", "dimensions": [2]}]}
    assert _validate(tmp_path, suffix, pickle.dumps(record))["success"]


@pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
def test_export_reducer_never_executes(tmp_path: Path, suffix: str) -> None:
    result = _validate(tmp_path, suffix, pickle.dumps(_Reducer()))
    assert result["success"] is False
    assert "refused" in result["invalid"][0]["error"]


@pytest.mark.parametrize(
    "suffix,payload", [(".pkl", b""), (".pickle", b"broken"), (".xml", b"<gnn>")]
)
def test_malformed_export_is_invalid(
    tmp_path: Path, suffix: str, payload: bytes
) -> None:
    result = _validate(tmp_path, suffix, payload)
    assert result["success"] is False
    assert len(result["invalid"]) == 1


@pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
def test_cached_export_extension_never_executes(tmp_path: Path, suffix: str) -> None:
    import copyreg

    key = (__name__, "_sentinel")
    code = 232
    copyreg.add_extension(*key, code)
    cache: dict[int, object] = vars(copyreg)["_extension_cache"]
    cache[code] = _sentinel
    try:
        result = _validate(tmp_path, suffix, b"\x80\x04\x82\xe8)R.")
        assert result["success"] is False
        assert "extension" in result["invalid"][0]["error"]
    finally:
        copyreg.remove_extension(*key, code)
        cache.pop(code, None)


def test_xml_missing_dependency_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    original = builtins.__import__

    def without_defusedxml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "defusedxml":
            raise ModuleNotFoundError("defusedxml intentionally unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_defusedxml)
    result = _validate(tmp_path, ".xml", b"<gnn/>")
    assert result["success"] is False
    assert "defusedxml intentionally unavailable" in result["invalid"][0]["error"]


@pytest.mark.parametrize("suffix", [".pkl", ".pickle"])
def test_trailing_export_data_is_invalid(tmp_path: Path, suffix: str) -> None:
    result = _validate(tmp_path, suffix, pickle.dumps({"name": "gnn"}) + b"trailing")
    assert result["success"] is False
    assert "Trailing data" in result["invalid"][0]["error"]
