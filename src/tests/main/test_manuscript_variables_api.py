"""Tests for the manuscript-variable round-trip API.

Pins ``load_variables`` (validated inverse of ``save_variables``) and
``token_checksum`` (stable canonical-JSON fingerprint) against the real
producer output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manuscript_variables import (  # noqa: E402
    generate_variables,
    load_variables,
    save_variables,
    token_checksum,
)

pytestmark = pytest.mark.unit
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def variables() -> dict[str, str]:
    return generate_variables(_PROJECT_ROOT)


def test_load_variables_roundtrip(tmp_path: Path, variables: dict[str, str]) -> None:
    out = save_variables(variables, tmp_path / "vars.json")
    assert load_variables(out) == variables


def test_load_variables_rejects_non_string_values(tmp_path: Path) -> None:
    bad = tmp_path / "nonstring.json"
    bad.write_text(json.dumps({"GNN_VERSION": 3}), encoding="utf-8")
    with pytest.raises(ValueError, match="flat"):
        load_variables(bad)


def test_load_variables_rejects_non_object(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text(json.dumps(["GNN_VERSION"]), encoding="utf-8")
    with pytest.raises(ValueError, match="flat"):
        load_variables(bad)


def test_token_checksum_stable_across_roundtrip(
    tmp_path: Path, variables: dict[str, str]
) -> None:
    out = save_variables(variables, tmp_path / "vars.json")
    assert token_checksum(load_variables(out)) == token_checksum(variables)


def test_token_checksum_deterministic_and_discriminating(
    variables: dict[str, str],
) -> None:
    checksum = token_checksum(variables)
    assert checksum == token_checksum(variables)
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum)

    perturbed = dict(variables)
    first_key = next(iter(perturbed))
    perturbed[first_key] = perturbed[first_key] + "x"
    assert token_checksum(perturbed) != checksum


def test_checksum_matches_save_variables_canonical_bytes(
    tmp_path: Path, variables: dict[str, str]
) -> None:
    """token_checksum fingerprints exactly the save_variables byte payload."""
    import hashlib

    out = save_variables(variables, tmp_path / "vars.json")
    body = out.read_text(encoding="utf-8").removesuffix("\n")
    assert token_checksum(variables) == hashlib.sha256(body.encode("utf-8")).hexdigest()
