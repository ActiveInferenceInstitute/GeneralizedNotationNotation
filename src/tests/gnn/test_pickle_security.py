"""Security-focused tests for restricted unpickling of untrusted GNN .pickle input.

The GNN binary parser must reconstruct only plain data containers and the
repo's own model classes. Any other global (which could execute arbitrary code
via __reduce__) must be refused and routed to a graceful parse-failure result.
"""

import base64
import pickle
from pathlib import Path

import pytest

from gnn.parsers.binary_parser import PickleGNNParser, safe_pickle_loads


class _NotAllowedOnAllowlist:
    """Module-level class that pickles fine but is not on the safe allowlist."""


def _enhanced_dict() -> dict:
    """A plain-dict GNN model in the repo's enhanced serialization shape."""
    return {
        "model_name": "TestPickle",
        "version": "1.0",
        "variables": [
            {
                "name": "x",
                "var_type": "hidden_state",
                "data_type": "categorical",
                "dimensions": [2, 2],
            }
        ],
        "connections": [],
        "parameters": [],
    }


@pytest.mark.unit
class TestRestrictedUnpickle:
    def test_legit_enhanced_dict_parses(self, tmp_path: Path) -> None:
        """A plain-dict pickle (the repo's own serialization) parses cleanly."""
        file_path = tmp_path / "model.pickle"
        file_path.write_bytes(pickle.dumps(_enhanced_dict()))
        result = PickleGNNParser().parse_file(str(file_path))
        assert result.model.model_name == "TestPickle"
        assert not result.errors

    def test_legit_simple_dict_parses_via_base64(self) -> None:
        """Simple dicts also parse via the base64-encoded text path."""
        payload = base64.b64encode(pickle.dumps({"model_name": "M", "a": 1})).decode()
        result = PickleGNNParser().parse_string(payload)
        assert result.model.model_name == "M"
        assert not result.errors

    def test_gnn_internal_representation_round_trip(self, tmp_path: Path) -> None:
        """The repo's own model class is on the allowlist and round-trips."""
        from gnn.parsers.common import GNNInternalRepresentation

        file_path = tmp_path / "model.pickle"
        file_path.write_bytes(pickle.dumps(GNNInternalRepresentation(model_name="RT")))
        result = PickleGNNParser().parse_file(str(file_path))
        assert result.model.model_name == "RT"

    def test_malicious_pickle_does_not_execute(self, tmp_path: Path) -> None:
        """A __reduce__ that would run code is refused; no side effect occurs."""
        sentinel = tmp_path / "pwned"

        class _Evil:
            def __reduce__(self) -> "tuple[object, tuple[str]]":
                import os

                return (os.system, (f"touch {sentinel}",))

        file_path = tmp_path / "evil.pickle"
        file_path.write_bytes(pickle.dumps(_Evil()))
        result = PickleGNNParser().parse_file(str(file_path))
        assert not sentinel.exists(), "malicious pickle executed code"
        assert result.errors, "malicious pickle should have produced a parse error"

    def test_restricted_loads_rejects_arbitrary_type(self) -> None:
        """Reconstructing a non-allowlisted class raises UnpicklingError."""
        with pytest.raises(pickle.UnpicklingError):
            safe_pickle_loads(pickle.dumps(_NotAllowedOnAllowlist()))
