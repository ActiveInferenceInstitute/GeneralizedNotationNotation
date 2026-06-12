from __future__ import annotations

import pytest

from scripts import check_capability_contracts
from scripts.check_capability_contracts import run_audit


def test_capability_contracts_are_current() -> None:
    assert run_audit() == []


def test_capability_contracts_fail_strict_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_capability_contracts, "run_audit", lambda: ["synthetic failure"]
    )

    assert check_capability_contracts.main([]) == 1
    assert check_capability_contracts.main(["--warn-only"]) == 0
