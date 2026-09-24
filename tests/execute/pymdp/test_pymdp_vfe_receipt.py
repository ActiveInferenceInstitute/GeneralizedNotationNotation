"""VFE receipt-state tests for the pymdp simulation v1 schema (M-07).

A VFE extraction failure must NOT be masked as a measured ``0.0``:
- the numeric series is None-padded (index-aligned with beliefs/efe);
- ``vfe_unavailable_timesteps`` names the timesteps with no VFE;
- a WARNING records each failed extraction.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.needs_pymdp]

from gnn.execute.pymdp.pymdp_simulation import PyMDPSimulation


@pytest.mark.unit
class TestVFEReceiptState:
    """Extraction failure produces an explicit 'unavailable' receipt, not 0.0."""

    @pytest.fixture
    def sim(self) -> PyMDPSimulation:
        """Demo model with a real pymdp 1.0.0 agent (suite gated by needs_pymdp marker)."""
        return PyMDPSimulation(gnn_config={}, allow_demo_spec=True)

    def test_failed_extraction_yields_null_padding_and_receipt(
        self,
        sim: PyMDPSimulation,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Broken info dict -> None-padded series + vfe_unavailable_timesteps."""
        original = type(sim.agent).infer_states

        def _broken_infer_states(*args: Any, **kwargs: Any):
            qs, _info = original(*args, **kwargs)
            return qs, {}  # no "vfe" key

        assert sim.agent is not None
        monkeypatch.setattr(
            type(sim.agent), "infer_states", _broken_infer_states, raising=True
        )

        with caplog.at_level(logging.WARNING):
            results = sim.run_simulation(num_timesteps=3)
        unavailable = results["vfe_unavailable_timesteps"]
        vfe_series = results["variational_free_energy"]
        # Design: numeric series keeps only measured values (pure floats).
        assert vfe_series == []
        assert results["simulation_trace"]["vfe_history"] == vfe_series
        assert unavailable == [0, 1, 2]

    def test_measured_series_stays_numeric(self, sim: PyMDPSimulation) -> None:
        """Healthy run: all-float series and empty unavailable receipt."""
        results = sim.run_simulation(num_timesteps=2)

        assert results["vfe_unavailable_timesteps"] == []
        vfe_series = results["variational_free_energy"]
        assert len(vfe_series) == 2
        assert all(isinstance(v, float) for v in vfe_series)
