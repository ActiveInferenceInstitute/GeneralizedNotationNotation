"""Canonical sample GNN content shared by conftest fixtures and module tests.

Single source for the minimal POMDP-flavored GNN markdown blobs used across
the suite. Module test directories should import these instead of re-declaring
near-identical inline strings; the conftest fixtures (``sample_gnn_files``,
``test_data_dir``, ``sample_gnn_file``) are built on top of them.
"""

from __future__ import annotations

from pathlib import Path

SAMPLE_GNN_CONTENT = """
# Test GNN Model

## ModelName
test_model

## StateSpaceBlock
s[3,1,type=int]
o[3,1,type=int]

## Connections
s -> o

## InitialParameterization
A = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
B = [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]]
C = [0.0, 0.0, 1.0]
D = [0.34, 0.33, 0.33]
"""


def write_sample_gnn_markdown(target: Path) -> None:
    """Write a minimal GNN markdown with ontology annotations to ``target``.

    Creates parent directories as needed. This is the shared content behind
    the ``test_data_dir`` and ``sample_gnn_file`` fixtures.
    """
    content = (
        "# Active Inference Model\n\n"
        "## ActInfOntologyAnnotation\n"
        "s = HiddenState\n"
        "s_prime = NextHiddenState\n"
        "o = Observation\n"
        "π = PolicyVector\n"
        "u = Action\n"
        "t = Time\n"
        "A = LikelihoodMatrix\n"
        "B = TransitionMatrix\n"
        "C = LogPreferenceVector\n"
        "D = PriorOverHiddenStates\n"
        "E = Habit\n"
        "F = VariationalFreeEnergy\n"
        "G = ExpectedFreeEnergy\n\n"
        "## Connections\n"
        "s -> o\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


__all__ = ["SAMPLE_GNN_CONTENT", "write_sample_gnn_markdown"]
