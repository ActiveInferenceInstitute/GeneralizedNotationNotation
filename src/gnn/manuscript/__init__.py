"""Manuscript variable production for the GNN repository.

Deterministic ``{{...}}`` token production for the manuscript: reads the
repository snapshot at the current commit, emits
``output/data/manuscript_variables.json``, and hydrates the manuscript
sources. Implementation lives in :mod:`gnn.manuscript.variables`.
"""

from gnn.manuscript.variables import (
    RepositorySnapshot,
    config_metadata_drift,
    generate_variables,
    load_variables,
    save_variables,
    select_cross_framework_family,
    sync_config_metadata,
    token_checksum,
)

__all__ = [
    "RepositorySnapshot",
    "config_metadata_drift",
    "generate_variables",
    "load_variables",
    "save_variables",
    "select_cross_framework_family",
    "sync_config_metadata",
    "token_checksum",
]
