#!/usr/bin/env python3
"""The token map, loaded so that a figure records which values it drew.

A manuscript figure that prints a repository count is a *second* renderer of
the same token the prose renders. The prose is re-hydrated from
``output/data/manuscript_variables.json`` on every render, but a figure is a
committed PNG: nothing re-runs its generator, so once a count moves in the
token map the committed PNG keeps printing the old one. That is exactly how
``fig:repo_metrics`` came to print 365 test files on the same PDF page on which
the prose printed 367.

The fix is to make the disagreement mechanically detectable. Generators load
the token map through :func:`load_tokens`, which returns a mapping that
remembers every key the generator actually read, along with the value it read.
When ``scripts/manuscript_build_figures.py`` runs a generator it sets
``GNN_FIGURE_TOKEN_PROVENANCE`` to a file path; the recorded ``{key: value}``
pairs are written there at interpreter exit and land in
``output/figures/figure_registry.json``. ``src/tests/test_manuscript_figure_freshness.py``
then compares those recorded values against the live token map, so a committed
figure built before a count moved fails the suite instead of shipping.

Recording is observed, not declared: the keys come from real reads —
``__getitem__``/``.get``, membership tests (``key in tokens``) and iteration
(``for key in tokens``, ``.keys()``/``.items()``/``.values()``) all record —
so a generator cannot claim to consume a token it never read, or quietly read
one it did not declare.

Run standalone (no ``GNN_FIGURE_TOKEN_PROVENANCE`` in the environment) the
mapping behaves as an ordinary ``dict`` and writes nothing.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The deterministic producer's output — the single source for every count.
TOKENS_PATH = _REPO_ROOT / "output" / "data" / "manuscript_variables.json"

#: Set by the figure build to the file the consumed ``{key: value}`` map is written to.
PROVENANCE_ENV = "GNN_FIGURE_TOKEN_PROVENANCE"

# Accumulated across every load_tokens() call in one interpreter, so a generator
# that loads the map twice records the union rather than losing the first read.
_CONSUMED: dict[str, str] = {}
_DUMP_REGISTERED = False


class RecordingTokens(dict):
    """The token map, remembering every key read through it.

    Values are recorded as ``str`` because that is how they are compared: the
    producer writes every token as a string, and the figure prints a string.
    Reads are observed on every access path, not just subscripting: direct
    ``__getitem__``/``.get``, membership tests, and full iterations
    (``__iter__``, ``keys``, ``items``, ``values``) all record the keys they
    touch. A generator that sweeps the whole map honestly records the whole
    map.
    """

    def __getitem__(self, key: str) -> Any:
        value = super().__getitem__(key)
        _CONSUMED[key] = str(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Record the read when the key is present; do not invent a token when it is not."""
        if key in self:
            return self[key]
        return default

    def __contains__(self, key: object) -> bool:
        """Record a present-key membership test; absent keys record nothing."""
        if dict.__contains__(self, key):
            assert isinstance(key, str)
            _CONSUMED[key] = str(dict.__getitem__(self, key))
            return True
        return False

    def __iter__(self):
        """Iteration is consumption: every yielded key is recorded."""
        for key in dict.keys(self):
            _CONSUMED[key] = str(dict.__getitem__(self, key))
            yield key

    def keys(self):
        """Record the whole key set (a ``keys()`` call sees every key)."""
        for key in dict.keys(self):
            _CONSUMED[key] = str(dict.__getitem__(self, key))
        return dict.keys(self)

    def items(self):
        """Record the whole map; the caller is handed the real view."""
        for key in dict.keys(self):
            _CONSUMED[key] = str(dict.__getitem__(self, key))
        return dict.items(self)

    def values(self):
        """Record the whole map: values are only meaningful with their keys."""
        for key in dict.keys(self):
            _CONSUMED[key] = str(dict.__getitem__(self, key))
        return dict.values(self)


def load_tokens() -> RecordingTokens:
    """Load ``manuscript_variables.json`` as a mapping that records its own reads."""
    global _DUMP_REGISTERED
    with TOKENS_PATH.open(encoding="utf-8") as fh:
        tokens = RecordingTokens(json.load(fh))
    destination = os.environ.get(PROVENANCE_ENV)
    if destination and not _DUMP_REGISTERED:
        atexit.register(_write_provenance, Path(destination))
        _DUMP_REGISTERED = True
    return tokens


def _write_provenance(destination: Path) -> None:
    """Write the consumed ``{key: value}`` pairs for the figure build to collect."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_CONSUMED, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
