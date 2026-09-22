"""Public-surface contract tests for gnn.main.

COMP-004: ``resolve_steps_to_execute`` is the public cross-module step
resolver; private underscore imports of ``gnn.main`` internals are
prohibited across the package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

SRC_GNN = Path(__file__).resolve().parents[2] / "src" / "gnn"


def test_public_step_resolver_is_importable_and_callable() -> None:
    from gnn.main import resolve_steps_to_execute

    assert callable(resolve_steps_to_execute)


def test_no_private_gnn_main_imports_anywhere_in_src() -> None:
    # A naive ``from gnn\.main import .*_[a-z]`` regex backtracks into
    # snake_case names ("resolve_steps_to_execute" contains "_s"), so scan
    # per imported token: every symbol imported from ``gnn.main`` must be
    # public, i.e. not underscore-prefixed. This is the executable form of
    # ``grep -rn 'from gnn\.main import _' src/`` -> 0 hits.
    import_line = re.compile(r"^\s*from gnn\.main import (.+)$")
    offenders: list[str] = []
    for py_file in sorted(SRC_GNN.rglob("*.py")):
        for lineno, line in enumerate(
            py_file.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = import_line.match(line)
            if not match:
                continue
            for token in match.group(1).split(","):
                name = token.strip().split(" as ")[0].strip()
                if name.startswith("_"):
                    offenders.append(
                        f"{py_file.relative_to(SRC_GNN)}:{lineno}: {line.strip()}"
                    )
                    break
    assert offenders == [], "private gnn.main imports found:\n" + "\n".join(offenders)
