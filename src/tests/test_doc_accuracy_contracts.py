"""Doc-accuracy contracts that keep documentation from rotting silently.

Two contract classes, both added after the 2026-08-07/10 documentation
review found the corresponding drift:

1. Orchestrator line counts — 14 of 25 module docs carried stale
   ``**Orchestrator**: `src/NN_x.py` (NN lines)`` figures because nothing
   checked them. This test parses every such citation under
   ``doc/gnn/modules/`` and compares it to the real file length, so a
   source edit that changes an orchestrator's length fails loudly here
   until the doc is updated (or the count is deliberately dropped).

2. Version banners — 24 docs carried a ``v1.6.0 Engine`` banner two major
   versions after the engine moved on. Every ``**Version**: vX.Y.Z
   Engine`` banner under ``doc/`` must carry the current
   ``pyproject.toml`` version.

Pure filesystem checks: no skips, no external tools.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ORCHESTRATOR_RE = re.compile(
    r"\*\*Orchestrator\*\*:\s*`(?P<script>src/[^`]+\.py)`\s*\((?P<count>\d+)\s+lines?\)"
)
_BANNER_RE = re.compile(r"\*\*Version\*\*:\s*v(?P<version>\d+\.\d+\.\d+)\s+Engine")


def _pyproject_version() -> str:
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(?P<v>\d+\.\d+\.\d+)"', text, re.M)
    assert match, "pyproject.toml carries no version"
    return match.group("v")


def test_module_doc_orchestrator_line_counts_match_source() -> None:
    """Every documented orchestrator line count equals the real file length."""
    module_docs = sorted((PROJECT_ROOT / "doc" / "gnn" / "modules").glob("*.md"))
    assert module_docs, "doc/gnn/modules is empty — layout changed?"

    citations = 0
    mismatches: list[str] = []
    for doc in module_docs:
        for match in _ORCHESTRATOR_RE.finditer(doc.read_text(encoding="utf-8")):
            citations += 1
            script = PROJECT_ROOT / match.group("script")
            if not script.exists():
                mismatches.append(
                    f"{doc.name}: cites {match.group('script')} which does not exist"
                )
                continue
            actual = len(script.read_text(encoding="utf-8").splitlines())
            documented = int(match.group("count"))
            if actual != documented:
                mismatches.append(
                    f"{doc.name}: {match.group('script')} documented as "
                    f"{documented} lines, actually {actual}"
                )
    assert citations >= 20, (
        f"only {citations} orchestrator citations found — the doc convention "
        "changed; update this test's pattern rather than deleting it"
    )
    assert not mismatches, "stale orchestrator line counts:\n" + "\n".join(mismatches)


def test_doc_version_banners_match_pyproject() -> None:
    """Every '**Version**: vX.Y.Z Engine' banner carries the current version."""
    current = _pyproject_version()
    stale: list[str] = []
    banners = 0
    for doc in sorted((PROJECT_ROOT / "doc").rglob("*.md")):
        for match in _BANNER_RE.finditer(doc.read_text(encoding="utf-8")):
            banners += 1
            if match.group("version") != current:
                stale.append(
                    f"{doc.relative_to(PROJECT_ROOT)}: banner v{match.group('version')} "
                    f"!= pyproject v{current}"
                )
    assert banners >= 20, (
        f"only {banners} version banners found — banner convention changed; "
        "update this test's pattern rather than deleting it"
    )
    assert not stale, "stale engine version banners:\n" + "\n".join(stale)
