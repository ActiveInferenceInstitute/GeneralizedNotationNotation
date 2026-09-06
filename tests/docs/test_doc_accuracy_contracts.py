"""Doc-accuracy contracts that keep documentation from rotting silently.

Two contract classes, both added after the 2026-08-07/10 documentation
review found the corresponding drift:

1. Orchestrator line counts — 14 of 25 module docs carried stale
   ``**Orchestrator**: `src/NN_x.py` (NN lines)`` figures because nothing
   checked them. This test parses every such citation under
   ``doc/gnn/modules/`` and compares it to the real file length, so a
   source edit that changes an orchestrator's length fails loudly here
   until the doc is updated (or the count is deliberately dropped).

2. Package-version ownership — module docs copied a package version that
   drifted two major versions behind ``pyproject.toml``. Module docs now point
   to the package metadata instead of transcribing a moving value.

Pure filesystem checks: no skips, no external tools.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ORCHESTRATOR_RE = re.compile(
    r"\*\*Orchestrator\*\*:\s*`(?P<script>src/[^`]+\.py)`\s*\((?P<count>\d+)\s+lines?\)"
)
_MODULE_VERSION_RE = re.compile(
    r"(?:\*\*Version\*\*|### Current Version):\s*v?\d+\.\d+\.\d+"
)
_PACKAGE_VERSION_POINTER = "[pyproject.toml](../../../pyproject.toml)"


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


def test_module_docs_point_to_package_version_source() -> None:
    """Module docs link to package metadata instead of copying its version."""
    module_docs = sorted((PROJECT_ROOT / "doc" / "gnn" / "modules").glob("*.md"))
    copied: list[str] = []
    pointers = 0
    for doc in module_docs:
        text = doc.read_text(encoding="utf-8")
        pointers += text.count(_PACKAGE_VERSION_POINTER)
        if _MODULE_VERSION_RE.search(text):
            copied.append(str(doc.relative_to(PROJECT_ROOT)))
    assert pointers >= 20, (
        f"only {pointers} package-version pointers found — the module-doc "
        "convention changed; update this test and the docs together"
    )
    assert not copied, "module docs copy a moving package version:\n" + "\n".join(
        copied
    )
