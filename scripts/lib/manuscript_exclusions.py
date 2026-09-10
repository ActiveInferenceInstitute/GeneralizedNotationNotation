#!/usr/bin/env python3
"""Single source for the manuscript filename exclusion sets.

Four call sites used to hand-copy these sets — the token gate's standalone
fallback, the figure build's label scan, the published-commands test and the
figure-build test — and they drifted independently (adding a fifth authoring
guide meant editing four places; missing one silently scanned or skipped the
wrong files). They live here now, with the contract stated once:

``EXCLUDED_DOC_FILENAMES``
    ``manuscript/*.md`` files the render pipeline does NOT token-substitute,
    so no scanner may treat their text as published prose. Frozen mirror of
    ``infrastructure.rendering.manuscript_injection.EXCLUDED_DOC_FILENAMES``;
    the token gate still prefers the template's own value when the template
    checkout is importable, because that is what actually ships — this set is
    the standalone fallback and the shared literal.

``AUTHORING_GUIDE_FILENAMES``
    The subset that is a mere authoring guide (examples, commands, embedded
    figures that are not the manuscript's own). Label scans and command scans
    skip these so a worked example in SYNTAX.md is never read as a declared
    figure or a published command. A superset would hide real content, so
    each consumer imports exactly this set rather than re-deciding.
"""

from __future__ import annotations

EXCLUDED_DOC_FILENAMES = frozenset(
    {"AGENTS.md", "MANUSCRIPT_STATUS.md", "README.md", "SYNTAX.md"}
)

AUTHORING_GUIDE_FILENAMES = frozenset({"SYNTAX.md", "README.md", "AGENTS.md"})
