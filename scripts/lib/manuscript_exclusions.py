#!/usr/bin/env python3
"""The one exclusion set for manuscript ``*.md`` authoring guides.

Four sites used to hand-duplicate the same ``{SYNTAX.md, README.md,
AGENTS.md}`` skip set — the token gate, the figure build, and two tests. A
fifth filename added to one site and not the others made the sites disagree
about what "the manuscript" is. Import :data:`AUTHORING_GUIDE_SKIP` everywhere
instead; the fallback set in ``scripts/check_manuscript_tokens.py`` (used when
the template checkout is absent) is a strict superset documented at its
definition and is deliberately not folded in here.
"""

from __future__ import annotations

#: manuscript/*.md files that are authoring guides, not published sections:
#: their example embeds, commands and labels are documentation, not claims.
AUTHORING_GUIDE_SKIP = frozenset({"SYNTAX.md", "README.md", "AGENTS.md"})

__all__ = ["AUTHORING_GUIDE_SKIP"]
