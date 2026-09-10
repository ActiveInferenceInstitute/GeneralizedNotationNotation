#!/usr/bin/env python3
"""The one exclusion set for manuscript ``*.md`` authoring guides.

Four sites used to hand-duplicate the same ``{SYNTAX.md, README.md,
AGENTS.md}`` skip set — the token gate, the figure build, and two tests. A
fifth filename added to one site and not the others made the sites disagree
about what "the manuscript" is. Import :data:`AUTHORING_GUIDE_SKIP` everywhere
instead. The token gate additionally never scans the producer's status log;
that superset lives here too as :data:`TOKEN_GATE_FALLBACK_EXCLUSIONS` (the
gate uses it only when the template checkout is absent — the template's own
``EXCLUDED_DOC_FILENAMES`` wins otherwise).
"""

from __future__ import annotations

#: manuscript/*.md files that are authoring guides, not published sections:
#: their example embeds, commands and labels are documentation, not claims.
AUTHORING_GUIDE_SKIP = frozenset({"SYNTAX.md", "README.md", "AGENTS.md"})

#: The token gate's standalone fallback: authoring guides PLUS the producer's
#: status log, which is an internal dashboard, not a published section — but
#: also not an authoring guide, so the figure/test sites do not skip it.
TOKEN_GATE_FALLBACK_EXCLUSIONS = AUTHORING_GUIDE_SKIP | {"MANUSCRIPT_STATUS.md"}

__all__ = ["AUTHORING_GUIDE_SKIP", "TOKEN_GATE_FALLBACK_EXCLUSIONS"]
