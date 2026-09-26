#!/usr/bin/env python3
"""Single page catalogue for the generated GNN website.

``SITE_PAGES`` is the one ordered description of the pages a complete
build writes. Every page inventory derives from it instead of repeating
its own hardcoded list:

- ``generator.py`` maps filenames to the page builders and renders the
  sidebar navigation from the catalogue;
- ``inspection.KEY_PAGES`` (key-page completeness) is the catalogue's
  filenames;
- the module-info page list (``mcp.py``) is the catalogue's page names.

Pipeline-step facts stay registry-derived: the catalogue imports the
canonical ``gnn.pipeline.step_registry.STEPS`` — the same registry source
``generator.py`` uses for its step catalogue — so a new pipeline step
updates derived text without edits here. The fixed site furniture (the
seven pages themselves) is explicit below.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnn.pipeline.step_registry import STEPS as _REGISTRY_STEPS

__all__ = [
    "PageSpec",
    "SITE_PAGES",
    "is_valid_page",
    "page_count",
    "page_filenames",
    "page_names",
]


@dataclass(frozen=True)
class PageSpec:
    """One static page of the generated website.

    ``name`` doubles as the nav active key, ``title`` as the nav label and
    page title, and ``builder`` names the ``WebsiteGenerator`` method that
    renders the page.
    """

    name: str
    title: str
    builder: str
    description: str
    icon: str

    @property
    def filename(self) -> str:
        """Output filename of this page, e.g. ``index.html``."""
        return f"{self.name}.html"


#: The site's pages, in pipeline order — the fixed site furniture.
SITE_PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        name="index",
        title="Dashboard",
        builder="_page_index",
        description="Pipeline dashboard with step cards and summary stats.",
        icon="🏠",
    ),
    PageSpec(
        name="pipeline",
        title="Pipeline",
        builder="_page_pipeline",
        description=f"Full {len(_REGISTRY_STEPS)}-step pipeline status table.",
        icon="⚡",
    ),
    PageSpec(
        name="gnn_files",
        title="GNN Files",
        builder="_page_gnn_files",
        description="Browser for the parsed GNN source files.",
        icon="📂",
    ),
    PageSpec(
        name="analysis",
        title="Analysis",
        builder="_page_analysis",
        description="Statistical analysis results from step 16 artifacts.",
        icon="📊",
    ),
    PageSpec(
        name="visualization",
        title="Visualizations",
        builder="_page_visualization",
        description="Gallery of all generated visualization artifacts.",
        icon="🖼️",
    ),
    PageSpec(
        name="reports",
        title="Reports",
        builder="_page_reports",
        description="Viewer for JSON/text report artifacts.",
        icon="📋",
    ),
    PageSpec(
        name="mcp",
        title="MCP Tools",
        builder="_page_mcp",
        description="MCP tools registry across all modules, from step 21 artifacts.",
        icon="🔧",
    ),
)


def page_names() -> tuple[str, ...]:
    """Ordered page keys (``"index"``, ``"pipeline"``, ...)."""
    return tuple(page.name for page in SITE_PAGES)


def page_filenames() -> tuple[str, ...]:
    """Ordered output filenames (``"index.html"``, ...)."""
    return tuple(page.filename for page in SITE_PAGES)


def is_valid_page(name: str) -> bool:
    """True when ``name`` is a known page key."""
    return name in page_names()


def page_count() -> int:
    """Number of pages the site catalogue defines.

    Stable receipt hook for downstream lanes that pin page counts.
    """
    return len(SITE_PAGES)
