"""Website inspection helpers — pure filesystem queries over a generated site.

Extracted from ``mcp.py`` so the Python API and the MCP tools share one
implementation: the MCP layer stays thin and the same inventory is available
programmatically via :func:`inspect_website` / :func:`list_website_pages`.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Pages a complete website build produces, in pipeline order.
KEY_PAGES: tuple[str, ...] = (
    "index.html",
    "pipeline.html",
    "gnn_files.html",
    "analysis.html",
    "visualization.html",
    "reports.html",
    "mcp.html",
)


def inspect_website(website_directory: str | Path) -> dict[str, Any]:
    """Inspect a generated website directory.

    Returns a dict with the page inventory, total size, per-key-page
    presence, and an ``all_key_pages_present`` flag. ``success`` is False
    (with an ``error`` message) when the directory does not exist.
    """
    try:
        wdir = Path(website_directory)
        if not wdir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {website_directory}",
            }

        pages = sorted(wdir.glob("*.html"))
        assets_dir = wdir / "assets"
        assets = list(assets_dir.glob("*")) if assets_dir.exists() else []
        completeness = {page: (wdir / page).exists() for page in KEY_PAGES}
        total_size = sum(f.stat().st_size for f in pages if f.exists())

        return {
            "success": True,
            "directory": str(wdir),
            "pages": [p.name for p in pages],
            "pages_count": len(pages),
            "assets_count": len(assets),
            "total_size_bytes": total_size,
            "completeness": completeness,
            "all_key_pages_present": all(completeness.values()),
        }
    except Exception as e:
        logger.error(f"inspect_website error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_website_pages(website_directory: str | Path) -> dict[str, Any]:
    """List all HTML pages in a generated website with size/mtime metadata."""
    try:
        wdir = Path(website_directory)
        if not wdir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {website_directory}",
            }

        pages: list[dict[str, Any]] = []
        for html_file in sorted(wdir.glob("*.html")):
            stat = html_file.stat()
            pages.append(
                {
                    "name": html_file.name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

        return {
            "success": True,
            "directory": str(wdir),
            "pages": pages,
            "total_pages": len(pages),
        }
    except Exception as e:
        logger.error(f"list_website_pages error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
