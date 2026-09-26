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

from .pages import SITE_PAGES, is_valid_page, page_filenames, page_names

logger = logging.getLogger(__name__)

#: Pages a complete website build produces, in pipeline order.
#: Derived from the one site page catalogue (``gnn.website.pages.SITE_PAGES``).
KEY_PAGES: tuple[str, ...] = page_filenames()
#: Marker appended when page content is capped (mirrors ``generator._truncate``
#: so page reads and page rendering agree on the truncation shape).
TRUNCATION_MARKER = "\n\n… [truncated]"

#: Default character cap for page content returned by ``read_website_page``.
DEFAULT_PAGE_MAX_CHARS = 20000


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
        model_dir = wdir / "model"
        model_pages = (
            sorted(p.relative_to(wdir).as_posix() for p in model_dir.rglob("*.html"))
            if model_dir.is_dir()
            else []
        )

        return {
            "success": True,
            "directory": str(wdir),
            "pages": [p.name for p in pages],
            "pages_count": len(pages),
            "model_pages": model_pages,
            "assets_count": len(assets),
            "total_size_bytes": total_size,
            "completeness": completeness,
            "all_key_pages_present": all(completeness.values()),
        }
    except OSError as e:
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


def read_website_page(
    website_directory: str | Path,
    page_name: str,
    max_chars: int = DEFAULT_PAGE_MAX_CHARS,
) -> dict[str, Any]:
    """Read one catalogue page's HTML from a generated website.

    The Python API and the MCP ``get_website_page`` tool share this one
    implementation. ``page_name`` must be a page key from the site
    catalogue (``gnn.website.pages.page_names()``); the output filename
    resolves from the same catalogue.

    Returns a dict with the page filename, path, size, and content — the
    content capped at ``max_chars`` characters with an explicit marker
    when truncated. ``success`` is False (with an ``error`` message) for
    an unknown page name, a missing website directory, or a missing page
    file.
    """
    try:
        if not is_valid_page(page_name):
            valid = ", ".join(page_names())
            return {
                "success": False,
                "error": f"Unknown page name {page_name!r} (valid pages: {valid})",
            }
        wdir = Path(website_directory)
        if not wdir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {website_directory}",
            }
        spec = next(spec for spec in SITE_PAGES if spec.name == page_name)
        page_file = wdir / spec.filename
        if not page_file.exists():
            return {
                "success": False,
                "error": f"Page not found: {spec.filename} in {wdir}",
            }
        content = page_file.read_text(encoding="utf-8", errors="replace")
        total_chars = len(content)
        truncated = total_chars > max_chars
        if truncated:
            content = content[:max_chars] + TRUNCATION_MARKER
        return {
            "success": True,
            "page": page_name,
            "filename": spec.filename,
            "path": str(page_file),
            "size_bytes": page_file.stat().st_size,
            "total_chars": total_chars,
            "truncated": truncated,
            "content": content,
        }
    except Exception as e:
        logger.error(f"read_website_page error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
