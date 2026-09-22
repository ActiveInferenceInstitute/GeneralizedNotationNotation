#!/usr/bin/env python3
"""
Website generator module for GNN pipeline.

Generates a full-featured, premium, dark-mode static HTML website
from GNN pipeline artifacts. Produces 7+ pages:
  - index.html         — Pipeline dashboard with step cards
  - pipeline.html      — Full 25-step pipeline status table
  - gnn_files.html     — GNN source file browser
  - analysis.html      — Analysis and complexity metrics
  - visualization.html — Gallery of all generated visualizations
  - reports.html       — JSON/text report viewer
  - mcp.html           — MCP tools registry across all modules
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gnn.pipeline.step_registry import STEPS as _REGISTRY_STEPS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Shared design system — dark-mode CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
/* ── GNN Pipeline Premium Design System ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-gradient: radial-gradient(circle at top right, #1b1e32, #07090f 80%);
  --bg-surface:  rgba(22, 27, 34, 0.45);
  --bg-card:     rgba(28, 34, 48, 0.6);
  --bg-hover:    rgba(33, 39, 58, 0.85);
  --border:      rgba(255, 255, 255, 0.08);
  --accent:      #a881fb;
  --accent-2:    #00f0ff;
  --success:     #3fb950;
  --warning:     #e3b341;
  --error:       #fc6c65;
  --text-1:      #ffffff;
  --text-2:      #aeb6c2;
  --text-3:      #606a78;
  --radius:      12px;
  --radius-lg:   16px;
  --shadow:      0 8px 32px rgba(0, 0, 0, 0.5);
  --glow:        0 0 24px rgba(168, 129, 251, 0.4);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

body {
  font-family: 'Outfit', 'Inter', sans-serif;
  background: #07090f;
  background-image: var(--bg-gradient);
  background-attachment: fixed;
  color: var(--text-1);
  min-height: 100vh;
  display: flex;
}

/* ── Sidebar (Glass) ── */
.sidebar {
  width: 260px;
  flex-shrink: 0;
  background: var(--bg-surface);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 0;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  box-shadow: 4px 0 24px rgba(0,0,0,0.2);
}
.sidebar-logo {
  padding: 24px 20px 20px;
  border-bottom: 1px solid var(--border);
}
.sidebar-logo h2 {
  font-size: 18px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.02em;
}
.sidebar-logo p { color: var(--text-2); font-size: 12px; margin-top: 4px; font-weight: 300; }
.sidebar-nav { padding: 16px 12px; flex: 1; }
.nav-section { margin-bottom: 12px; }
.nav-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-3);
  padding: 4px 12px;
  margin-bottom: 4px;
}
.nav-link {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: var(--radius);
  color: var(--text-2);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.nav-link:hover, .nav-link.active {
  background: rgba(255,255,255,0.05);
  color: var(--text-1);
  transform: translateX(4px);
}
.nav-link.active {
  background: linear-gradient(90deg, rgba(168,129,251,0.15), transparent);
  border-left: 3px solid var(--accent);
  color: var(--text-1);
}
.nav-link .icon { font-size: 16px; width: 22px; text-align: center; }

/* ── Main content ── */
.main {
  flex: 1;
  min-width: 0;
  padding: 40px 48px;
  overflow-x: auto;
  animation: fadeIn 0.6s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ── Page header ── */
.page-header {
  margin-bottom: 36px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}
.page-header h1 {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-1);
  letter-spacing: -0.02em;
}
.page-header .subtitle {
  color: var(--text-2);
  font-size: 15px;
  margin-top: 8px;
  font-weight: 300;
}

/* ── Stat cards (Glass) ── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 36px;
}
.stat-card {
  background: var(--bg-card);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}
.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: transparent;
  transition: background 0.3s ease;
}
.stat-card:hover { 
  transform: translateY(-6px) scale(1.02);
  box-shadow: var(--glow); 
  border-color: rgba(255,255,255,0.15);
}
.stat-card.success:hover::before { background: var(--success); }
.stat-card.accent:hover::before { background: var(--accent); }
.stat-card.accent2:hover::before { background: var(--accent-2); }

.stat-card .label { font-size: 11px; color: var(--text-2); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.stat-card .value { font-size: 36px; font-weight: 700; margin-top: 8px; letter-spacing: -0.03em; }
.stat-card .sub   { font-size: 12px; color: var(--text-2); margin-top: 4px; font-weight: 300; }

.stat-card.success .value { color: var(--success); text-shadow: 0 0 16px rgba(63,185,80,0.4); }
.stat-card.accent  .value { color: var(--accent); text-shadow: 0 0 16px rgba(168,129,251,0.4); }
.stat-card.accent2 .value { color: var(--accent-2); text-shadow: 0 0 16px rgba(0,240,255,0.4); }

/* ── Step grid ── */
.step-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 40px;
}
.step-card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
  transition: all 0.3s ease;
  cursor: default;
  display: flex;
  flex-direction: column;
}
.step-card:hover {
  border-color: rgba(255,255,255,0.2);
  box-shadow: 0 12px 32px rgba(0,0,0,0.6), var(--glow);
  transform: translateY(-4px);
  background: var(--bg-hover);
}
.step-card .step-num {
  font-size: 10px;
  font-weight: 700;
  color: var(--text-3);
  letter-spacing: 0.15em;
}
.step-card .step-name {
  font-size: 15px;
  font-weight: 600;
  margin: 6px 0 10px;
  color: var(--text-1);
}
.step-card .step-desc {
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.6;
  font-weight: 300;
  flex-grow: 1;
}
.step-card .step-badge {
  display: inline-flex;
  align-items: center;
  margin-top: 16px;
  font-size: 10px;
  font-weight: 700;
  padding: 4px 10px;
  border-radius: 20px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  align-self: flex-start;
}

/* Animations for badges */
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}
.badge-ok      { background: rgba(63,185,80,0.15);    color: var(--success); border: 1px solid rgba(63,185,80,0.3); }
.badge-skip    { background: rgba(227,179,65,0.15);   color: var(--warning); border: 1px solid rgba(227,179,65,0.3); }
.badge-error   { background: rgba(252,108,101,0.15);  color: var(--error); border: 1px solid rgba(252,108,101,0.3); }
.badge-pending { 
  background: rgba(255,255,255,0.05); color: var(--text-2); border: 1px solid var(--border);
  animation: pulse 2s infinite ease-in-out; 
}

/* ── Table (Glass) ── */
.table-wrap { 
  overflow-x: auto; 
  border-radius: var(--radius-lg); 
  border: 1px solid var(--border); 
  background: var(--bg-card);
  backdrop-filter: blur(12px);
}
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th {
  background: rgba(0,0,0,0.2);
  color: var(--text-2);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 14px 18px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
td {
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  color: var(--text-1);
  vertical-align: top;
  font-weight: 300;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.03); }

/* ── Code ── */
pre, code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}
pre {
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px;
  overflow-x: auto;
  line-height: 1.6;
  color: var(--text-2);
  box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
}
code { color: var(--accent-2); }

/* ── Section ── */
.section      { margin-bottom: 40px; }
.section-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
  color: var(--text-1);
  display: flex;
  align-items: center;
  gap: 12px;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Cards ── */
.card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px;
  margin-bottom: 16px;
  transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow);
  border-color: rgba(255,255,255,0.15);
}
.card h3 { font-size: 16px; font-weight: 600; margin-bottom: 10px; }
.card p  { font-size: 14px; color: var(--text-2); line-height: 1.6; font-weight: 300; }

/* ── Visualization gallery ── */
.viz-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}
.viz-card {
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  flex-direction: column;
}
.viz-card:hover { 
  box-shadow: 0 16px 40px rgba(0,0,0,0.6), var(--glow); 
  transform: translateY(-6px);
  border-color: rgba(255,255,255,0.2);
}
.viz-card img  { width: 100%; display: block; border-bottom: 1px solid var(--border); }
.viz-card .viz-info { padding: 16px 20px; background: rgba(0,0,0,0.2); flex-grow: 1; }
.viz-card .viz-title { font-size: 14px; font-weight: 600; }
.viz-card .viz-desc  { font-size: 12px; color: var(--text-2); margin-top: 4px; font-weight: 300; }

/* ── MCP tool card ── */
.tool-card {
  background: rgba(0,0,0,0.2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-2);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-bottom: 12px;
  transition: background 0.2s;
}
.tool-card:hover { background: rgba(255,255,255,0.05); }
.tool-card .tool-name { font-family: 'JetBrains Mono', monospace; font-size: 14px; color: var(--accent-2); font-weight: 600; }
.tool-card .tool-mod  { font-size: 11px; color: var(--text-3); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.1em; }
.tool-card .tool-desc { font-size: 13px; color: var(--text-2); margin-top: 8px; line-height: 1.6; font-weight: 300; }

/* ── Pill badge ── */
.pill {
  display: inline-block;
  font-size: 10px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Collapsible (Glass) ── */
details { margin-bottom: 12px; }
summary {
  cursor: pointer;
  background: var(--bg-card);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 18px;
  font-size: 14px;
  font-weight: 600;
  list-style: none;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background 0.2s, border-color 0.2s;
}
summary:hover { background: var(--bg-hover); border-color: rgba(255,255,255,0.15); }
summary::-webkit-details-marker { display: none; }
summary::after { content: '▸'; color: var(--text-3); font-size: 16px; transition: transform 0.2s; }
details[open] summary::after { transform: rotate(90deg); }
details[open] summary { border-radius: var(--radius) var(--radius) 0 0; background: rgba(0,0,0,0.3); border-bottom: none; }
.details-body {
  border: 1px solid var(--border);
  border-top: none;
  border-radius: 0 0 var(--radius) var(--radius);
  padding: 16px 18px;
  background: rgba(0,0,0,0.2);
  backdrop-filter: blur(12px);
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .sidebar { position: fixed; transform: translateX(-100%); z-index: 100; transition: transform 0.3s; }
  .sidebar.open { transform: translateX(0); }
  .main { padding: 20px; }
  .page-header h1 { font-size: 26px; }
  .stat-card .value { font-size: 28px; }
}
"""


def _page(title: str, active: str, body: str, *, nav_extra: str = "") -> str:
    """Wrap body in the shared page shell with sidebar and nav."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    nav_items: list[tuple[str, str, str, str]] = [
        ("🏠", "Dashboard", "index.html", "index"),
        ("⚡", "Pipeline", "pipeline.html", "pipeline"),
        ("📂", "GNN Files", "gnn_files.html", "gnn_files"),
        ("📊", "Analysis", "analysis.html", "analysis"),
        ("🖼️", "Visualizations", "visualization.html", "visualization"),
        ("📋", "Reports", "reports.html", "reports"),
        ("🔧", "MCP Tools", "mcp.html", "mcp"),
    ]
    nav_html = ""
    for icon, label, href, key in nav_items:
        cls = "nav-link active" if key == active else "nav-link"
        nav_html += f'<a href="{href}" class="{cls}"><span class="icon">{icon}</span>{label}</a>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="GNN Pipeline Results — {title}">
  <title>{title} — GNN Pipeline</title>
  <style>{_CSS}</style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-logo">
      <h2>GNN Pipeline</h2>
      <p>Generated {ts}</p>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-section">
        <div class="nav-label">Navigation</div>
        {nav_html}
      </div>
      {nav_extra}
    </nav>
  </aside>
  <main class="main">
    {body}
  </main>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline step catalogue
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepInfo:
    """One pipeline step in the static 25-step site catalogue."""

    number: int
    name: str
    description: str

    @property
    def script_name(self) -> str:
        """Conventional display name of the numbered orchestrator script."""
        return f"{self.number}_{self.name.lower().replace(' ', '_')}.py"


# Acronym casing for stem suffixes that must not be title-cased
# (``"mcp".title()`` would yield "Mcp", breaking the display name and the
# ``script_name`` round-trip back to the real orchestrator script stem).
_ACRONYM_DISPLAY: dict[str, str] = {
    "gnn": "GNN",
    "gui": "GUI",
    "llm": "LLM",
    "mcp": "MCP",
    "ml": "ML",
}


def _display_name_from_stem_suffix(suffix: str) -> str:
    """Display name for a registry stem suffix (``"advanced_viz"`` → ``"Advanced Viz"``)."""
    return " ".join(
        _ACRONYM_DISPLAY.get(word, word.title()) for word in suffix.split("_")
    )


def _steps_from_registry() -> tuple[StepInfo, ...]:
    """Derive the site catalogue from the canonical ``step_registry.STEPS``."""
    infos: list[StepInfo] = []
    for registry_step in _REGISTRY_STEPS:
        number_str, _, suffix = registry_step.script_stem.partition("_")
        infos.append(
            StepInfo(
                number=int(number_str),
                name=_display_name_from_stem_suffix(suffix),
                description=registry_step.description,
            )
        )
    return tuple(infos)


PIPELINE_STEPS: tuple[StepInfo, ...] = _steps_from_registry()


def get_pipeline_steps() -> tuple[StepInfo, ...]:
    """Return the immutable 25-step catalogue rendered across the site."""
    return PIPELINE_STEPS


# Shared status-badge styling: single source of truth for the index and
# pipeline pages (previously duplicated inline in each page builder).
_BADGE_CLASS: dict[str, str] = {
    "ok": "badge-ok",
    "error": "badge-error",
    "skip": "badge-skip",
}
_STEP_BADGE_LABEL: dict[str, str] = {
    "ok": "✓ Complete",
    "error": "✗ Error",
    "skip": "⊘ Skipped",
}
_PIPELINE_BADGE_LABEL: dict[str, str] = {
    "ok": "Complete",
    "error": "Error",
    "skip": "Skipped",
}


def _esc(value: Any) -> str:
    """HTML-escape any value for safe interpolation into page markup."""
    return escape(str(value))


def _stat_int(value: Any, default: int = 0) -> int:
    """Best-effort int for artifact-recorded stat values, falling back to ``default``.

    Guard against ``null``/non-numeric JSON fields: an unusable stat must
    render as the fallback instead of failing the whole page.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _truncate(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars with an explicit marker when truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n… [truncated]"


def _write_atomic(dest: Path, content: str) -> None:
    """Write ``content`` to ``dest`` via a temp file and atomic rename."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dest.parent, delete=False
    )
    try:
        with tmp:
            tmp.write(content)
        os.replace(tmp.name, dest)
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def _collect_gnn_files(p_root: Path, input_dir: Path) -> tuple[list[Path], bool]:
    """Discover GNN source markdown files, preferring ``<root>/input/gnn_files``."""
    for search_dir in (p_root.parent / "input" / "gnn_files", input_dir):
        if search_dir.exists():
            return sorted(search_dir.glob("*.md")), True
    return [], False


def _load_pipeline_summary(p_root: Path) -> dict[str, Any]:
    """Load ``pipeline_execution_summary.json`` (absent or malformed → ``{}``)."""
    for candidate in (
        p_root / "00_pipeline_summary" / "pipeline_execution_summary.json",
        p_root / "pipeline_execution_summary.json",
    ):
        if not candidate.exists():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug(f"Skipped malformed pipeline summary file: {e}")
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _normalize_website_step_status(raw: Any) -> str | None:
    """Map a recorded pipeline step status to the site badge vocabulary.

    Returns ``ok``/``skip``/``error``, or ``None`` for an empty/unusable
    status (the step then stays ``pending``).
    """
    normalized = str(raw or "").strip().upper()
    if not normalized:
        return None
    if "SKIP" in normalized:
        return "skip"
    if "SUCCESS" in normalized and "PARTIAL" not in normalized:
        return "ok"
    if normalized in ("PASS", "PASSED", "OK", "COMPLETED"):
        return "ok"
    return "error"


def _step_statuses_from_summary(summary: dict[str, Any]) -> dict[int, str]:
    """Per-step site statuses derived from the pipeline summary's ``steps`` list."""
    statuses: dict[int, str] = {}
    raw_steps = summary.get("steps")
    if not isinstance(raw_steps, list):
        return statuses
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        match = re.match(r"(?P<number>\d+)_", str(raw_step.get("script_name", "")))
        if match:
            step_number = int(match.group("number"))
        elif isinstance(raw_step.get("step_number"), int):
            step_number = int(raw_step["step_number"])
        else:
            continue
        badge = _normalize_website_step_status(raw_step.get("status"))
        if badge is not None:
            statuses[step_number] = badge
    return statuses


def _collect_step_statuses(p_root: Path) -> dict[int, str]:
    """Map each step number to ``ok``/``skip``/``error``/``pending``.

    Primary source is the durable ``pipeline_execution_summary.json`` written
    by the orchestrator: a step whose output dir exists but whose recorded
    status is ``FAILED`` or ``SKIPPED`` must not be advertised as complete.
    Falls back to the numbered-output-dir heuristic only when the summary is
    absent, malformed, or carries no parseable per-step records.
    """
    pending = {step.number: "pending" for step in PIPELINE_STEPS}
    if not p_root.exists():
        return pending
    summary = _load_pipeline_summary(p_root)
    if summary:
        from_summary = _step_statuses_from_summary(summary)
        if from_summary:
            return {**pending, **from_summary}
    dir_names = [d.name for d in p_root.iterdir() if d.is_dir()]
    statuses: dict[int, str] = {}
    for step in PIPELINE_STEPS:
        found = any(
            name.startswith(f"{step.number:02d}_") or name.startswith(f"{step.number}_")
            for name in dir_names
        )
        statuses[step.number] = "ok" if found else "pending"
    return statuses


def _collect_analysis_results(p_root: Path) -> list[Any]:
    """Load Step 16 analysis JSON files (best effort, deduplicated by name)."""
    results: list[Any] = []
    seen: set[str] = set()
    for candidate in (
        p_root / "16_analysis_output" / "analysis_results",
        p_root / "16_analysis_output",
    ):
        if not candidate.exists():
            continue
        for jf in candidate.glob("*.json"):
            if jf.name in seen:
                continue
            seen.add(jf.name)
            try:
                loaded = json.loads(jf.read_text())
            except Exception as e:
                logger.debug(f"Skipped malformed analysis file {jf.name}: {e}")
                continue
            if not isinstance(loaded, dict):
                logger.debug(f"Skipped non-dict analysis file {jf.name}")
                continue
            results.append(loaded)

    return results


def _load_mcp_summary(p_root: Path) -> dict[str, Any]:
    """Load the Step 21 MCP processing summary (absent or malformed → ``{}``)."""
    candidate = p_root / "21_mcp_output" / "mcp_processing_summary.json"
    if not candidate.exists():
        return {}
    try:
        loaded = json.loads(candidate.read_text())
    except Exception as e:
        logger.debug(f"Skipped malformed MCP summary file: {e}")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_registered_tools(p_root: Path) -> list[dict[str, str]]:
    """Map ``registered_tools.json`` (step 21) to the page tool-card shape."""
    candidate = p_root / "21_mcp_output" / "registered_tools.json"
    if not candidate.exists():
        logger.debug("No registered_tools.json found for website (step 21 optional)")
        return []
    try:
        loaded = json.loads(candidate.read_text())
    except Exception as e:
        logger.debug(f"Skipped malformed registered tools file: {e}")
        return []
    if not isinstance(loaded, list):
        logger.debug("registered_tools.json is not a list; ignoring it")
        return []
    cards: list[dict[str, str]] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        cards.append(
            {
                "name": item.get("name", ""),
                "module": item.get("module", ""),
                "desc": item.get("description") or item.get("desc") or "",
                "category": item.get("category", ""),
            }
        )
    return cards


def _collect_visualizations(
    viz_dirs: list[Path], assets_dir: Path
) -> list[dict[str, Any]]:
    """Copy PNG/HTML visualization artifacts into ``assets_dir`` and describe them.

    Identical artifact filenames from different source directories must not
    silently overwrite each other in the shared ``assets_dir``: later
    collisions are copied under a ``<source-dir>__``-prefixed (or counter-
    disambiguated) name so every artifact keeps its own gallery card.
    """
    visualizations: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for viz_dir in viz_dirs:
        if not viz_dir.exists():
            continue
        for pattern, artifact_type in (("*.png", "image"), ("*.html", "html")):
            for artifact in viz_dir.rglob(pattern):
                dest_name = artifact.name
                if dest_name.lower() in used_names:
                    prefix = f"{viz_dir.name}__"
                    dest_name = f"{prefix}{artifact.name}"
                    stem, ext = os.path.splitext(dest_name)
                    counter = 2
                    while dest_name.lower() in used_names:
                        dest_name = f"{stem}_{counter}{ext}"
                        counter += 1
                used_names.add(dest_name.lower())
                dest = assets_dir / dest_name
                try:
                    shutil.copy2(artifact, dest)
                except Exception:
                    dest = artifact
                visualizations.append(
                    {
                        "title": artifact.stem,
                        "path": dest.name,
                        "type": artifact_type,
                        "abs": dest,
                    }
                )
    return visualizations


def _collect_reports(p_root: Path) -> list[dict[str, Any]]:
    """Collect capped JSON artifacts from every numbered output directory."""
    reports: list[dict[str, Any]] = []
    if not p_root.exists():
        return reports
    for entry in sorted(p_root.iterdir()):
        if not (entry.is_dir() and entry.name[0].isdigit()):
            continue
        for jf in list(entry.rglob("*.json"))[:5]:  # cap per dir
            try:
                content = jf.read_text(encoding="utf-8", errors="replace")
                reports.append(
                    {
                        "name": jf.name,
                        "dir": entry.name,
                        "content": content[:2000],
                        "size": jf.stat().st_size,
                    }
                )
            except Exception as e:
                logger.debug(f"Skipped unreadable report file {jf.name}: {e}")
    return reports


def collect_website_data(
    pipeline_output_root: Path,
    input_dir: Path,
    assets_dir: Path,
    *,
    output_dir: Path | None = None,
    user_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate every artifact the website pages render.

    Pure with respect to the repository state except for copying
    visualization assets into ``assets_dir``. Step statuses come from the
    durable ``pipeline_execution_summary.json`` written by the orchestrator
    (numbered-output-dir heuristic only as a fallback when the summary is
    absent). The MCP page data comes from the step-21 artifacts:
    ``21_mcp_output/mcp_processing_summary.json`` for the summary and
    ``registered_tools.json`` for the tool inventory, so the site reflects
    what steps actually recorded.
    """
    p_root = Path(pipeline_output_root)
    data: dict[str, Any] = {
        "p_root": p_root,
        "output_dir": Path(output_dir) if output_dir is not None else None,
        "gnn_files": [],
        "analysis": [],
        "complexity": [],
        "visualizations": [],
        "reports": [],
        "mcp_tools": [],
        "mcp_summary": {},
        "step_statuses": {},
        "processed_files": 0,
        "gui_navigation": (p_root / "22_gui_output" / "navigation.html").is_file(),
    }
    if user_data:
        data.update(
            {
                k: v
                for k, v in user_data.items()
                if k not in ("output_dir", "input_dir", "pipeline_output_root")
            }
        )

    discovered, found_source = _collect_gnn_files(p_root, input_dir)
    data["gnn_files"].extend(discovered)
    if found_source:
        data["processed_files"] = len(data["gnn_files"])
    data["step_statuses"] = _collect_step_statuses(p_root)
    data["analysis"].extend(_collect_analysis_results(p_root))
    data["visualizations"].extend(
        _collect_visualizations(
            [
                p_root / "08_visualization_output" / "visualization_results",
                p_root / "8_visualization_output" / "visualization_results",
                p_root / "09_advanced_viz_output",
                p_root / "9_advanced_viz_output",
            ],
            assets_dir,
        )
    )
    data["reports"].extend(_collect_reports(p_root))
    data["mcp_summary"] = _load_mcp_summary(p_root)
    data["mcp_tools"] = _load_registered_tools(p_root)
    return data


class WebsiteGenerator:
    """Generates a premium multi-page static HTML website from pipeline artifacts."""

    def __init__(self) -> None:
        """Initialize the instance.

        MCP page data is sourced from the step-21 artifacts in the pipeline
        output root at generation time; no constructor configuration needed.
        """

    # ── Public API ──────────────────────────────────────────────────────────

    def generate_website(self, website_data: dict) -> dict:
        """Generate the complete static website.

        Each page is rendered and written independently: a failure on one
        page records an error and leaves the remaining pages intact, and the
        overall ``success`` flag is ``True`` only when no errors occurred.
        """
        result: dict[str, Any] = {
            "success": True,
            "pages_created": 0,
            "pages": [],
            "errors": [],
            "warnings": [],
        }
        try:
            output_dir = Path(
                website_data.get("output_dir", "output/20_website_output")
            )
            input_dir = Path(website_data.get("input_dir", "output"))
            p_root = Path(website_data.get("pipeline_output_root", str(input_dir)))

            output_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            data = self._collect_all_data(
                p_root, input_dir, output_dir, assets_dir, website_data
            )

            builders: dict[str, Callable[[dict], str]] = {
                "index.html": self._page_index,
                "pipeline.html": self._page_pipeline,
                "gnn_files.html": self._page_gnn_files,
                "analysis.html": self._page_analysis,
                "visualization.html": self._page_visualization,
                "reports.html": self._page_reports,
                "mcp.html": self._page_mcp,
            }
            for filename, build_page in builders.items():
                try:
                    page_html = build_page(data)
                except Exception as e:
                    result["errors"].append(f"Failed to render {filename}: {e}")
                    continue
                try:
                    _write_atomic(output_dir / filename, page_html)
                    result["pages_created"] += 1
                    result["pages"].append(filename)
                except Exception as e:
                    result["errors"].append(f"Failed to write {filename}: {e}")

        except Exception as e:
            result["errors"].append(str(e))

        result["success"] = not result["errors"]
        return result

    def create_pages(self, output_dir: Path, data: dict) -> dict:
        """Create individual website pages (compatibility API)."""
        return self.generate_website({**data, "output_dir": str(output_dir)})

    # ── Data collection ─────────────────────────────────────────────────────

    def _collect_all_data(
        self,
        p_root: Path,
        input_dir: Path,
        output_dir: Path,
        assets_dir: Path,
        user_data: dict,
    ) -> dict:
        """Collect all data (delegates to the pure ``collect_website_data``)."""
        return collect_website_data(
            p_root,
            input_dir,
            assets_dir,
            output_dir=output_dir,
            user_data=user_data,
        )

    # ── Page generators ─────────────────────────────────────────────────────

    def _page_index(self, data: dict) -> str:
        """Render the dashboard landing page."""
        n_ok = sum(1 for s in data["step_statuses"].values() if s == "ok")
        n_steps = len(PIPELINE_STEPS)
        n_files = data["processed_files"]
        mcp_summary = data.get("mcp_summary")
        n_tools = _stat_int(
            mcp_summary.get("tools_registered", 0)
            if isinstance(mcp_summary, dict)
            else 0
        )

        stats = f"""
<div class="stats-row">
  <div class="stat-card success">
    <div class="label">Steps Complete</div>
    <div class="value">{n_ok}</div>
    <div class="sub">of {n_steps} pipeline steps</div>
  </div>
  <div class="stat-card accent">
    <div class="label">GNN Files</div>
    <div class="value">{n_files}</div>
    <div class="sub">source models</div>
  </div>
  <div class="stat-card accent2">
    <div class="label">MCP Tools</div>
    <div class="value">{n_tools}</div>
    <div class="sub">registered tools</div>
  </div>
  <div class="stat-card accent">
    <div class="label">Visualizations</div>
    <div class="value">{len(data["visualizations"])}</div>
    <div class="sub">generated artifacts</div>
  </div>
</div>"""

        # Step grid
        cards = ""
        for step in PIPELINE_STEPS:
            status = data["step_statuses"].get(step.number, "pending")
            badge_cls = _BADGE_CLASS.get(status, "badge-pending")
            badge_label = _STEP_BADGE_LABEL.get(status, "○ Pending")
            cards += f"""
<div class="step-card">
  <div class="step-num">STEP {step.number:02d}</div>
  <div class="step-name">{_esc(step.name)}</div>
  <div class="step-desc">{_esc(step.description)}</div>
  <span class="step-badge {badge_cls}">{badge_label}</span>
</div>"""

        body = f"""
<div class="page-header">
  <h1>GNN Pipeline Dashboard</h1>
  <p class="subtitle">Results overview — generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</div>
{stats}
<div class="section">
  <div class="section-title">Pipeline Steps</div>
  <div class="step-grid">{cards}</div>
</div>"""
        return _page("Dashboard", "index", body)

    def _page_pipeline(self, data: dict) -> str:
        """Render the full 25-step pipeline status table."""
        rows = ""
        for step in PIPELINE_STEPS:
            status = data["step_statuses"].get(step.number, "pending")
            badge_cls = _BADGE_CLASS.get(status, "badge-pending")
            badge_label = _PIPELINE_BADGE_LABEL.get(status, "Pending")
            rows += f"""<tr>
  <td><code>{step.number:02d}</code></td>
  <td>{_esc(step.name)}</td>
  <td>{_esc(step.description)}</td>
  <td><span class="step-badge {badge_cls}">{badge_label}</span></td>
  <td><code style="font-size:11px;color:var(--text-3)">{_esc(step.script_name)}</code></td>
</tr>"""
        body = f"""
<div class="page-header">
  <h1>⚡ Pipeline Steps</h1>
  <p class="subtitle">Full 25-step GNN processing pipeline status</p>
</div>
<div class="table-wrap">
  <table>
    <thead><tr><th>#</th><th>Step</th><th>Description</th><th>Status</th><th>Script</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""
        return _page("Pipeline", "pipeline", body)

    def _page_gnn_files(self, data: dict) -> str:
        """Render the GNN source file browser."""
        if not data["gnn_files"]:
            content = '<div class="card"><p>No GNN source files found.</p></div>'
        else:
            content = ""
            for gf in data["gnn_files"]:
                try:
                    src = _truncate(
                        gf.read_text(encoding="utf-8", errors="replace"), 3000
                    )
                    size = gf.stat().st_size
                except Exception:
                    src, size = "(could not read)", 0
                content += f"""
<details>
  <summary>{_esc(gf.name)} <span class="pill badge-pending" style="margin-left:8px">{size} bytes</span></summary>
  <div class="details-body"><pre>{_esc(src)}</pre></div>
</details>"""
        body = f"""
<div class="page-header">
  <h1>📂 GNN Source Files</h1>
  <p class="subtitle">{len(data["gnn_files"])} models discovered</p>
</div>
<div class="section">{content}</div>"""
        return _page("GNN Files", "gnn_files", body)

    def _page_analysis(self, data: dict) -> str:
        """Render the analysis metrics page."""
        if not data["analysis"]:
            inner = '<div class="card"><p>No analysis results found. Run step 16 (Analysis) to generate results.</p></div>'
        else:
            inner = ""
            for item in data["analysis"]:
                name = item.get("file_name") or item.get("name", "Result")
                stats_html = ""
                for k, v in item.items():
                    if k in ("file_name", "name"):
                        continue
                    stats_html += (
                        f"<tr><td><code>{_esc(k)}</code></td><td>{_esc(v)}</td></tr>"
                    )
                inner += f"""
<div class="card">
  <h3>{_esc(name)}</h3>
  <div class="table-wrap" style="margin-top:8px">
    <table><tbody>{stats_html}</tbody></table>
  </div>
</div>"""
        body = f"""
<div class="page-header">
  <h1>📊 Analysis</h1>
  <p class="subtitle">Statistical analysis and complexity metrics</p>
</div>
<div class="section">{inner}</div>"""
        return _page("Analysis", "analysis", body)

    def _page_visualization(self, data: dict) -> str:
        """Render the visualization gallery."""
        gui_card = ""
        if data.get("gui_navigation"):
            gui_card = """
<div class="card" style="margin-bottom:24px">
  <div class="viz-title" style="color:var(--accent-2)">Interactive GUI navigation</div>
  <p style="margin:8px 0 0">Interactive model editors and artifacts generated by Step 22.</p>
  <p style="margin:12px 0 0"><a href="../22_gui_output/navigation.html" target="_blank" style="color:var(--accent-2)">Open GUI navigation →</a></p>
</div>"""
        if not data["visualizations"]:
            inner = '<div class="card"><p>No visualizations found. Run steps 8–9 to generate visualizations.</p></div>'
        else:
            cards = ""
            for v in data["visualizations"]:
                title = _esc(v["title"])
                path = _esc(v["path"])
                kind = _esc(str(v.get("type", "image")).title())
                if v["type"] == "image":
                    cards += f"""
<div class="viz-card">
  <img src="assets/{path}" alt="{title}" loading="lazy">
  <div class="viz-info">
    <div class="viz-title">{title}</div>
    <div class="viz-desc">{kind} artifact</div>
  </div>
</div>"""
                else:
                    cards += f"""
<div class="viz-card">
  <div style="padding:16px;background:var(--bg-surface);text-align:center">
    <a href="assets/{path}" target="_blank" style="color:var(--accent-2);font-size:13px">🔗 Open interactive: {title}</a>
  </div>
  <div class="viz-info">
    <div class="viz-title">{title}</div>
    <div class="viz-desc">Interactive HTML visualization</div>
  </div>
</div>"""
            inner = f'<div class="viz-grid">{cards}</div>'
        body = f"""
<div class="page-header">
  <h1>🖼️ Visualizations</h1>
  <p class="subtitle">{len(data["visualizations"])} artifacts generated</p>
</div>
<div class="section">{gui_card}{inner}</div>"""
        return _page("Visualizations", "visualization", body)

    def _page_reports(self, data: dict) -> str:
        """Render the report artifact viewer."""
        if not data["reports"]:
            inner = '<div class="card"><p>No report artifacts found in pipeline output directories.</p></div>'
        else:
            inner = ""
            for rep in data["reports"]:
                try:
                    parsed = json.loads(rep["content"])
                    pretty = _truncate(json.dumps(parsed, indent=2), 1500)
                except Exception:
                    pretty = _truncate(rep["content"], 1500)
                name = _esc(rep["name"])
                origin = _esc(rep["dir"])
                size = _esc(rep["size"])
                inner += f"""
<details>
  <summary>{name} <span style="color:var(--text-3);font-size:11px;margin-left:8px">{origin} · {size} bytes</span></summary>
  <div class="details-body"><pre>{_esc(pretty)}</pre></div>
</details>"""
        body = f"""
<div class="page-header">
  <h1>📋 Reports</h1>
  <p class="subtitle">{len(data["reports"])} report artifacts collected from pipeline output</p>
</div>
<div class="section">{inner}</div>"""
        return _page("Reports", "reports", body)

    def _page_mcp(self, data: dict) -> str:
        """Render the MCP registry from the step-21 processing artifacts."""
        summary: dict[str, Any] = data.get("mcp_summary") or {}
        if not summary:
            # Truthful empty state: step 21 produced nothing we can report.
            inner = (
                '<div class="card"><p>No MCP data recorded.</p>'
                "<p>Step 21 (MCP Processing) did not run or wrote no MCP output "
                "for this pipeline run.</p></div>"
            )
            n_tools = 0
        else:
            n_tools = int(summary.get("tools_registered", 0) or 0)
            status = str(summary.get("processing_status", "unknown"))
            failed = status == "failed"
            badge_cls = "badge-error" if failed else "badge-ok"
            message = str(summary.get("message", ""))
            timestamp = str(summary.get("timestamp", ""))
            mcp_version = str(summary.get("mcp_version", ""))
            n_modules = int(summary.get("registered_modules_count", 0) or 0)
            n_resources = int(summary.get("resources_count", 0) or 0)

            rows = (
                f"<tr><td>Status</td><td>"
                f'<span class="pill {badge_cls}">{_esc(status)}</span></td></tr>'
                f"<tr><td>Timestamp</td><td>{_esc(timestamp)}</td></tr>"
                f"<tr><td>MCP version</td><td>{_esc(mcp_version)}</td></tr>"
                f"<tr><td>Registered modules</td><td>{n_modules}</td></tr>"
                f"<tr><td>Resources</td><td>{n_resources}</td></tr>"
            )
            if failed and summary.get("error"):
                rows += (
                    "<tr><td>Error</td>"
                    '<td style="color:var(--error)">'
                    f"{_esc(str(summary['error']))}</td></tr>"
                )

            chips = "".join(
                f'<span class="pill badge-pending" style="margin-left:6px">'
                f"{_esc(str(m))}</span>"
                for m in (summary.get("registered_modules") or [])
            )
            chips_html = f'<div style="margin:10px 0">{chips}</div>' if chips else ""

            inner = f"""
<div class="card">
  <h3>MCP Processing Summary</h3>
  <p>{_esc(message)}</p>
  <div class="table-wrap" style="margin-top:8px">
    <table><tbody>{rows}</tbody></table>
  </div>
  {chips_html}
</div>"""

            tools: list[dict[str, Any]] = data["mcp_tools"]
            if tools:
                by_mod: dict[str, list[dict[str, Any]]] = {}
                for t in sorted(tools, key=lambda x: (x.get("module", ""), x["name"])):
                    mod = t.get("module") or "core"
                    by_mod.setdefault(mod, []).append(t)

                for mod, mod_tools in sorted(by_mod.items()):
                    cards_html = ""
                    for t in mod_tools:
                        desc = t.get("desc") or ""
                        cat = t.get("category") or ""
                        cat_html = f" · {_esc(cat)}" if cat else ""
                        desc_html = (
                            f'<div class="tool-desc">{_esc(desc)}</div>' if desc else ""
                        )
                        cards_html += f"""
<div class="tool-card">
  <div class="tool-name">{_esc(t["name"])}</div>
  <div class="tool-mod">{_esc(mod)}{cat_html}</div>
  {desc_html}
</div>"""
                    inner += f"""
<div class="section">
  <div class="section-title">{_esc(mod)} <span class="pill badge-pending" style="margin-left:6px">{len(mod_tools)}</span></div>
  {cards_html}
</div>"""

        body = f"""
<div class="page-header">
  <h1>🔧 MCP Tools Registry</h1>
  <p class="subtitle">{n_tools} tools registered across all modules via the Model Context Protocol</p>
</div>
{inner}"""
        return _page("MCP Tools", "mcp", body)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────


def generate_website(
    logger: logging.Logger,
    input_dir: Path,
    output_dir: Path,
    *,
    pipeline_output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate a premium website from GNN pipeline artifacts."""
    try:
        generator = WebsiteGenerator()
        p_root = pipeline_output_root if pipeline_output_root else output_dir.parent
        website_data: dict[str, Any] = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "pipeline_output_root": str(p_root),
        }
        if not input_dir.exists():
            return {
                "success": False,
                "pages_created": 0,
                "errors": [f"Input directory not found: {input_dir}"],
                "warnings": [],
            }
        result = generator.generate_website(website_data)
        if result["success"]:
            logger.info(
                f"Website generated: {result['pages_created']} pages → {output_dir}"
            )
        else:
            for e in result["errors"]:
                logger.error(f"Website error: {e}")
        return result
    except Exception as e:
        logger.error(f"Website generation failed: {e}")
        return {
            "success": False,
            "pages_created": 0,
            "errors": [str(e)],
            "warnings": [],
        }
