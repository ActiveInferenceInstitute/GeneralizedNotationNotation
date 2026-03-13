#!/usr/bin/env python3
"""
Dashboard SPA — Self-contained HTML dashboard for pipeline results.

Generates a single-file HTML dashboard with:
  - Step sidebar with status indicators
  - Pipeline timeline SVG (Gantt-style)
  - Artifact browser
  - Summary statistics

No external dependencies required — CSS/JS inlined.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from html import escape

logger = logging.getLogger(__name__)


def render_dashboard(
    results_dir: Path,
    output_path: Path,
    summary_path: Optional[Path] = None,
) -> bool:
    """
    Render self-contained HTML dashboard from pipeline results.

    Args:
        results_dir: Root output directory with per-step subdirectories.
        output_path: Where to write the dashboard HTML file.
        summary_path: Optional path to pipeline_execution_summary.json.

    Returns:
        True if dashboard was successfully generated.
    """
    results_dir = Path(results_dir)
    summary_path = summary_path or results_dir / "pipeline_execution_summary.json"

    # Load data
    summary = _load_json(summary_path)
    steps = summary.get("steps", [])
    step_dirs = _discover_step_dirs(results_dir)

    # Build components
    sidebar_html = _render_sidebar(steps, step_dirs)
    timeline_svg = _render_timeline_svg(steps)
    detail_html = _render_step_details(steps, step_dirs)
    stats_html = _render_stats(summary, step_dirs)

    # Mermaid Dependency Graph
    mermaid_graph = _render_mermaid_graph(summary)
    graph_section = ""
    if mermaid_graph:
        graph_section = f"""  <div class="section">
    <h2>Model Dependency Graph</h2>
    <div class="mermaid-container" style="background: var(--surface); padding: 20px; border-radius: 8px;">
      <pre class="mermaid">
{mermaid_graph}
      </pre>
    </div>
  </div>"""

    # Assemble full page
    timestamp = summary.get("timestamp", datetime.now().isoformat())
    total_dur = summary.get("total_duration", "N/A")
    success = summary.get("success", None)
    badge = "🟢 SUCCESS" if success is True else "🔴 FAILED" if success is False else "⚪ UNKNOWN"

    html = _TEMPLATE.format(
        title="GNN Pipeline Dashboard",
        timestamp=timestamp,
        badge=badge,
        total_duration=total_dur,
        sidebar=sidebar_html,
        timeline=timeline_svg,
        details=detail_html,
        stats=stats_html,
        graph_section=graph_section,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"🌐 Dashboard written to: {output_path} ({len(html)} bytes)")
    return True


# ─── Internal Helpers ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}

def _render_mermaid_graph(summary: dict) -> str:
    target_dir = summary.get("target_dir")
    if not target_dir:
        return ""
    
    target_path = Path(target_dir)
    if not target_path.exists():
        return ""
        
    try:
        from gnn.dep_graph import render_graph_from_file
        # Find first .gnn file
        gnn_files = list(target_path.glob("*.gnn"))
        if not gnn_files:
            return ""
            
        return render_graph_from_file(str(gnn_files[0]), output_format="mermaid")
    except Exception as e:
        logger.warning(f"Could not render dependency graph to dashboard: {e}")
        return ""


def _discover_step_dirs(results_dir: Path) -> List[Path]:
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name[0].isdigit() and "_output" in d.name
    )


def _status_color(status: str) -> str:
    s = str(status).lower()
    if s in ("success", "passed"):
        return "#22c55e"
    elif "warn" in s:
        return "#f59e0b"
    elif s in ("failed", "error"):
        return "#ef4444"
    return "#6b7280"


def _render_sidebar(steps: List[dict], step_dirs: List[Path]) -> str:
    items = []
    if steps:
        for s in steps:
            name = escape(s.get("name", "?"))
            status = s.get("status", "?")
            color = _status_color(status)
            items.append(
                f'<div class="step-item" onclick="showStep(\'{escape(name)}\')">'
                f'<span class="dot" style="background:{color}"></span>'
                f'<span class="step-name">{name}</span>'
                f'</div>'
            )
    elif step_dirs:
        for d in step_dirs:
            items.append(
                f'<div class="step-item">'
                f'<span class="dot" style="background:#6b7280"></span>'
                f'<span class="step-name">{escape(d.name)}</span>'
                f'</div>'
            )
    return "\n".join(items)


def _render_timeline_svg(steps: List[dict]) -> str:
    if not steps:
        return '<p class="muted">No timing data available.</p>'

    max_dur = max((s.get("duration_seconds", 0) for s in steps), default=1) or 1
    bar_height = 28
    padding = 4
    label_width = 200
    bar_width = 500
    svg_height = len(steps) * (bar_height + padding) + 20

    bars = []
    for i, s in enumerate(steps):
        name = escape(s.get("name", "?"))
        dur = s.get("duration_seconds", 0)
        w = max(2, dur / max_dur * bar_width)
        y = i * (bar_height + padding) + 10
        color = _status_color(s.get("status", "?"))
        bars.append(
            f'<text x="0" y="{y + 18}" font-size="12" fill="#d1d5db">{name}</text>'
            f'<rect x="{label_width}" y="{y}" width="{w}" height="{bar_height}" '
            f'rx="4" fill="{color}" opacity="0.85"/>'
            f'<text x="{label_width + w + 6}" y="{y + 18}" font-size="11" fill="#9ca3af">'
            f'{dur:.1f}s</text>'
        )

    return (
        f'<svg width="{label_width + bar_width + 60}" height="{svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg">\n'
        + "\n".join(bars)
        + "\n</svg>"
    )


def _render_step_details(steps: List[dict], step_dirs: List[Path]) -> str:
    sections = []
    dir_map = {d.name: d for d in step_dirs}

    for s in steps:
        name = escape(s.get("name", "?"))
        status = s.get("status", "?")
        dur = s.get("duration_seconds", 0)
        out_dir = s.get("output_dir", "")
        color = _status_color(status)

        # List artifacts if directory found
        artifacts_html = ""
        d = dir_map.get(out_dir)
        if d and d.exists():
            files = sorted(f.name for f in d.rglob("*") if f.is_file())[:15]
            if files:
                file_list = "".join(f"<li>{escape(f)}</li>" for f in files)
                more = "<li><em>…and more</em></li>" if len(list(d.rglob("*"))) > 15 else ""
                artifacts_html = f"<ul class='artifact-list'>{file_list}{more}</ul>"

        sections.append(
            f'<div class="step-detail" id="step-{escape(name)}">'
            f'<h3><span class="dot" style="background:{color}"></span> {name}</h3>'
            f'<p>Status: <strong>{escape(status)}</strong> · Duration: {dur:.1f}s</p>'
            f'{artifacts_html}'
            f'</div>'
        )

    return "\n".join(sections)


def _render_stats(summary: dict, step_dirs: List[Path]) -> str:
    total_files = sum(
        sum(1 for f in d.rglob("*") if f.is_file()) for d in step_dirs
    )
    total_size = sum(
        sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) for d in step_dirs
    )
    n_steps = len(summary.get("steps", step_dirs))

    return (
        f'<div class="stat-card"><div class="stat-value">{n_steps}</div>'
        f'<div class="stat-label">Steps</div></div>'
        f'<div class="stat-card"><div class="stat-value">{total_files}</div>'
        f'<div class="stat-label">Artifacts</div></div>'
        f'<div class="stat-card"><div class="stat-value">{total_size/1024:.0f} KB</div>'
        f'<div class="stat-label">Total Size</div></div>'
    )


# ─── HTML Template ───────────────────────────────────────────────────────────────

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{
  --bg: #0f172a; --surface: #1e293b; --border: #334155;
  --text: #e2e8f0; --muted: #94a3b8; --accent: #3b82f6;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Inter', system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; min-height: 100vh; }}
.sidebar {{ width: 260px; background: var(--surface); border-right: 1px solid var(--border); padding: 20px; overflow-y: auto; flex-shrink: 0; }}
.sidebar h2 {{ font-size: 14px; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; letter-spacing: 1px; }}
.step-item {{ display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 8px; cursor: pointer; transition: background 0.15s; }}
.step-item:hover {{ background: rgba(59,130,246,0.1); }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.step-name {{ font-size: 13px; }}
.main {{ flex: 1; padding: 32px; overflow-y: auto; }}
.header {{ margin-bottom: 32px; }}
.header h1 {{ font-size: 24px; margin-bottom: 8px; }}
.header .meta {{ color: var(--muted); font-size: 13px; }}
.stats {{ display: flex; gap: 16px; margin-bottom: 32px; }}
.stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; min-width: 120px; }}
.stat-value {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
.stat-label {{ color: var(--muted); font-size: 12px; margin-top: 4px; text-transform: uppercase; }}
.section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
.section h2 {{ font-size: 18px; margin-bottom: 16px; }}
.step-detail {{ padding: 12px 0; border-bottom: 1px solid var(--border); }}
.step-detail:last-child {{ border-bottom: none; }}
.step-detail h3 {{ font-size: 15px; display: flex; align-items: center; gap: 8px; }}
.artifact-list {{ list-style: none; margin-top: 8px; padding-left: 18px; }}
.artifact-list li {{ font-size: 12px; color: var(--muted); padding: 2px 0; }}
.artifact-list li::before {{ content: '📄 '; }}
.muted {{ color: var(--muted); font-style: italic; }}
svg text {{ font-family: 'Inter', system-ui, sans-serif; }}
</style>
</head>
<body>
<div class="sidebar">
  <h2>Pipeline Steps</h2>
  {sidebar}
</div>
<div class="main">
  <div class="header">
    <h1>{title}</h1>
    <p class="meta">{badge} · {timestamp} · Total: {total_duration}s</p>
  </div>
  <div class="stats">{stats}</div>
  <div class="section">
    <h2>Pipeline Timeline</h2>
    {timeline}
  </div>
  <div class="section">
    <h2>Step Details</h2>
    {details}
  </div>
  {graph_section}
</div>
<script>
function showStep(name) {{
  document.querySelectorAll('.step-detail').forEach(el => {{
    el.style.display = el.id === 'step-' + name ? 'block' : 'none';
  }});
}}
</script>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
</script>
</body>
</html>"""
