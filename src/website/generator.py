#!/usr/bin/env python3
from __future__ import annotations

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

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    nav_items = [
        ("🏠", "Dashboard",      "index.html",          "index"),
        ("⚡", "Pipeline",       "pipeline.html",        "pipeline"),
        ("📂", "GNN Files",      "gnn_files.html",       "gnn_files"),
        ("📊", "Analysis",       "analysis.html",        "analysis"),
        ("🖼️", "Visualizations", "visualization.html",   "visualization"),
        ("📋", "Reports",        "reports.html",         "reports"),
        ("🔧", "MCP Tools",      "mcp.html",             "mcp"),
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

_PIPELINE_STEPS = [
    (0,  "Template",            "Pipeline template and initialization"),
    (1,  "Setup",               "Environment setup and dependency install"),
    (2,  "Tests",               "Test suite execution (pytest)"),
    (3,  "GNN Processing",      "GNN file discovery, parsing, and multi-format serialization"),
    (4,  "Model Registry",      "Model versioning and registry management"),
    (5,  "Type Checking",       "GNN type validation and resource estimation"),
    (6,  "Validation",          "Consistency and semantic quality checking"),
    (7,  "Export",              "Multi-format export (JSON, XML, GraphML, GEXF, Pickle)"),
    (8,  "Visualization",       "Graph and matrix visualization generation"),
    (9,  "Advanced Viz",        "Interactive and advanced visualization (Plotly, D3)"),
    (10, "Ontology",            "Active Inference ontology processing and validation"),
    (11, "Rendering",           "Code generation for simulation frameworks"),
    (12, "Execution",           "Execute rendered simulation scripts"),
    (13, "LLM",                 "LLM-enhanced analysis and model interpretation"),
    (14, "ML Integration",      "Machine learning integration and model training"),
    (15, "Audio",               "Audio sonification generation (SAPF)"),
    (16, "Analysis",            "Statistical analysis and cross-simulation aggregation"),
    (17, "Integration",         "System integration and cross-module coordination"),
    (18, "Security",            "Security validation and generated code scanning"),
    (19, "Research",            "Research tools and literature references"),
    (20, "Website",             "Static HTML website generation from pipeline artifacts"),
    (21, "MCP Processing",      "Model Context Protocol processing and tool registration"),
    (22, "GUI",                 "Interactive GNN constructor GUI"),
    (23, "Report",              "Comprehensive analysis report generation"),
    (24, "Intelligent Analysis", "AI-powered pipeline analysis and executive reports"),
]


class WebsiteGenerator:
    """Generates a premium multi-page static HTML website from pipeline artifacts."""

    def __init__(self):
        self.template_dir = Path(__file__).parent / "templates"
        self.static_dir   = Path(__file__).parent / "static"

    # ── Public API ──────────────────────────────────────────────────────────

    def generate_website(self, website_data: dict) -> dict:
        """Generate the complete static website."""
        result = {"success": True, "pages_created": 0, "errors": [], "warnings": []}
        try:
            output_dir = Path(website_data.get("output_dir", "output/20_website_output"))
            input_dir  = Path(website_data.get("input_dir", "output"))
            p_root     = Path(website_data.get("pipeline_output_root", str(input_dir)))

            output_dir.mkdir(parents=True, exist_ok=True)
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Aggregate data from pipeline outputs
            data = self._collect_all_data(p_root, input_dir, output_dir, assets_dir, website_data)

            pages = {
                "index.html":         self._page_index(data),
                "pipeline.html":      self._page_pipeline(data),
                "gnn_files.html":     self._page_gnn_files(data),
                "analysis.html":      self._page_analysis(data),
                "visualization.html": self._page_visualization(data),
                "reports.html":       self._page_reports(data),
                "mcp.html":           self._page_mcp(data),
            }

            for filename, html in pages.items():
                try:
                    import os as _os
                    import tempfile as _tempfile
                    _dest = output_dir / filename
                    with _tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_dir, delete=False) as _tmp:
                        _tmp.write(html)
                    _os.replace(_tmp.name, str(_dest))
                    result["pages_created"] += 1
                except Exception as e:
                    result["errors"].append(f"Failed to write {filename}: {e}")

            # Copy static assets if available
            if self.static_dir.exists():
                shutil.copytree(self.static_dir, output_dir / "static", dirs_exist_ok=True)

        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))

        return result

    def create_pages(self, output_dir: Path, data: dict) -> dict:
        """Create individual website pages (compatibility API)."""
        return self.generate_website({**data, "output_dir": str(output_dir)})

    # ── Data collection ─────────────────────────────────────────────────────

    def _collect_all_data(self, p_root: Path, input_dir: Path,
                          output_dir: Path, assets_dir: Path,
                          user_data: dict) -> dict:
        data: dict = {
            "p_root":        p_root,
            "output_dir":    output_dir,
            "gnn_files":     [],
            "analysis":      [],
            "complexity":    [],
            "visualizations":[],
            "reports":       [],
            "mcp_tools":     [],
            "step_statuses": {},
            "exec_summary":  {},
            "processed_files": 0,
        }

        # Merge any caller-supplied data
        data.update({k: v for k, v in user_data.items()
                     if k not in ("output_dir", "input_dir", "pipeline_output_root")})

        # GNN source files
        for search_dir in [p_root.parent / "input" / "gnn_files", input_dir]:
            if search_dir.exists():
                for f in sorted(search_dir.glob("*.md")):
                    data["gnn_files"].append(f)
                data["processed_files"] = len(data["gnn_files"])
                break

        # Pipeline step statuses from numbered output dirs
        self._collect_step_statuses(p_root, data)

        # Analysis JSON files
        seen = set()
        for candidate in [
            p_root / "16_analysis_output" / "analysis_results",
            p_root / "16_analysis_output",
        ]:
            if candidate.exists():
                for jf in candidate.glob("*.json"):
                    if jf.name not in seen:
                        seen.add(jf.name)
                        try:
                            d = json.loads(jf.read_text())
                            data["analysis"].append(d)
                        except Exception as e:
                            logger.debug(f"Skipped malformed analysis file {jf.name}: {e}")

        # Execution summary
        for ec in [
            p_root / "12_execute_output" / "summaries" / "execution_summary.json",
            p_root / "12_execute_output" / "execution_summary.json",
        ]:
            if ec.exists():
                try:
                    data["exec_summary"] = json.loads(ec.read_text())
                except Exception as e:
                    logger.debug(f"Skipped malformed execution summary file: {e}")
                break

        # Visualizations — copy assets
        viz_dirs = [
            p_root / "08_visualization_output" / "visualization_results",
            p_root / "8_visualization_output"  / "visualization_results",
            p_root / "09_advanced_viz_output",
            p_root / "9_advanced_viz_output",
        ]
        for vd in viz_dirs:
            if not vd.exists():
                continue
            for img in vd.rglob("*.png"):
                dest = assets_dir / img.name
                try:
                    shutil.copy2(img, dest)
                except Exception:
                    dest = img
                data["visualizations"].append({"title": img.stem, "path": dest.name,
                                               "type": "image", "abs": dest})
            for html_f in vd.rglob("*.html"):
                dest = assets_dir / html_f.name
                try:
                    shutil.copy2(html_f, dest)
                except Exception:
                    dest = html_f
                data["visualizations"].append({"title": html_f.stem, "path": dest.name,
                                               "type": "html", "abs": dest})

        # Reports — collect all JSON/txt artifacts from numbered output dirs
        for d in sorted(p_root.iterdir()) if p_root.exists() else []:
            if not (d.is_dir() and d.name[0].isdigit()):
                continue
            for jf in list(d.rglob("*.json"))[:5]:  # cap per dir
                try:
                    content = jf.read_text(encoding="utf-8", errors="replace")
                    data["reports"].append({
                        "name":    jf.name,
                        "dir":     d.name,
                        "content": content[:2000],
                        "size":    jf.stat().st_size,
                    })
                except Exception as e:
                    logger.debug(f"Skipped unreadable report file {jf.name}: {e}")

        # MCP tools — try to load live
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from mcp.mcp import mcp_instance
            if mcp_instance.tools:
                for name, tool in mcp_instance.tools.items():
                    data["mcp_tools"].append({
                        "name":     name,
                        "module":   getattr(tool, "module", ""),
                        "category": getattr(tool, "category", ""),
                        "desc":     getattr(tool, "description", ""),
                    })
        except Exception as e:
            logger.debug(f"MCP tools not loaded for website (optional): {e}")

        return data

    def _collect_step_statuses(self, p_root: Path, data: dict) -> None:
        """Scan output directories to determine each step's status."""
        if not p_root.exists():
            return
        step_dirs = {d.name: d for d in p_root.iterdir() if d.is_dir()}
        for step_num, _, _ in _PIPELINE_STEPS:
            for prefix in [f"{step_num:02d}_", f"{step_num}_"]:
                matching = [d for n, d in step_dirs.items() if n.startswith(prefix)]
                if matching:
                    data["step_statuses"][step_num] = "ok"
                    break
            else:
                data["step_statuses"][step_num] = "pending"

    # ── Page generators ─────────────────────────────────────────────────────

    def _page_index(self, data: dict) -> str:
        n_ok      = sum(1 for s in data["step_statuses"].values() if s == "ok")
        n_steps   = len(_PIPELINE_STEPS)
        n_files   = data["processed_files"]
        n_tools   = len(data["mcp_tools"])

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
        for step_num, step_name, step_desc in _PIPELINE_STEPS:
            status = data["step_statuses"].get(step_num, "pending")
            badge_cls = {"ok": "badge-ok", "error": "badge-error",
                         "skip": "badge-skip"}.get(status, "badge-pending")
            badge_label = {"ok": "✓ Complete", "error": "✗ Error",
                           "skip": "⊘ Skipped"}.get(status, "○ Pending")
            cards += f"""
<div class="step-card">
  <div class="step-num">STEP {step_num:02d}</div>
  <div class="step-name">{step_name}</div>
  <div class="step-desc">{step_desc}</div>
  <span class="step-badge {badge_cls}">{badge_label}</span>
</div>"""

        body = f"""
<div class="page-header">
  <h1>GNN Pipeline Dashboard</h1>
  <p class="subtitle">Results overview — generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
{stats}
<div class="section">
  <div class="section-title">Pipeline Steps</div>
  <div class="step-grid">{cards}</div>
</div>"""
        return _page("Dashboard", "index", body)

    def _page_pipeline(self, data: dict) -> str:
        rows = ""
        for step_num, step_name, step_desc in _PIPELINE_STEPS:
            status   = data["step_statuses"].get(step_num, "pending")
            badge_cls = {"ok": "badge-ok", "error": "badge-error",
                          "skip": "badge-skip"}.get(status, "badge-pending")
            badge_label = {"ok": "Complete", "error": "Error",
                           "skip": "Skipped"}.get(status, "Pending")
            f"{step_num}_{step_name.lower().replace(' ', '_')}.py"
            rows += f"""<tr>
  <td><code>{step_num:02d}</code></td>
  <td>{step_name}</td>
  <td>{step_desc}</td>
  <td><span class="step-badge {badge_cls}">{badge_label}</span></td>
  <td><code style="font-size:11px;color:var(--text-3)">{step_num}_{step_name.lower().replace(' ','_')}.py</code></td>
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
        if not data["gnn_files"]:
            content = '<div class="card"><p>No GNN source files found.</p></div>'
        else:
            content = ""
            for gf in data["gnn_files"]:
                try:
                    src = gf.read_text(encoding="utf-8", errors="replace")[:3000]
                    escaped = src.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                except Exception:
                    escaped = "(could not read)"
                content += f"""
<details>
  <summary>{gf.name} <span class="pill badge-pending" style="margin-left:8px">{gf.stat().st_size} bytes</span></summary>
  <div class="details-body"><pre>{escaped}</pre></div>
</details>"""
        body = f"""
<div class="page-header">
  <h1>📂 GNN Source Files</h1>
  <p class="subtitle">{len(data["gnn_files"])} models discovered</p>
</div>
<div class="section">{content}</div>"""
        return _page("GNN Files", "gnn_files", body)

    def _page_analysis(self, data: dict) -> str:
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
                    stats_html += f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
                inner += f"""
<div class="card">
  <h3>{name}</h3>
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
        if not data["visualizations"]:
            inner = '<div class="card"><p>No visualizations found. Run steps 8–9 to generate visualizations.</p></div>'
        else:
            cards = ""
            for v in data["visualizations"]:
                if v["type"] == "image":
                    cards += f"""
<div class="viz-card">
  <img src="assets/{v['path']}" alt="{v['title']}" loading="lazy">
  <div class="viz-info">
    <div class="viz-title">{v['title']}</div>
    <div class="viz-desc">{v.get('type','image').title()} artifact</div>
  </div>
</div>"""
                else:
                    cards += f"""
<div class="viz-card">
  <div style="padding:16px;background:var(--bg-surface);text-align:center">
    <a href="assets/{v['path']}" target="_blank" style="color:var(--accent-2);font-size:13px">🔗 Open interactive: {v['title']}</a>
  </div>
  <div class="viz-info">
    <div class="viz-title">{v['title']}</div>
    <div class="viz-desc">Interactive HTML visualization</div>
  </div>
</div>"""
            inner = f'<div class="viz-grid">{cards}</div>'
        body = f"""
<div class="page-header">
  <h1>🖼️ Visualizations</h1>
  <p class="subtitle">{len(data["visualizations"])} artifacts generated</p>
</div>
<div class="section">{inner}</div>"""
        return _page("Visualizations", "visualization", body)

    def _page_reports(self, data: dict) -> str:
        if not data["reports"]:
            inner = '<div class="card"><p>No report artifacts found in pipeline output directories.</p></div>'
        else:
            inner = ""
            for rep in data["reports"]:
                try:
                    parsed = json.loads(rep["content"])
                    pretty = json.dumps(parsed, indent=2)[:1500]
                except Exception:
                    pretty = rep["content"][:1500]
                escaped = pretty.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                inner += f"""
<details>
  <summary>{rep['name']} <span style="color:var(--text-3);font-size:11px;margin-left:8px">{rep['dir']} · {rep['size']} bytes</span></summary>
  <div class="details-body"><pre>{escaped}</pre></div>
</details>"""
        body = f"""
<div class="page-header">
  <h1>📋 Reports</h1>
  <p class="subtitle">{len(data["reports"])} report artifacts collected from pipeline output</p>
</div>
<div class="section">{inner}</div>"""
        return _page("Reports", "reports", body)

    def _page_mcp(self, data: dict) -> str:
        tools = data["mcp_tools"]
        if not tools:
            by_mod_html = '<div class="card"><p>No MCP tools registered. Run step 21 (MCP Processing) to register tools.</p></div>'
        else:
            # Group by module
            by_mod: Dict[str, List[dict]] = {}
            for t in sorted(tools, key=lambda x: (x.get("module",""), x["name"])):
                mod = t.get("module") or "core"
                by_mod.setdefault(mod, []).append(t)

            by_mod_html = ""
            for mod, mod_tools in sorted(by_mod.items()):
                cards_html = ""
                for t in mod_tools:
                    desc = t.get("desc") or ""
                    cat  = t.get("category") or ""
                    cards_html += f"""
<div class="tool-card">
  <div class="tool-name">{t['name']}</div>
  <div class="tool-mod">{mod}{f" · {cat}" if cat else ""}</div>
  {f'<div class="tool-desc">{desc}</div>' if desc else ""}
</div>"""
                by_mod_html += f"""
<div class="section">
  <div class="section-title">{mod} <span class="pill badge-pending" style="margin-left:6px">{len(mod_tools)}</span></div>
  {cards_html}
</div>"""

        body = f"""
<div class="page-header">
  <h1>🔧 MCP Tools Registry</h1>
  <p class="subtitle">{len(tools)} tools registered across all modules via the Model Context Protocol</p>
</div>
{by_mod_html}"""
        return _page("MCP Tools", "mcp", body)


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def generate_website(logger: logging.Logger, input_dir: Path, output_dir: Path, *,
                     pipeline_output_root: Optional[Path] = None) -> Dict[str, Any]:
    """Generate a premium website from GNN pipeline artifacts."""
    try:
        generator = WebsiteGenerator()
        p_root = pipeline_output_root if pipeline_output_root else output_dir.parent
        website_data = {
            "input_dir":            str(input_dir),
            "output_dir":           str(output_dir),
            "pipeline_output_root": str(p_root),
        }
        if not input_dir.exists():
            return {"success": False, "pages_created": 0,
                    "errors": [f"Input directory not found: {input_dir}"], "warnings": []}
        result = generator.generate_website(website_data)
        if result["success"]:
            logger.info(f"Website generated: {result['pages_created']} pages → {output_dir}")
        else:
            for e in result["errors"]:
                logger.error(f"Website error: {e}")
        return result
    except Exception as e:
        logger.error(f"Website generation failed: {e}")
        return {"success": False, "pages_created": 0, "errors": [str(e)], "warnings": []}
