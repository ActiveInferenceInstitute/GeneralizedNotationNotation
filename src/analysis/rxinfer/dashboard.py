#!/usr/bin/env python3
"""Interactive HTML dashboard generator for RxInfer GIF animations.

Scans a directory for *_100steps.gif files and generates a single
self-contained HTML page with a model selector, grouped by category,
with embedded GIF references and stats.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _categorize_gif(filename: str) -> str:
    """Categorize a GIF filename by model type."""
    name = filename.lower()
    if "scaling" in name:
        return "Scaling Study"
    if "multiagent" in name or "stigmergic" in name or "coordination" in name:
        return "Multi-Agent"
    if "hierarchical" in name or "temporal_hierarchy" in name:
        return "Hierarchical"
    if "continuous" in name or "stochastic" in name or "navigation" in name:
        return "Continuous"
    return "Discrete"


def generate_dashboard(
    gif_dir: Path,
    output_path: Path,
    title: str = "RxInfer Animation Dashboard",
) -> str:
    """Generate an interactive HTML dashboard from GIF files.

    Args:
        gif_dir: Directory containing *_100steps.gif files
        output_path: Where to write the HTML file
        title: Dashboard title

    Returns:
        Path to the generated HTML file, or "" if no GIFs found
    """
    gifs = sorted(gif_dir.glob("*_100steps.gif"))
    if not gifs:
        logger.warning("No GIF files found in %s", gif_dir)
        return ""

    # Group by category
    categories: dict[str, list[Path]] = {}
    for gif in gifs:
        cat = _categorize_gif(gif.name)
        categories.setdefault(cat, []).append(gif)

    # Build HTML
    html_parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; color: #222; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ text-align: center; margin: 20px 0; color: #222; font-size: 1.6em; }}
.controls {{ display: flex; justify-content: center; gap: 12px; margin: 20px 0; flex-wrap: wrap; }}
select {{ padding: 8px 16px; border: 1px solid #ccc; border-radius: 6px; font-size: 1em; background: white; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.card img {{ width: 100%; border-radius: 6px; }}
.card-title {{ font-size: 1em; font-weight: bold; margin-bottom: 8px; color: #333; }}
.card-meta {{ font-size: 0.8em; color: #888; margin-bottom: 8px; }}
.category-header {{ font-size: 1.3em; font-weight: bold; color: #222; margin: 30px 0 10px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
.hidden {{ display: none; }}
.footer {{ text-align: center; color: #999; margin-top: 30px; font-size: 0.8em; }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<div class="controls">
<label for="filter">Filter:</label>
<select id="filter" onchange="applyFilter()">
<option value="all">All Models</option>"""
    ]

    for cat in sorted(categories.keys()):
        html_parts.append(f'<option value="{cat}">{cat}</option>')

    html_parts.append("""</select>
</div>
""")

    for cat, gif_list in sorted(categories.items()):
        html_parts.append(
            f'<div class="category" data-category="{cat}">\n'
            f'<div class="category-header">{cat} ({len(gif_list)})</div>\n'
            '<div class="gallery">\n'
        )
        for gif in gif_list:
            stem = gif.stem.replace("_100steps", "")
            # Try to read sidecar manifest
            manifest_path = gif.with_suffix(".manifest.json")
            meta_text = ""
            if manifest_path.exists():
                import json

                try:
                    manifest = json.loads(manifest_path.read_text())
                    states = manifest.get("num_states", "?")
                    steps = manifest.get("timesteps", "?")
                    acc = manifest.get("belief_accuracy", "?")
                    meta_text = f"{states} states, {steps} steps, acc={acc}"
                except Exception:
                    pass

            html_parts.append(
                f'<div class="card" data-category="{cat}">\n'
                f'<div class="card-title">{stem}</div>\n'
                f'<div class="card-meta">{meta_text}</div>\n'
                f'<img src="{gif.name}" alt="{stem}" loading="lazy">\n'
                "</div>\n"
            )
        html_parts.append("</div>\n</div>\n")

    html_parts.append("""
<div class="footer">
Generated from RxInfer.jl simulations — real @model + infer() with free_energy=true.
Offline batch inference (Bayesian smoothing) with post-hoc EFE policy evaluation.
</div>
</div>
<script>
function applyFilter() {
    const filter = document.getElementById('filter').value;
    const categories = document.querySelectorAll('.category');
    categories.forEach(cat => {
        if (filter === 'all' || cat.dataset.category === filter) {
            cat.classList.remove('hidden');
        } else {
            cat.classList.add('hidden');
        }
    });
}
</script>
</body>
</html>""")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Generated dashboard: %s with %d GIFs", output_path, len(gifs))
    return str(output_path)


__all__: list[Any] = ["generate_dashboard"]
