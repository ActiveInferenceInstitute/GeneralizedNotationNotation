#!/usr/bin/env python3
"""Interactive HTML dashboard generator for RxInfer GIF animations.

Scans a directory for *_100steps.gif files and generates a single
self-contained HTML page (plain vanilla JS, no external assets) with:

* a model-category filter,
* a state-space-size filter bucketed by each GIF's ``.manifest.json``
  ``num_states`` (<=4, 5-16, 17-64, 65+ states),
* a side-by-side compare mode (two dropdowns showing any two models'
  GIFs adjacent with their manifest stats),

in the house neutral dark-gray/black style with WCAG-adequate contrast
and visible focus states on interactive controls.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ordered state-space-size buckets (roadmap A5). "unknown" is appended for
# GIFs whose manifest is missing or carries no usable num_states.
SIZE_BUCKET_LABELS: tuple[str, ...] = (
    "≤4 states",
    "5-16 states",
    "17-64 states",
    "65+ states",
)
UNKNOWN_SIZE_BUCKET = "unknown size"


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


def _size_bucket(num_states: Any) -> str:
    """Bucket a manifest ``num_states`` value into a state-space-size label."""
    if isinstance(num_states, bool) or not isinstance(num_states, (int, float)):
        return UNKNOWN_SIZE_BUCKET
    n = int(num_states)
    if n <= 0:
        return UNKNOWN_SIZE_BUCKET
    if n <= 4:
        return SIZE_BUCKET_LABELS[0]
    if n <= 16:
        return SIZE_BUCKET_LABELS[1]
    if n <= 64:
        return SIZE_BUCKET_LABELS[2]
    return SIZE_BUCKET_LABELS[3]


def _read_manifest(gif: Path) -> dict[str, Any]:
    """Read a GIF's sidecar ``.manifest.json``; {} when absent or unreadable."""
    manifest_path = gif.with_suffix(".manifest.json")
    if not manifest_path.exists():
        logger.warning("No manifest sidecar for %s", gif.name)
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Unreadable manifest for %s: %s", gif.name, exc)
        return {}
    return manifest if isinstance(manifest, dict) else {}


# Plain (non f-string) CSS so braces need no doubling. Neutral dark-gray /
# black house style — NOT dark blue — with WCAG-adequate contrast and
# visible focus states on the interactive controls.
_STYLE = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #121212; color: #e6e6e6; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; margin: 20px 0; color: #f0f0f0; font-size: 1.6em; }
h2 { color: #f0f0f0; font-size: 1.2em; margin: 30px 0 10px; }
.controls { display: flex; justify-content: center; align-items: center; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
label { color: #d0d0d0; }
select { padding: 8px 16px; border: 1px solid #555; border-radius: 6px; font-size: 1em; background: #1c1c1c; color: #e6e6e6; }
select:focus, select:focus-visible { outline: 2px solid #f0f0f0; outline-offset: 2px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }
.card { background: #1c1c1c; border: 1px solid #333; border-radius: 10px; padding: 15px; }
.card img { width: 100%; border-radius: 6px; background: #fff; }
.card-title { font-size: 1em; font-weight: bold; margin-bottom: 8px; color: #f0f0f0; }
.card-meta { font-size: 0.8em; color: #b8b8b8; margin-bottom: 8px; }
.category-header { font-size: 1.3em; font-weight: bold; color: #f0f0f0; margin: 30px 0 10px; border-bottom: 2px solid #333; padding-bottom: 5px; }
.hidden { display: none; }
.compare { border: 1px solid #333; border-radius: 10px; padding: 15px; margin-top: 25px; background: #181818; }
.compare-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px; }
.compare-item { background: #1c1c1c; border: 1px solid #333; border-radius: 10px; padding: 15px; }
.compare-item img { width: 100%; border-radius: 6px; background: #fff; }
.compare-item figcaption { font-weight: bold; color: #f0f0f0; margin-bottom: 8px; }
.compare-item table { width: 100%; margin-top: 8px; border-collapse: collapse; font-size: 0.8em; }
.compare-item th { text-align: left; color: #b8b8b8; font-weight: normal; padding: 2px 8px 2px 0; }
.compare-item td { color: #e6e6e6; padding: 2px 0; }
.footer { text-align: center; color: #a0a0a0; margin-top: 30px; font-size: 0.8em; }
"""

# Plain (non f-string) vanilla JS; MODELS is injected above it.
_SCRIPT = """
function applyFilter() {
    const cat = document.getElementById('filter').value;
    const size = document.getElementById('size-filter').value;
    document.querySelectorAll('.card').forEach(card => {
        const catOk = cat === 'all' || card.dataset.category === cat;
        const sizeOk = size === 'all' || card.dataset.sizeBucket === size;
        card.classList.toggle('hidden', !(catOk && sizeOk));
    });
    document.querySelectorAll('.category').forEach(section => {
        const anyVisible = section.querySelectorAll('.card:not(.hidden)').length > 0;
        section.classList.toggle('hidden', !anyVisible);
    });
}
function compareSide(name) {
    const model = MODELS[name];
    const rows = Object.entries(model.stats).map(
        ([key, value]) =>
            '<tr><th>' + key + '</th><td>' +
            (value === null || value === undefined ? '\\u2014' : value) +
            '</td></tr>'
    ).join('');
    return '<figure class="compare-item">' +
        '<figcaption>' + name + ' \\u2014 ' + model.category + ', ' + model.bucket + '</figcaption>' +
        '<img src="' + model.gif + '" alt="' + name + '" loading="lazy">' +
        '<table>' + rows + '</table>' +
        '</figure>';
}
function updateCompare() {
    const a = document.getElementById('compare-a').value;
    const b = document.getElementById('compare-b').value;
    const panel = document.getElementById('compare-panel');
    if (!a || !b) {
        panel.classList.add('hidden');
        panel.innerHTML = '';
        return;
    }
    panel.innerHTML = compareSide(a) + compareSide(b);
    panel.classList.remove('hidden');
}
"""

# Manifest keys surfaced as compare-mode stats, in display order.
_COMPARE_STAT_KEYS = (
    "num_states",
    "timesteps",
    "belief_accuracy",
    "inference_converged",
    "model_kind",
    "seed",
)


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

    # One record per GIF: name, category, size bucket, manifest stats.
    records: list[dict[str, Any]] = []
    for gif in gifs:
        manifest = _read_manifest(gif)
        records.append(
            {
                "stem": gif.stem.replace("_100steps", ""),
                "gif": gif.name,
                "category": _categorize_gif(gif.name),
                "bucket": _size_bucket(manifest.get("num_states")),
                "manifest": manifest,
            }
        )

    categories: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        categories.setdefault(record["category"], []).append(record)

    present_buckets = [
        bucket
        for bucket in (*SIZE_BUCKET_LABELS, UNKNOWN_SIZE_BUCKET)
        if any(record["bucket"] == bucket for record in records)
    ]

    # Compare-mode model data, embedded as JSON for the vanilla JS.
    models_json = json.dumps(
        {
            record["stem"]: {
                "gif": record["gif"],
                "category": record["category"],
                "bucket": record["bucket"],
                "stats": {
                    key: record["manifest"].get(key) for key in _COMPARE_STAT_KEYS
                },
            }
            for record in records
        },
        sort_keys=True,
    )

    html_parts = [
        f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<div class="controls">
<label for="filter">Category:</label>
<select id="filter" onchange="applyFilter()">
<option value="all">All Models</option>"""
    ]

    for cat in sorted(categories.keys()):
        html_parts.append(f'<option value="{cat}">{cat}</option>')

    html_parts.append(
        """</select>
<label for="size-filter">State space:</label>
<select id="size-filter" onchange="applyFilter()">
<option value="all">All Sizes</option>"""
    )

    for bucket in present_buckets:
        html_parts.append(f'<option value="{bucket}">{bucket}</option>')

    html_parts.append(
        """</select>
</div>
<div class="compare">
<h2>Compare models side-by-side</h2>
<div class="controls">
<label for="compare-a">Model A:</label>
<select id="compare-a" onchange="updateCompare()">
<option value="">(none)</option>"""
    )
    for record in records:
        html_parts.append(f'<option value="{record["stem"]}">{record["stem"]}</option>')
    html_parts.append(
        """</select>
<label for="compare-b">Model B:</label>
<select id="compare-b" onchange="updateCompare()">
<option value="">(none)</option>"""
    )
    for record in records:
        html_parts.append(f'<option value="{record["stem"]}">{record["stem"]}</option>')
    html_parts.append(
        """</select>
</div>
<div id="compare-panel" class="compare-panel hidden"></div>
</div>
"""
    )

    for cat, cat_records in sorted(categories.items()):
        html_parts.append(
            f'<div class="category" data-category="{cat}">\n'
            f'<div class="category-header">{cat} ({len(cat_records)})</div>\n'
            '<div class="gallery">\n'
        )
        for record in cat_records:
            manifest = record["manifest"]
            states = manifest.get("num_states", "?")
            steps = manifest.get("timesteps", "?")
            acc = manifest.get("belief_accuracy", "?")
            meta_text = f"{states} states, {steps} steps, acc={acc}" if manifest else ""

            html_parts.append(
                f'<div class="card" data-category="{cat}" '
                f'data-model="{record["stem"]}" '
                f'data-size-bucket="{record["bucket"]}">\n'
                f'<div class="card-title">{record["stem"]}</div>\n'
                f'<div class="card-meta">{meta_text}</div>\n'
                f'<img src="{record["gif"]}" alt="{record["stem"]}" loading="lazy">\n'
                "</div>\n"
            )
        html_parts.append("</div>\n</div>\n")

    html_parts.append(
        f"""
<div class="footer">
Generated from RxInfer.jl simulations — real @model + infer() with free_energy=true.
Offline batch inference (Bayesian smoothing) with post-hoc EFE policy evaluation.
</div>
</div>
<script>
const MODELS = {models_json};
{_SCRIPT}
</script>
</body>
</html>"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info("Generated dashboard: %s with %d GIFs", output_path, len(gifs))
    return str(output_path)


__all__: list[Any] = [
    "generate_dashboard",
    "SIZE_BUCKET_LABELS",
    "UNKNOWN_SIZE_BUCKET",
]
