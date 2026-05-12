"""
Shared visual theme for GNN visualization modules (Steps 8 & 9).

Provides a single source of truth for:
- Active Inference variable-type color palettes
- Edge / connection-type styling
- Matplotlib figure defaults (DPI, figsize, font sizes)
- Colormap presets for matrices, correlations, and transitions

Usage::

    from visualization.theme import VAR_TYPE_COLORS, EDGE_STYLES, FIGURE_DEFAULTS
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Variable-type colour palette (harmonised hex, colour-blind-safe leaning)
# ---------------------------------------------------------------------------
VAR_TYPE_COLORS: Dict[str, str] = {
    "hidden_state":       "#5B9BD5",   # steel blue
    "observation":        "#70C1B3",   # sea green
    "action":             "#F2C14E",   # amber gold
    "policy":             "#E07A5F",   # terracotta
    "prior_vector":       "#A78BCA",   # soft violet
    "likelihood_matrix":  "#F4845F",   # warm coral
    "transition_matrix":  "#48BFE3",   # bright cyan
    "preference_vector":  "#81B29A",   # sage
    "free_energy":        "#D4A5A5",   # dusty rose
    "unknown":            "#B0B0B0",   # neutral gray
}

# 3-D scatter palette (same semantics, distinct enough for mpl 3-D)
VAR_TYPE_COLORS_3D: Dict[str, str] = {
    "hidden_state":       "#FECA57",   # bright yellow
    "observation":        "#FF9FF3",   # pink
    "action":             "#DCE9BE",   # pale lime
    "policy":             "#A8E6CF",   # mint
    "prior_vector":       "#96CEB4",   # teal
    "likelihood_matrix":  "#FF6B6B",   # red
    "transition_matrix":  "#4ECDC4",   # teal-cyan
    "preference_vector":  "#45B7D1",   # sky blue
    "free_energy":        "#D4A5A5",   # dusty rose
    "unknown":            "#CCCCCC",   # light gray
}

# ---------------------------------------------------------------------------
# Edge / connection-type styles
# ---------------------------------------------------------------------------
EDGE_STYLES: Dict[str, Dict[str, Any]] = {
    "state_transition":       {"color": "#3B5998", "width": 3,   "alpha": 0.8, "style": "solid"},
    "observation_generation": {"color": "#70C1B3", "width": 2,   "alpha": 0.7, "style": "dashed"},
    "state_action_influence":  {"color": "#F2C14E", "width": 2,   "alpha": 0.7, "style": "dotted"},
    "action_effect":           {"color": "#E07A5F", "width": 3,   "alpha": 0.8, "style": "solid"},
    "policy_selection":        {"color": "#A78BCA", "width": 2,   "alpha": 0.7, "style": "solid"},
    "prior_influence":         {"color": "#48BFE3", "width": 2,   "alpha": 0.6, "style": "dashed"},
    "likelihood_influence":    {"color": "#F4845F", "width": 2,   "alpha": 0.6, "style": "dotted"},
    "energy_flow":             {"color": "#D4A5A5", "width": 1.5, "alpha": 0.5, "style": "dashed"},
    "preference_energy":       {"color": "#81B29A", "width": 2,   "alpha": 0.7, "style": "solid"},
    "habit_policy":            {"color": "#FFB6C1", "width": 2,   "alpha": 0.7, "style": "solid"},
    "state_observation":       {"color": "#70C1B3", "width": 2,   "alpha": 0.7, "style": "solid"},
    "state_transition_matrix": {"color": "#48BFE3", "width": 2,   "alpha": 0.7, "style": "solid"},
    "policy_action":           {"color": "#A78BCA", "width": 2,   "alpha": 0.7, "style": "solid"},
    "generic_causal":          {"color": "#999999", "width": 1,   "alpha": 0.5, "style": "solid"},
}

# ---------------------------------------------------------------------------
# Generative-model node colours (for the POMDP diagram in combined_analysis)
# ---------------------------------------------------------------------------
GENERATIVE_MODEL_COLORS: Dict[str, str] = {
    "D":  "#98D8C8",  # prior
    "s":  "#7EC8E3",  # hidden state
    "s'": "#7EC8E3",
    "A":  "#F7DC6F",  # likelihood
    "o":  "#82E0AA",  # observation
    "B":  "#F1948A",  # transition
    "C":  "#C39BD3",  # preferences
    "E":  "#F5B7B1",  # habit
    "π":  "#FAD7A0",  # policy
    "G":  "#D2B4DE",  # expected free energy
    "u":  "#ABEBC6",  # action
}

# ---------------------------------------------------------------------------
# Matplotlib figure defaults
# ---------------------------------------------------------------------------
FIGURE_DEFAULTS: Dict[str, Any] = {
    "dpi": 300,
    "figsize": (10, 8),
    "figsize_wide": (14, 12),
    "figsize_combined": (15, 12),
    "figsize_small": (8, 6),
    "bbox_inches": "tight",
    "font_family": "sans-serif",
    "title_fontsize": 16,
    "subtitle_fontsize": 14,
    "label_fontsize": 12,
    "tick_fontsize": 10,
    "annotation_fontsize": 8,
    "legend_fontsize": 10,
}

# Maximum figure dimension (inches) to prevent RendererAgg pixel overflow.
MAX_FIGURE_DIMENSION = 200

# ---------------------------------------------------------------------------
# Colormap presets
# ---------------------------------------------------------------------------
COLORMAP_PRESETS: Dict[str, str] = {
    "heatmap":      "viridis",
    "transition":   "Blues",
    "correlation":  "coolwarm",
    "diverging":    "RdBu_r",
    "statistical":  "Set3",
    "categorical":  "tab10",
}

# Annotation cell limit — suppress cell-value text beyond this count.
ANNOTATION_CELL_LIMIT = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_var_type_color(var_type: str, palette_3d: bool = False) -> str:
    """Return hex colour for a variable type, falling back to ``unknown``."""
    palette = VAR_TYPE_COLORS_3D if palette_3d else VAR_TYPE_COLORS
    return palette.get(var_type, palette["unknown"])


def get_edge_style(connection_type: str) -> Dict[str, Any]:
    """Return edge-rendering kwargs for *connection_type*."""
    return EDGE_STYLES.get(connection_type, EDGE_STYLES["generic_causal"])
