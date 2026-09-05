"""Shared CSS constants for the HTML emitters.

Only byte-identical (after whitespace normalization) rules are extracted
here — dashboard.py and html_generator.py keep every rule with cosmetic
differences inline. Pure data: no logic, no imports.
"""

__all__ = [
    "BASE_CSS",
    "FONT_STACK",
    "BODY_GRADIENT",
    "HEADER_H2_CSS",
    "PARAMETER_NAME_CSS",
    "STAT_LABEL_CSS",
]

# Universal reset — identical in both emitters.
BASE_CSS = """        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
"""

# Shared font stack (interpolated into each file's `body` rule).
FONT_STACK = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"

# Shared page gradient (interpolated into each file's `body` rule).
BODY_GRADIENT = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"

HEADER_H2_CSS = """        .header h2 {
            color: #7f8c8d;
            font-size: 1.3em;
            font-weight: 300;
        }
"""

PARAMETER_NAME_CSS = """        .parameter-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
"""

STAT_LABEL_CSS = """        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
"""
