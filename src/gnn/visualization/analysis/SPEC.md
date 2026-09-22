# Visualization Analysis — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Input Sources

- Step 7 exports, Step 8 visualizations, Step 12 execution results, Step 16 analysis

## Output

- Combined HTML dashboard
- Cross-step correlation plots (PNG)

## Aggregation Strategy

- Merges data by model name across steps
- Missing steps produce partial dashboards with warnings
