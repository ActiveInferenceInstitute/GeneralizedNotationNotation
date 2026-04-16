# Matrix Visualization — Technical Specification

**Version**: 1.6.0

## Supported Matrix Types

- 2D transition matrices (A)
- 2D observation matrices (B)
- 3D POMDP tensors (per-action slices)
- Prior distributions (D vectors)

## Heatmap Configuration

- Colormap: `viridis` (default), configurable
- Annotations: cell values shown when matrix ≤ 10×10
- Statistical sidebar: mean, std, min, max per row/column

## 3D Tensor Handling

Per-action slice visualization with shared colorbar and cross-slice statistics.

## Output Formats

- PNG (default, 150 DPI)
- SVG (when `--svg` flag set)
- CSV (always, alongside visual output)
