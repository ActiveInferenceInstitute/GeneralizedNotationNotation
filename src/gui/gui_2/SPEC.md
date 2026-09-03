# GUI 2 — Technical Specification

**Version**: 1.6.0

## Purpose

Second-generation GNN constructor GUI: a Gradio web app for visual matrix editing.

## Features

- Interactive DataFrame matrix editing with +/- dimension controls
- Matrix heatmaps and vector bar charts (Plotly when available)
- Drag-and-drop model construction
- Real-time validation feedback
- POMDP template-based initialization

## Technology

- `gradio` (from the `gui` extra in `pyproject.toml`)
- `plotly` (interactive plots; recovery: basic displays)
- `numpy` (matrix handling)

## Architecture

- `ui.py` — Main UI layout and event handling
- `matrix_editor.py` — Matrix editing grid logic
- `ui_simple.py`, `ui_minimal.py` — Reduced-dependency UI variants
