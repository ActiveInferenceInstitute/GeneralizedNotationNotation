# GUI 3 — Technical Specification

**Version**: 1.6.0

## Purpose

Third-generation GNN constructor interface: a Gradio-based State Space Design Studio.

## Features

- Browser-based UI (Gradio `Blocks`)
- State-space variable editing with dimension and type controls
- Ontology term mapping and connection topology editing
- Export to GNN markdown with live preview

## Technology

- `gradio` (optional; from the `gui` extra in `pyproject.toml`)
- Headless fallback when Gradio is unavailable
