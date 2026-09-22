# GUI 3 — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

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
