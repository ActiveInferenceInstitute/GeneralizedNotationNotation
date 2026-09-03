# GUI 1 — Technical Specification

**Version**: 1.6.0

## Purpose

Form-based GNN model constructor GUI served as a Gradio web application.

## Features

- Model structure editor (states, observations, actions)
- Component management with live state-space validation
- Synchronized plaintext GNN markdown editing
- Headless artifact generation when Gradio is unavailable

## Technology

- `gradio` (from the `gui` extra in `pyproject.toml`)
- Cross-platform (Windows, macOS, Linux)

## Input/Output

- Input: Existing GNN files or blank canvas
- Output: GNN model files in markdown format (`constructed_model_gui1.md` by default)
