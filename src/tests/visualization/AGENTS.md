# Visualization Tests Agent

## Overview
This directory owns pytest coverage for `src/visualization/`.

## Purpose
- Validate real visualization processors, graph/matrix artifacts, Mermaid/D2 conversions, and ontology visualization.
- Keep tests aligned with `src/visualization/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/visualization/ -q`.
