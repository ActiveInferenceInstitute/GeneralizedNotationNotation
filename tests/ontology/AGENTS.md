# Ontology Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/ontology/`.

## Purpose
- Validate real ontology processing, annotations, and MCP wrapper behavior.
- Keep tests aligned with `src/gnn/ontology/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/ontology/ -q`.
