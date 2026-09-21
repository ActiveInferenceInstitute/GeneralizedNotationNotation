# Extract Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/extract/`.

## Purpose
- Validate the headless POMDP extractor and its MCP surface.
- Keep tests aligned with `src/gnn/extract/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/extract/ -q`.
