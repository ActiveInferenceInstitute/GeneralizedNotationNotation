# Processing Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/processing/`.

## Purpose
- Validate the public processing surface: package exports, lightweight parsing,
  structural checking, and GNN file discovery.
- Keep tests aligned with `src/gnn/processing/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Scope
`mcp.py` exposes the MCP tool surface and is out of scope for these minimal tests.

## Verification
Run `uv run --extra dev python -m pytest tests/processing/ -q`.
