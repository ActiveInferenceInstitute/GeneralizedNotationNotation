# Multimodel Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/multimodel/`.

## Purpose
- Validate the MCP dependency-graph tool wrapper and registration shape.
- Exercise real `gnn.multimodel` rendering behavior, not implementation details.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/multimodel/ -q`.
