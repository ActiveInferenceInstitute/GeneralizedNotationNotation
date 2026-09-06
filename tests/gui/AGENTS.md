# GUI Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/gui/`.

## Purpose
- Validate real GUI processors, Oxdraw integration, and headless-safe functionality.
- Keep tests aligned with `src/gnn/gui/AGENTS.md` and `src/gnn/gui/README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/gui/ -q`.
