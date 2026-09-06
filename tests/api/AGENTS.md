# API Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/api/`.

## Purpose
- Validate real FastAPI startup, endpoint, and integration behavior when dependencies are available.
- Keep tests aligned with `src/gnn/api/AGENTS.md` and `src/gnn/api/README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/api/ -q`.
