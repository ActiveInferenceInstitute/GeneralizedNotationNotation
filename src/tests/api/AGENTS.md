# API Tests Agent

## Overview
This directory owns pytest coverage for `src/api/`.

## Purpose
- Validate real FastAPI startup, endpoint, and integration behavior when dependencies are available.
- Keep tests aligned with `src/api/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/api/ -q`.
