# SAPF Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/sapf/`.

## Purpose
- Validate real SAPF public-entry behavior and audio processor wiring.
- Keep tests aligned with `src/gnn/sapf/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/sapf/ -q`.
