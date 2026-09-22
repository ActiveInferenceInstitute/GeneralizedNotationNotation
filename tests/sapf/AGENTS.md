# SAPF Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/audio/sapf/`.

## Purpose
- Validate real SAPF module behavior and audio processor wiring.
- Keep tests aligned with `src/gnn/audio/sapf/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/sapf/ -q`.
