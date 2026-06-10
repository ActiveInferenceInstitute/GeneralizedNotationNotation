# Audio Tests Agent

## Overview
This directory owns pytest coverage for `src/audio/`.

## Purpose
- Validate real audio processing, SAPF integration, generation utilities, and graceful dependency behavior.
- Keep tests aligned with `src/audio/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/audio/ -q`.
