# SAPF Tests Agent

## Overview
This directory owns pytest coverage for `src/sapf/`.

## Purpose
- Validate real SAPF compatibility-shim behavior and audio processor wiring.
- Keep tests aligned with `src/sapf/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/sapf/ -q`.
