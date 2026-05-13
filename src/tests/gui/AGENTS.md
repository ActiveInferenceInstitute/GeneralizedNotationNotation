# GUI Tests Agent

## Overview
This directory owns pytest coverage for `src/gui/`.

## Purpose
- Validate real GUI processors, Oxdraw integration, and headless-safe functionality.
- Keep tests aligned with `src/gui/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/gui/ -q`.
