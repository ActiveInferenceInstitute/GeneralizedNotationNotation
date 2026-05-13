# Render Tests Agent

## Overview
This directory owns pytest coverage for `src/render/`.

## Purpose
- Validate real renderer processors, framework emitters, CLI target behavior, and render-to-execute contracts.
- Keep tests aligned with `src/render/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/render/ -q`.
