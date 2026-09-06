# Render Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/render/`.

## Purpose
- Validate real renderer processors, framework emitters, CLI target behavior, and render-to-execute contracts.
- Keep tests aligned with `src/gnn/render/AGENTS.md` and the render module README.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/render/ -q`.
