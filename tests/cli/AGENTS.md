# CLI Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/cli/`.

## Purpose
- Validate real CLI parsing, subcommand routing, and command behavior.
- Keep tests aligned with `src/gnn/cli/AGENTS.md` and `src/gnn/cli/README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/cli/ -q`.
