# GNN Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/`.

## Purpose
- Validate real GNN parsing, schema validation, cross-format processing, and adversarial cases.
- Keep tests aligned with `src/gnn/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/gnn/ -q`.
