# Ontology Tests Agent

## Overview
This directory owns pytest coverage for `src/ontology/`.

## Purpose
- Validate real ontology processing, annotations, and MCP wrapper behavior.
- Keep tests aligned with `src/ontology/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/ontology/ -q`.
