# LLM Tests Agent

## Overview
This directory owns pytest coverage for `src/llm/`.

## Purpose
- Validate real LLM provider configuration, pipeline model wiring, cache behavior, and local Ollama integration where available.
- Keep tests aligned with `src/llm/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/llm/ -q`.
