# Pytest Benchmark Metrics Caching (.benchmarks/)

## Purpose

This directory serves as the localized runtime metrics persistence boundary for pytest-based performance execution benchmarking across the GNN repository (`pytest-benchmark`).

## Components

- `Linux-CPython-X` / `Darwin-CPython-X`: Platform-independent JSON-serialized logging metrics detailing the exact functional execution times during rigorous active inference structural executions and LLM integration unit tests.

## Operational Standards

- Managed strictly under the `pytest` orchestration parameters.
- Protected natively via repository `.gitignore` standards to prevent caching pollution against master branching contexts. No manual interventions required.
