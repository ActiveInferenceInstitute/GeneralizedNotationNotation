# Pytest Benchmark Logs

The `.benchmarks/` directory acts as the transient JSON logging hub for the project's zero-mock performance execution testing tier. Generated dynamically by the `pytest-benchmark` package plugin, it maps execution history timestamps and regression variances tracking exact functional efficiency over subsequent CI/CD suite completions.

## Interaction

Developers do not manually configure this folder. Review recent regression history natively using:

```bash
uv run pytest src/tests --benchmark-compare
```
