# AGENTS.md “Test Coverage” percentages

Many module `AGENTS.md` files include a **Current** coverage percentage. Those values are **not** automatically refreshed on every commit.

**Visualization (step 8):** the percentage was removed in favor of an explicit `pytest --cov=src.visualization` command in [`src/visualization/AGENTS.md`](../../src/visualization/AGENTS.md).

**Other modules:** treat inline percentages as historical targets unless you re-run, for example:

```bash
uv run pytest src/tests/test_<module>_*.py --cov=src.<module> --cov-report=term-missing
```

When updating a module’s AGENTS file, either refresh the number from the command output or replace it with the same “measure with pytest” pattern used in visualization.
