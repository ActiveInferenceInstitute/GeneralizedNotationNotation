# Export Tests

Pytest coverage for `src/export/`.

This folder contains module-focused tests for export processing and serialization round trips.

Run:

```bash
uv run --extra dev python -m pytest src/tests/export/ -q
```

`test_geo_infer_gaussian.py` and `gaussian_rectangular.md` cover the v2 Gaussian
interchange, unequal axes, required metadata, source provenance and opt-in Step 7.

`test_step7_geo_cli.py` runs the numbered Step 3/7 commands, verifies source
provenance, and rejects missing or duplicate physical metadata.
