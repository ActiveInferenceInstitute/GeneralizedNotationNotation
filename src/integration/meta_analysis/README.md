# Integration Meta-Analysis

Meta-analysis helpers for Step 17 integration reporting.

## Components
- `collector.py` gathers cross-step result artifacts.
- `statistics.py` computes aggregate metrics.
- `validator.py` checks collected data for consistency.
- `reporter.py` turns metrics and validation output into summaries.
- `visualizer.py` creates visual summaries when plotting dependencies are available.

## Verification
Run:

```bash
uv run pytest src/tests/integration/ -q
```
