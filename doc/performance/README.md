# GNN Performance Guide

Performance work should begin with a measured run. The pipeline exposes resource
estimation, profiling, explicit framework selection, execution workers, and output
scoping; it does not expose a generic cache, memory-limit, streaming, or optimization
flag layer.

## Measure a focused run

```bash
uv run python src/main.py \
  --target-dir input/gnn_files \
  --output-dir output/performance-baseline \
  --only-steps "3,5,11,12" \
  --profile \
  --verbose
```

Use a small model directory for repeatable comparisons. Record the command, Python/uv
versions, selected frameworks, and the generated pipeline summary. Do not turn one
local timing into a general performance guarantee.

## Estimate model resources

```bash
uv run python src/5_type_checker.py \
  --target-dir input/gnn_files \
  --estimate-resources \
  --verbose
```

Large dense transition tensors can grow cubically with state size. Use the scaling
orchestrator's preflight limits before generating large studies:

```bash
uv run python scripts/run_pymdp_gnn_scaling_analysis.py --help
```

## Select the backend and scope outputs

```bash
# Render only one backend.
uv run python src/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output/pymdp-run \
  --frameworks pymdp \
  --strict-framework-success

# Execute only that run's artifacts.
uv run python src/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output/pymdp-run \
  --render-output-dir output/pymdp-run/11_render_output \
  --frameworks pymdp \
  --execution-workers 2 \
  --timeout 1200
```

`--execution-workers` parallelizes rendered scripts, not timesteps inside one
simulation. `--distributed --backend ray` or `--distributed --backend dask` selects
the optional distributed dispatcher when its dependency is installed.

## Reduce expensive work

Use explicit step selection rather than undocumented optimization flags:

```bash
# Parse and type-check only.
uv run python src/main.py --only-steps "3,5" --estimate-resources --verbose

# Skip visual and LLM work for a validation loop.
uv run python src/main.py --only-steps "3,5,6" --skip-llm --verbose

# Disable Step 16 GridWorld animation artifacts.
uv run python src/main.py --only-steps "16" --no-animations --verbose
```

The following are not main-pipeline options: `--performance-tracking`, `--workers`,
`--parallel-strategy`, `--memory-limit`, `--streaming-mode`, `--enable-cache`,
`--complexity-analysis`, `--optimization-suggestions`, `--resource-estimation`, and
`--target`. Use the documented options above or the relevant module API.

## Inspect results

- Pipeline summary: `output/<run>/00_pipeline_summary/pipeline_execution_summary.json`
- Step summaries: `output/<run>/<step>_output/`
- Execution details: `output/<run>/12_execute_output/`

Use `python -m json.tool` or `jq` to inspect generated JSON. Compare runs only when
input models, selected steps, framework availability, and output scope are held
constant.

## Related references

- [Resource metrics](../gnn/operations/resource_metrics.md)
- [Setup](../SETUP.md)
- [Pipeline](../pipeline/README.md)
- [Troubleshooting](../troubleshooting/README.md)
