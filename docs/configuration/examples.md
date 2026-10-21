# GNN configuration examples

These examples use implemented command-line options and the checked-in
[input/config.yaml](../../input/config.yaml). The [configuration guide](README.md)
owns precedence and limitations. Run commands from the repository root after
[setup](../SETUP.md); each output directory below is a new run destination.

## Basic configuration

Validate a complete committed model before a pipeline run:

```bash
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md --strict
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/basics --output-dir /tmp/gnn-config-basic \
  --only-steps "3,5" --verbose
```

`gnn validate` takes a file; the pipeline takes a directory. Inspect the run summary
and model receipts; exit 2 represents completed work with warnings.

## Development configurations

Use CLI selection to iterate on the affected stages:

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/basics --output-dir /tmp/gnn-config-export \
  --only-steps "3,5,7" --skip-llm --verbose
```

For Step 2 defaults, edit the existing `pipeline.fast_only` or
`pipeline.comprehensive` fields in `input/config.yaml`. Provision dependencies
explicitly with `uv sync --frozen --extra dev`; test selection is documented in
the [testing guide](../testing/README.md).

## Multi-target configurations

Select backends with `--frameworks`, after checking
[model admission and runtimes](../execution/FRAMEWORK_AVAILABILITY.md).
For example, a render-only path requires no Julia execution:

```bash
uv run --frozen --no-sync python src/gnn/main.py \
  --target-dir input/gnn_files/basics --output-dir /tmp/gnn-config-render \
  --only-steps "3,5,11" --frameworks "pymdp,jax" --verbose
```

Run Step 12 only with admitted rendered models and installed runtimes. Rendering
is not numerical acceptance. Experimental THRML/cpomdp need explicit selection.

## Performance optimization

Reduce selected steps or choose a bounded input directory during development.
Use the consuming module's documented timeout and resource options. YAML fields
are not universal switches for caching, GPUs, parallel execution or remote workers.
Read [run ownership](../development/run_ownership_migration.md) before sharing an
output destination or comparing results from separate runs.

## Production configurations

Freeze the environment and source, retain the resolved run configuration and
model selection, and check outcome/cleanup receipts. Configure credentials outside
tracked YAML. Deployment-specific containers and scheduling belong to the
operator's environment; [security boundaries](../security/README.md) describe
what GNN itself enforces.

## Cloud configurations

There is no generic `--config`, named profile loader, environment interpolation
or cloud/distributed scheduler in the main pipeline. Provider-specific LLM
configuration lives in the [LLM guide](../llm/README.md). Source-preserving context
limits remain explicit [roadmap work](../../TO-DO.md#major-work).

## Specialized workflows

Read [RxInfer](../rxinfer/README.md), [PyMDP](../pymdp/README.md), and
[framework integration](../gnn/integration/framework_integration_guide.md) for
backend-specific options. Their admission contracts take precedence over a generic
example. Model parameters belong to complete authored GNN files.

## Troubleshooting configurations

```bash
uv run --frozen --no-sync gnn preflight
uv run --frozen --no-sync gnn health
uv run --frozen --no-sync python src/gnn/main.py --help
```

Compare supplied flags with help and inspect the current run's summaries. Keep
unsupported configuration, missing optional runtimes and backend failures distinct.
Return to [configuration](README.md), [troubleshooting](../troubleshooting/README.md)
or the [documentation hub](../README.md).
