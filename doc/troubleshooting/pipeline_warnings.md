# Pipeline Warning Troubleshooting Guide

A pipeline warning means the step produced a result with a caveat. Always inspect the
step summary and distinguish `SKIPPED`, `SUCCESS_WITH_WARNINGS`, and `FAILED` rather
than relying on a fixed framework count.

## First checks

```bash
uv run gnn preflight
uv run gnn health
cat output/00_pipeline_summary/pipeline_execution_summary.json | python -m json.tool
```

The exact output root may be a custom `--output-dir`; use that run directory when it
is not `output/`.

## Missing framework warnings

Step 12 can skip an unavailable Python or Julia runtime. Check the executor status:

```bash
uv run gnn health
uv run python src/12_execute.py \
  --target-dir input/gnn_files \
  --output-dir output \
  --render-output-dir output/11_render_output \
  --frameworks "pymdp,jax" \
  --verbose
```

For Julia, instantiate the matching committed environment:

```bash
julia --startup-file=no --project=src/execute/rxinfer -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=src/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'
```

For optional PyTorch or bnlearn paths, read the availability reason in
`src/render/framework_registry.py` before installing anything. These are not part of
the default lock.

## Render warnings

Regenerate into an isolated output directory by running Step 11 again; there is no
`--force-regenerate` flag:

```bash
uv run python src/11_render.py \
  --target-dir input/gnn_files \
  --output-dir output/render-retry \
  --frameworks pymdp \
  --verbose
```

## Test-step warnings or failures

Run the test surface separately for a clearer error:

```bash
uv run --extra dev python -m pytest src/tests/ -q \
  --ignore=src/tests/llm/test_llm_ollama.py \
  --ignore=src/tests/llm/test_llm_ollama_integration.py
```

A focused pipeline can skip Step 2 with `--skip-steps "2"`; this does not replace
running the test suite when validating a change.

## LLM warnings

Step 13 uses the local Ollama configuration in `input/config.yaml` and environment
variables such as `OLLAMA_MODEL`. If no local or cloud provider is available, skip it
explicitly:

```bash
uv run python src/main.py --skip-llm --verbose
```

## Visualization warnings

Headless Matplotlib warnings are expected in environments without a display when the
artifact itself is produced. Inspect the generated files and step summary. Do not add
an undocumented `--include-d2` flag; D2 support is owned by the visualization module
and its own installed-tool checks.

## Reproduce and report

1. Re-run one model directory, one framework, and one or two steps.
2. Use `--verbose` and an isolated `--output-dir`.
3. Include the exact command and the relevant JSON/log summary.
4. State whether the result was skipped, warning, or failed.

```bash
uv run python src/main.py \
  --target-dir input/gnn_files/pomdp_gridworld \
  --output-dir output/repro \
  --only-steps "3,5,11,12" \
  --frameworks pymdp \
  --verbose
```
