# California factored monitoring models

The companion PLACE example uses three explicit GNN sources for mapped fire
perimeter, catalogued earthquake and year-end drought indicators. Their input
geography and prior estimation belong to GEO_INFER. GNN owns production of
the categorical factored artifacts. The contract is
`gnn-geo-infer/factored/1`, using explicit JSON dependency axes.

The source paths in the companion checkout are
`GEO-INFER-PLACE/locations/california/models/{fire,seismic,drought}.gnn.json`.
Their exported counterparts end in `.geo-infer.json`. These are foreign
checkout paths, not a GNN Python dependency. See the sibling's
`GEO-INFER-PLACE/locations/california/README.md` and
`GEO-INFER-INTRA/docs/examples/california_statewide_monitoring.md` for the
data definitions and measured experiment. GNN means Generalized Notation
Notation here.

## Model contract

| Component | Declared meaning |
| --- | --- |
| Indicator factor | Binary status of a source-derived historical threshold label |
| Sensor factor | Passive, coarse or detailed monitoring mode |
| Control factor | Selects the next sensor mode |
| Indicator-report modality | No report, negative or positive; depends on both factors |
| Resource-use modality | Zero, one or three units; depends on sensor mode |
| Indicator transition | Identity within the static indicator episode |
| Sensor transition | Controlled categorical mode selection |
| Preferences | Explicit resource log utilities; report preference is constant |
| Policy roster | Three distinct one-step policies with uniform prior |

Sensor reporting coverage and classification accuracy are declared assumptions.
The real historical indicators do not calibrate those sensor parameters.
Source-derived priors are passed to the consumer explicitly. The source's
prototype initial joint belief is not claimed as a fitted California prior.
The physical step is 86,400 seconds; monitoring rounds in the teaching
experiment do not imply newly acquired daily empirical observations.

## Separate producer qualification

From the GNN root, bind the actual GNN environment and set `GEO_ROOT` to the
qualified sibling checkout. For existing artifacts, reproduction is read-only:

```bash
for hazard in fire seismic drought; do
  uv run python docs/other/geo_infer/qualify_factored.py \
    "$GEO_ROOT/GEO-INFER-PLACE/locations/california/models/$hazard.gnn.json" \
    "$GEO_ROOT/GEO-INFER-PLACE/locations/california/models/$hazard.geo-infer.json" \
    --step-seconds 86400 --check
done
```

Without `--check`, the qualifier creates one artifact and refuses an existing
target. It invokes `build_geo_infer_factored_artifact`; it does not reproduce
GNN generation by copying a consumer implementation. Both source and artifact
inputs have a four-MiB limit and refuse symlinks. Duplicate artifact fields,
changed clocks, altered arrays and provenance drift fail reproduction.

```bash
uv run pytest docs/other/geo_infer/test_qualify_factored.py \
  tests/export/test_geo_infer_factored.py -q --no-cov
```

The qualifier prints the canonical artifact digest. Source-byte custody,
producer-owner revision, actual environment identity and experiment execution
need their own retained records. A successful reproduction alone is neither
native Lean evidence nor a physical hazard forecast.

## Consumer and reusable methods

The consumer uses ACT's existing factored contract and exact reference solver.
The new `FactoredPolicyEvaluator` compiles the same joint transition and
emission tensors for bounded one-step batches. It retains information gain,
expected preference cost, EFE and policy probabilities. Tests compare these
against exact future-observation trees for correlated beliefs and zero-prior
policies. Explicit longer policies are refused by the batch interface and
remain supported by the reference path.

ACT's resource allocator selects sensor actions under a shared budget, with
optional group expenditure floors. The exact optimization objective sums local
EFE improvements. The statewide field is mean-field approximate, and the sum
of local information gains is not exact joint spatial information gain.

Do not conflate byte reproduction, numerical equality, source currency,
field calibration, mathematical proof and video acceptance. Each is a distinct
evidence plane with its own inputs and checks.
