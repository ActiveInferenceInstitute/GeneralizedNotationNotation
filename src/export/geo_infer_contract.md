# GNN / GEO-INFER artifact contract version 1

GNN owns this format's producer and notation semantics. GEO-INFER owns the
consumer validator and spatial/temporal inference runtime. Exchange JSON data
between independently installed repositories; neither environment imports the
other repository's source. `geo_infer` is an opt-in export registry format,
currently supported by `export_model` and the standalone CLI. The five default
Step 7 exports retain their existing behavior.

## Required structure

Every object below has exactly the listed keys. Unknown keys or versions fail
on the consumer. Top-level `schema_version` is `gnn-geo-infer/1`, `model_type`
is `categorical`, and `model_name` is a nonempty string.

| Object | Required keys and interpretation |
| --- | --- |
| dimensions | `states`, `observations`, `actions`: positive integers |
| matrices | `A`, `B`, `C`, `D`, `E`: finite numeric arrays |
| space | `kind`: `categorical` or `h3`; `state_ids`: unique strings in matrix state order |
| time | `step_seconds`: positive seconds per B transition, explicitly supplied by the caller |
| provenance | `producer`: nonempty label; `source_sha256`: lowercase SHA-256 of original UTF-8 source |

A axes are `[observation,state]`; B axes are `[next_state,current_state,action]`.
C contains log preferences, D is the initial state prior, and E is the prior over
one-step policies in action order. A/B/D/E sum to one along axis zero (absolute
tolerance 1e-8), with nonnegative entries. C has one value per observation; D one
per state; E one per action. Matrices are not normalized, resized or substituted
on export. All five must be explicit in `InitialParameterization`.

Only one state factor, observation modality and control factor are accepted.
Noncanonical or contradictory B axis provenance fails; users must explicitly
resolve the source model. The general `canonicalize_pomdp` helper separately
supports action-first B conversion and maintains its resulting orientation
metadata, including repeated-call idempotence and unequal action/state counts.

H3 IDs must be canonical cells at one resolution, and must be supplied in the
existing matrix order. Labels do not change matrix axes. H3 validation requires
the optional `geo-infer` dependency group. Categorical IDs default to numeric
strings; they make no geographic claim.

## Producer usage

```bash
uv sync --extra dev --extra geo-infer
PYTHONPATH=src uv run --no-sync python -m export.geo_infer \
  input/gnn_files/pomdp_gridworld/pomdp_gridworld_3x3.md \
  /tmp/gridworld.geo-infer.json --step-seconds 60
```

For spatial states, add `--space-kind h3 --state-ids /path/to/ordered_cells.json`.
The IDs file must contain a JSON list. The exporter never infers seconds or CRS
from notation labels. Input source and IDs files are bounded to four MiB each;
the dense matrix entry budget is 1,000,000.

```python
from pathlib import Path
from export.processor import export_model

result = export_model({
    'raw_content': Path('model.md').read_text(),
    'geo_infer': {'step_seconds': 60},
}, Path('/tmp/exports'), formats=['geo_infer'])
assert result['success'], result
```

## Consumer timing and verification

GEO-INFER-ACT's `GNNArtifact` loads bounded JSON (32 MiB) and rejects duplicate
keys. `run_gnn_inference` accepts records containing `timestamp` and an integer
`observation` index. D is the prior at the first timestamp. Each timestamp
conditions once; the selected action produces the next prior through B. TIME
requires UTC-convertible aware timestamps and fixed intervals, rejecting gaps,
reversals and duplicates. Default execution budget: 10,000 steps. Zero-likelihood
observations fail. E reaches real pymdp 1.0.3 policy inference.

Artifact digests use sorted compact Python JSON, UTF-8, without a trailing
newline; source digests identify bytes and do not authenticate the producer.
This format does not claim RFC 8785 canonicalization.

Run `src/tests/export/` and `src/tests/gnn/test_pomdp_extractor*` locally. GEO's
`GEO-INFER-TEST/validate_gnn_interchange.py` takes explicit GNN checkout and Python
paths, checks the real gridworld and a SPACE-generated seven-cell H3 model, and
verifies exact matrix/order preservation and deterministic real inference.

Continuous GNN, factorized models, irregular time and longer policies require
separate contracts. See GNN-02 through GNN-05 in `TO-DO.md`; they are not accepted
by version 1.

## Verified companion implementation

The paired GEO implementation is
[`e028aa90`](https://github.com/ActiveInferenceInstitute/GEO-INFER/commit/e028aa9060e05f765762224499f5e2c714cf25a3).
Its receipt records identical categorical/H3 artifacts and complete traces across
GEO Python 3.11 and 3.12, with GNN running separately on Python 3.11. Main-branch
integration and CI revision pairing remain the explicitly scoped GNN-04/GNN-06
work; this reference identifies the tested topic implementation.

## Linear Gaussian v2

`geo_infer_gaussian.build_geo_infer_gaussian_artifact` exports the explicit
`gnn-geo-infer/2`, `linear_gaussian` model type. The categorical v1 contract above
remains unchanged. The GNN source must explicitly declare `Discrete` in `Time`;
continuous state is supported, while continuous-time generators require a separate
explicit discretization before export. No Euler approximation is performed here.

| Field | Required value or shape |
| --- | --- |
| `dimensions` | Positive integer `states=n`, `observations=m`, `controls=k` |
| `matrices.F` | State transition, `[next_state, previous_state]`, shape `(n,n)` |
| `matrices.G` | Control map, `[next_state, control]`, shape `(n,k)` |
| `matrices.H` | Observation map, `[observation, state]`, shape `(m,n)` |
| `matrices.Q` | Per-interval process covariance `(n,n)`, symmetric positive semidefinite |
| `matrices.R` | Observation covariance `(m,m)`, symmetric positive definite |
| `initial_belief` | `mean` `(n,)` and positive definite `covariance` `(n,n)` |
| `units` | `states`, `observations`, `controls`: one nonempty unit string per coordinate |
| `time` | Exactly `domain="discrete"`, positive finite `step_seconds` |
| `provenance` | Producer string and SHA-256 of the exact UTF-8 GNN source |

Every matrix and initial belief must occur explicitly in GNN
`InitialParameterization` as `F`, `G`, `H`, `Q`, `R`, `prior_mean`, `prior_cov`.
Units describe the coordinates; matrix units follow their row/column ratios and
covariance units follow coordinate products. No unit conversion is implicit.
The artifact accepts only the listed top-level fields plus `schema_version`,
`model_type`, and nonempty `model_name`. JSON numbers must be finite and cannot
be booleans. Dense storage is bounded to one million entries. GEO additionally
bounds JSON input to 32 MiB and rejects duplicate keys.

```bash
PYTHONPATH=src uv run python -m export.geo_infer \
  src/tests/export/gaussian_rectangular.md gaussian.geo-infer.json \
  --model-type linear_gaussian --step-seconds 2 --units units.json
```

For this fixture, `units.json` contains
`{"states":["m","m/s","K"],"observations":["m","m/s"],"controls":["N"]}`.
The three state coordinates, two observations and one control provide a
non-square axis regression fixture.

GEO's `GaussianGNNArtifact` loads the artifact. Its
`run_gaussian_gnn_inference` accepts records with exactly `timestamp`,
`observation` and `control`. Timestamps are aware and regularly spaced by
`step_seconds`; observations and controls are explicit numeric vectors.
The initial belief applies at the first measurement, which is conditioned once.
The record's control then predicts the next prior using `F @ mean + G @ control`
and `F @ covariance @ F.T + Q`. The trace includes prior, posterior, next prior,
and negative log evidence. The runner performs filtering with supplied controls;
it does not synthesize an active-inference policy or label supplied controls as
optimal actions.

## Explicit Step 7 metadata

`process_export(..., formats=["geo_infer"], geo_infer_options=...)` supports
per-source options keyed by exact `file_name` in the Step 3 manifest:

```python
process_export(
    target_dir, output_dir,
    formats=["json", "geo_infer"],
    geo_infer_options={
        "navigation.md": {
            "model_type": "linear_gaussian",
            "step_seconds": 2,
            "units": {
                "states": ["m", "m/s", "K"],
                "observations": ["m", "m/s"], "controls": ["N"],
            },
        },
        "gridworld.md": {"step_seconds": 60},
    },
)
```

The interchange writer reads bounded original source bytes under `target_dir`,
checks resolved containment (including symlinks), and computes source provenance
from those bytes. It does not reconstruct source from the Step 3 summary. Missing
per-model options produce a failed format entry in `export_results.json`. A run
with any model export failure returns `False`, even when another model succeeds.
The default format set remains JSON, XML, GraphML, GEXF and pickle. No physical
step, Gaussian units, or H3 identities are inferred from filenames or examples.

Gaussian Markdown uses one explicit `StateSpaceBlock` with continuous `x`, `y`
and `u` coordinate declarations matching state, observation and control sizes.
Present matrix and prior declarations must match their literal values. Duplicate
semantic sections, additional unsupported coordinates and contradictory shapes
are rejected. The numeric covariance check rejects nonfinite eigenspectra and
uses overflow-safe symmetrization before checking definiteness.

For recursive Step 3 input, metadata keys may be contained relative paths such as
`first/navigation.md`. Basename lookup is accepted only when that basename is
unique in the manifest. Source identity comes from the contained Step 3
`file_path`, whose basename must match `file_name`; nested output directories
preserve distinct models with equal basenames. Existing output-directory,
artifact-file and manifest-file symlinks cannot redirect writes outside the
requested output directory.

## Numbered Step 7 command

After Step 3, select the interchange format and explicit metadata file:

```bash
PYTHONPATH=src uv run --extra dev --extra geo-infer python src/7_export.py \
  --target-dir input/gnn_files --output-dir output \
  --formats geo_infer --geo-infer-options-file geo-options.json
```

`geo-options.json` maps relative source paths to the same `geo_infer` options
shown above. Duplicate JSON keys are rejected. Nested source directories retain
their relative identity; basename metadata is accepted only when unambiguous.
The default invocation still emits the original five formats. These flags also
pass through the main pipeline argument registry. A requested export with absent
metadata returns failure, while retaining per-model diagnostics.
