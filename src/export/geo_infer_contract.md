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
