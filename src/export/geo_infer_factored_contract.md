# Explicit factored categorical interchange

`gnn-geo-infer/factored/1` is a separate categorical contract for declared
state factors, observation modalities, conditional dependencies, and finite
policy sequences. It does not change `gnn-geo-infer/1` or infer dependency axes
from GNN Markdown. The input is explicitly structured JSON; source provenance
identifies it as `explicit_factored_json`. This standalone exporter is separate
from the Step 7 Markdown pipeline. Run the export command from the GNN checkout;
run the consumer example from the GEO-INFER environment.

```bash
PYTHONPATH=src uv run python -m export.geo_infer_factored \
  src/tests/export/factored_example.json /tmp/factored.geo-infer.json \
  --step-seconds 60
```

`build_geo_infer_factored_artifact(content, step_seconds=60)` returns a validated
artifact and hashes the exact UTF-8 source text. It requires every field below;
it never supplies missing matrices, dependencies, policies, or probability mass.

| Source field | Meaning and order |
| --- | --- |
| `model_name` | Nonempty model label. |
| `state_factors` | Ordered objects with `id` and unique `states` labels. |
| `control_factors` | Ordered objects with `id` and unique `actions` labels. |
| `modalities` | Ordered objects with `id`, `outcomes`, `dependencies`, `likelihood`, and `preferences`. |
| `transitions` | One object per state factor, with `dependencies`, `control_factor`, and `probabilities`. |
| `initial_joint` | Joint probability vector in lexicographic state-tuple order; the last state factor varies fastest. Correlations are retained. |
| `policies` | Explicit unique policies, each `[time][control_factor]`; every policy has the same finite horizon. |
| `policy_prior` | E over the listed policies in exactly that order. Its length is not the action count unless only one-step policies are enumerated. |

Dependencies are unique zero-based state-factor indices in tensor-axis order.
For modality m, A has axes `[outcome_m, dependency_0, dependency_1, ...]`.
For state factor f, B has axes
`[next_state_f, previous_dependency_0, previous_dependency_1, ..., action]`;
`control_factor` selects that final action axis. An empty dependency list is
valid and expresses independence from the current state. Arrays are finite;
A and B sum to one along axis zero. C preferences are real log preference scores.
The initial joint and E are normalized nonnegative vectors; exact zeros remain
zero. Distinct factors and modalities are conditionally independent only given
the dependencies explicitly declared by their tensors.

The exporter adds `schema_version`, `model_type='categorical_factored'`,
`time.step_seconds`, and `provenance` containing producer, source kind, and source
SHA-256. Unknown keys, duplicate JSON keys, bool indices, reordered incompatible
axes, and missing values are rejected. Artifact SHA-256 uses sorted compact
Python JSON over the full artifact; it is distinct from the exact source digest.

## Exact reference inference in GEO-INFER

```python
from geo_infer_act.core.gnn_factored_contract import (
    FactoredGNNArtifact, infer_factored_step,
)

artifact = FactoredGNNArtifact.load('/tmp/factored.geo-infer.json')
step = infer_factored_step(artifact, [0, 2])
following = infer_factored_step(artifact, [1, 0], prior=step['next_prior'])
```

Each call conditions the joint prior once using one integer observation per
modality. The result contains posterior, evidence, free energy `-log(evidence)`,
policy posterior, expected free energy, selected policy, its first action, and
`next_prior`. The next prior applies that action once. Calling the function again
without `prior` intentionally starts from `initial_joint`; no hidden state or
implicit wall-clock propagation is maintained. `step_seconds` declares the
physical duration represented by each B transition; it does not resample data.

The reference backend is `geo-infer-exact-joint`, not pymdp. For every explicit
policy it sums negative expected log preferences minus expected state information
gain. It enumerates all future observation histories and conditions each branch
before the next transition. Policy probabilities are `softmax(-G + log(E))`,
with precision one and deterministic first-index argmax selection. Future
information is therefore conditioned on earlier observations rather than counted
again from an unchanged open-loop uncertainty. This is a bounded exact reference,
not a scalable approximation or a claim of mean-field equivalence.

## Bounds and conformance

The implementation permits at most 256 joint states, 8 state/control factors,
8 modalities, 256 policies, and horizon 8. It additionally limits total matrix
entries to one million and the conservative exact observation-tree work estimate
to 20 million. Dimension products, tensor nesting, and work bounds are checked
before NumPy tensor conversion. A source or artifact is at most four MiB. These
limits deliberately reject state or policy explosions. Finite individual entries
can still exceed floating-point range when combined across modalities or time;
the consumer raises `ValueError` for such arithmetic instead of returning
nonfinite policy scores or selecting an action from NaN probabilities. E-zero
policies stay excluded when stabilizing the policy softmax.

The paired fixture has 2×3 states, 2×4 outcomes, two control factors, A dependency
order `[1, 0]`, action-dependent B, a correlated initial joint distribution, and
four two-step policies. GEO tests compare its information value against an
independent entropy-chain-rule calculation over all 64 future observation pairs.
Exporter and consumer independently validate the same wire contract in their
own locked environments. Their validation implementations must remain aligned;
paired conformance is the compatibility gate.
