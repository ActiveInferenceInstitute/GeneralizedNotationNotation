# bnlearn Renderer

`generate_bnlearn_code(model_data, output_path=None)` emits a standalone native
bnlearn program. Complete stationary categorical A/B/D models use authored CPTs
for exact filtering, with canonical B axes `(next_state, previous_state, action)`.
The renderer itself never imports bnlearn. Install the runtime with
`uv sync --extra bnlearn`.

## Usage and output

```python
from gnn.render.bnlearn import generate_bnlearn_code
code = generate_bnlearn_code(parsed_spec, "output/model_bnlearn.py")
```

Step 12 sets `BNLEARN_OUTPUT_DIR`. The program writes `simulation_results.json`
containing all timestep beliefs, observations, transition actions, source and
canonical parameters, matrix provenance, model digest and generated-script digest.
Within a pipeline run, source identity also binds the selected source digest.
The source timestep count and seed are retained. CLI flags `--observations-json`
and `--transition-actions-json` accept complete integer sequences. Without them,
the program samples observations from the declared model with cyclic fixed actions.

## Scientific contract

The first update conditions D on the first observation. Later updates condition
`S_prev -> S`, `Action -> S`, `S -> O` on the previous posterior, selected action
and observed outcome. Native `bn.make_DAG`, pgmpy `TabularCPD` and
`bn.inference.fit` perform every conditional query. C/E do not enter a fixed-action
filter; this contract does not optimize preferences or policies. Shared canonical
validation retains any declared-precision mass correction in matrix provenance;
original source parameters remain available separately.

Missing or malformed A/B/D, unknown B orientation, and extra component matrices
cannot fall back to synthetic CPTs. Continuous, nonstationary and unsupported
compositions produce explicit refusals. The current compiler admits at most
2,000,000 source cells, 100,000 timesteps and 100,000,000 timestep/CPT cells.
The supervised process deadline remains authoritative for runtime. Metadata-only compatibility calls retain the historical
synthetic learning demonstration, labelled `synthetic_backend_smoke_v1` with
`source_parameter_fidelity: false`.

`tests/render/test_bnlearn_source_contract.py` compares native asymmetric
posteriors with independent forward-filter calculations. These finite-model
witnesses support this translation contract; source-specific causal discovery,
component composition and Active Inference control remain outside its scope.
See [roadmap contracts](../../../../docs/development/remaining_roadmap_contracts.md).
