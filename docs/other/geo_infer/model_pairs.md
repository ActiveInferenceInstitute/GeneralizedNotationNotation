# GNN agent/process pairs

> **Document Metadata**
> **Type**: Integration guide | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [Export qualification](experiment_qualification.md) | [Overview](README.md)
> **Last Updated**: 2026-10-13 | **Version**: 0.1

GEO-INFER-ACT can run a categorical GNN agent against a separate GNN-defined
process. Reproduce both artifacts with this repository's actual producer:

```bash
uv run python docs/other/geo_infer/qualify_export.py \
  STUDY/reference --step-seconds 21600 --output output/agent-export-new.json
uv run python docs/other/geo_infer/qualify_export.py \
  STUDY/uninformative_odor --step-seconds 21600 --output output/process-export-new.json
```

`STUDY` denotes a retained Cascadia edition with source, ordered H3 identifiers
and artifact in each condition folder. The output parent must exist and the
receipt paths must be new. GNN's producer extracts the declared matrices and
axis sizes from notation; these are not substituted by a simulation script.
The qualifier checks complete canonical artifact equality, including the clock,
and binds current producer-source bytes. It does not execute the agent.

The consumer's `run_closed_loop(..., process_artifact=...)` uses process A/B
only to generate observations and transitions. Agent A/B/C/D/E govern real
pymdp state and policy inference. Its
`GEO-INFER-ACT/examples/cascadia/run_model_pair.py` entry point retains both
consumed artifacts and supports a fresh process/agent replay. The full consumer
guide is `GEO-INFER-ACT/docs/cascadia_model_mismatch.md` in the GEO_INFER checkout.

| Boundary | Actual responsibility |
| --- | --- |
| GNN | Reproduce notation-to-artifact extraction and ordered state IDs |
| ACT | Validate compatible carrier/clock, infer, act, sample process and replay |
| fep_formal | Distinguish generic theorems and hypotheses from model-instance evidence |
| LectureCreate | Invoke repository tools in selected environments and retain teaching outputs |

Schema `gnn-geo-infer/1` does not label observation/action alphabets. Matching
dimensions cannot prove their meanings align. The Cascadia compiler retains
those alphabets across its declared cue variants; use that explicit model
construction, or review alignment separately. No exporter change, new contract
version or native-proof promotion is implied by this consumer extension.

If a process cue has zero agent evidence, execution fails instead of repairing
the model. Retain that attempt; a producer-reproduced artifact can still encode
an inadequate agent. Portable file integrity, current-source reproduction,
fresh execution, native proof and ecological validity remain separate claims.

The consumer also ships a retained four-pair edition and deterministic HTML/SVG
comparison commands in `GEO-INFER-ACT/docs/cascadia_model_mismatch.md`.
HTML generation qualifies supplied-manifest integrity and finite aligned scores;
it does not reproduce the GNN exporter or perform numerical inference. Run the
producer qualification and consumer replay commands as separate evidence steps.
