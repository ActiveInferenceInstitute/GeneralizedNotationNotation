# GNN model registries for geospatial experiment matrices

> **Document Metadata**
> **Type**: Integration guide | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [Model pairs](model_pairs.md) | [Export qualification](experiment_qualification.md)
> **Last Updated**: 2026-10-14 | **Version**: 0.1

A GEO-INFER-ACT matrix selects an ordered registry of categorical GNN exports
and explicit agent/process cases. Its schema is
`geo-infer-act/model-matrix-design/1`; it does not change GNN's artifact
contract. The consumer's guide is `GEO-INFER-ACT/docs/cascadia_model_matrix.md`
in GEO_INFER. That checkout owns batching, inference, process sampling and replay.

Reproduce every selected condition independently in this checkout:

```bash
uv run --no-sync python docs/other/geo_infer/qualify_export.py \
  STUDY/reference --step-seconds 21600 --output output/reference-export-new.json
uv run --no-sync python docs/other/geo_infer/qualify_export.py \
  STUDY/uninformative_odor --step-seconds 21600 --output output/odor-export-new.json
uv run --no-sync python docs/other/geo_infer/qualify_export.py \
  STUDY/reference --step-seconds 21600 --check-receipt output/reference-export-new.json
```

Use the condition's declared physical clock, an existing output parent and
new receipt names. Each condition needs its notation, ordered H3 state IDs,
domain sidecar and `smith_river_homing.geo-infer.json`. Qualification executes
the actual producer and checks the complete canonical artifact, binding source
and producer bytes. Retain receipts beside the experiment rather than adding
files to an already closed matrix edition.

The matrix captures both the raw input-file digest and canonical artifact
digest. An exporter qualification ties the artifact to its notation; consumer
replay ties retained trajectories to execution owners. Reformatting JSON can
change raw custody while preserving the canonical model. The LectureCreate
matrix companion requires the original raw artifact bytes to match the
captured matrix input before qualifying that source directory.

Registry names such as `reference` are aliases, not state, cue or action
semantics. The matrix checks dimensions, ordered spatial carrier and clock
compatibility. Schema `gnn-geo-infer/1` does not label observation/action
alphabets, so their common meanings must be reviewed explicitly. Cascadia's
compiler preserves those alphabets across its reference and cue variants.

An export can be valid while the agent assigns zero evidence to a cue the
process can produce. ACT retains that failed case and completed predecessors
without a success manifest. Neither GNN qualification nor a successful matrix
establishes native proof, calibrated beliefs or ecological validity. For the
mathematical hypotheses, see the foreign path
`specs/cascadia-experiments/matrix-proof-obligations.md` in fep_formal.
