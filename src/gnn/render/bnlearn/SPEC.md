# bnlearn Renderer — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

`generate_bnlearn_code(model_data, output_path=None) -> str` validates and emits
a standalone Python program. With complete finite A/B/D it delegates to
`source_program.py`, installing canonical conditional probability tables and
performing exact native fixed-action filtering. Source matrices, normalization
provenance, timestep count, seed, source identity and script digest are retained.
C/E are unused by conditional filtering, explicitly recorded in scientific limits.

The source compiler rejects missing matrices, malformed mass, undeclared 3-D B
orientation and unrepresented component matrices before replacing any output.
Unsupported continuous/nonstationary/composed semantics never reuse the
metadata-only synthetic demonstration. That compatibility demonstration has an
explicit `synthetic_backend_smoke_v1` contract and false source-fidelity flag.

Step 12 uses `BNLEARN_OUTPUT_DIR`; the runtime writes `simulation_results.json`.
Rendering requires core dependencies only. Runtime requires the `bnlearn` extra
(bnlearn, pgmpy and NumPy). Availability alone does not establish scientific
correctness. Native tests compare the complete belief sequence against independent
forward filtering using asymmetric probabilities and nontrivial actions.

See [README](README.md) for commands, admission and scientific boundaries.
