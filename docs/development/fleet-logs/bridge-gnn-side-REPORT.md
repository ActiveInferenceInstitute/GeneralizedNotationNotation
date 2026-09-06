# bridge-gnn-side REPORT — fep_lean bridge hardening (2026-09-04)

Worker scope: the three Phase-5 bridge-hardening sites (O1 EFE
convention, F2 connection-annotation fix, provenance strict mode) plus
their tests. NOT committed — the GNN fleet/herdr tabs own GNN commits;
the fep_lean-side orchestrator (omp, w18) leaves this tree as-is.

fep_lean-side companion slices (committed over there, digest `e648016`):
Q3/Q4 exit, W1 bridge-operations, contract v0.2, P4b continuous
emission. Repo HEAD at GNN work start: `aa20514c77bb` (fleet-dirty tree;
landed on top of it, no fleet edit reverted). GNN HEAD moved mid-run to
`64d49355acf1` (docs-only) — no impact on the touched files.

## Files changed + why

| File | Change | Why |
|---|---|---|
| `src/gnn/render/pomdp_contract.py` | `RxInferSimulationV1` TypedDict: added `expected_free_energy_convention: str` (additive) | Finding O1: EFE values from different backends are convention-labeled; certificates may compare only convention-matched quantities |
| `src/gnn/execute/pymdp/pymdp_simulation.py` | New module constant `EFE_CONVENTION_PYMDP` (pymdp 1.0.0 `neg_efe = -EFE`, expected-utility = linear payoff sum `sum_o q(o) C[o]` + states info gain; verified against installed `pymdp/control.py` `compute_neg_efe_policy` / `compute_expected_utility`); `pymdp_simulation_v1` results dict now carries `expected_free_energy_convention` | Honest per-backend EFE declaration (finding O1); pymdp's quantity is NOT the risk+ambiguity EFE |
| `src/gnn/render/jax/jax_renderer.py` | New constants `EFE_CONVENTION_JAX` / `EFE_CONVENTION_JAX_JSON`; the generated model script's `metrics` block now embeds `expected_free_energy_convention` | jax EFE is a heuristic (`obs_entropy - pragmatic + 0.1*KL(next||D)`; pragmatic is a linear payoff, not a KL vs C) — declared, not passed off as risk+ambiguity |
| `src/gnn/render/pytorch/pytorch_renderer.py` | `simulation_results.json` gains `expected_free_energy_convention` | pytorch EFE = ambiguity + risk with softmax(C) as the preference reference — closest to Lean's shape but not directly comparable (bridge finding O1) |
| `src/gnn/execute/data_extractors.py` | `expected_free_energy` + `expected_free_energy_convention` propagated in all three file-based extractors (`extract_pymdp_data_from_files`, `extract_rxinfer_data_from_files`, `extract_pymdp_like_data_from_files`, shared `extract_activeinference_jl_data_from_files` tuple) | The convention must survive extraction into execution summaries |
| numpyro / stan / rxinfer-continuous / discopy / bnlearn | no edit — these emit no EFE surface; omission is the honest declaration per the plan's "backends that emit no EFE → omit the field" rule | — |
| `src/gnn/parsers/common.py` | `Connection` AST node: additive `annotation: Optional[str]` field (flows through `to_dict()` via `__dict__` automatically) | v1.1 annotations were parsed-but-lost (glued onto the target name) |
| `src/gnn/parsers/markdown_parser.py` | `_parse_connection_definition`: strips the FIRST `:annotation` suffix from the target side per `doc/gnn/gnn_syntax.md` section 3 ("parsers must accept and preserve them"); annotation stored on the Connection, target name clean | F2: annotated edges (`A>B:label`) previously resolved target `B:label` → "Connection references unknown target variables" warnings |
| `src/gnn/parsers/markdown_serializer.py` | Connection writer re-emits `:annotation` when present | Round-trip preservation (syntax doc: parsers "may ignore them for structural validation" but must preserve) |
| `src/gnn/schema_validator.py` | `_validate_strict_requirements`: a document whose `gnn_section` starts with `FepLean` (bridge convention, bridge contract §4) must carry Signature keys `source_repository`, `source_commit`, `lean_module`, `projection_tool`, `target_syntax` — one error per missing key, naming the key. Non-bridge documents unchanged. Enforced only at STRICT rank+ (dispatch: `_validate_strict_requirements` runs at ≥ STRICT); STANDARD level untouched | Provenance strict mode for bridge artifacts (Phase 5.3); fail-closed exactly where `--strict`/STRICT validation runs |
| `tests/gnn/test_gnn_parsing.py` | **NEW** `test_parse_annotated_edges_v11` (annotated directed/undirected edges resolve targets, annotations preserved, zero unknown-target warnings) + `test_annotated_edge_serialization_round_trip` (`D>s:prior_initialization` survives serialize→reparse) | Pin the F2 fix behaviorally |
| `tests/gnn/test_gnn_validation.py` | **NEW** `TestBridgeProvenanceStrict` — 4 tests: missing-keys → 4 named errors + invalid; complete-keys → no provenance errors; non-bridge → unaffected; STANDARD level → not enforced | Pin the provenance strict-mode contract incl. the level gate |

## Verification

| Check | Result |
|---|---|
| `uv run ruff check` on all touched files | All checks passed |
| `uv run mypy src/gnn/execute/data_extractors.py src/gnn/execute/pymdp/pymdp_simulation.py src/gnn/render/pomdp_contract.py` | Success, no issues |
| Focused suites `tests/gnn tests/render` | `tests/gnn`: 410 passed; `tests/render`: 320 passed, 1 skipped (pre-existing torch-absence opt-in skip), 0 failed — three consecutive full green runs after fixing a jax f-string quoting defect found by the first run (the generated script embedded the convention string unquoted → SyntaxError; fixed via `EFE_CONVENTION_JAX_JSON_LITERAL` = `json.dumps(...)` and re-verified) |
| Zero-skip contract (`tests/test_zero_skip_contracts.py` policy) | No `pytest.skip`/`xfail` in the new tests (verified by grep) |
| Annotated-edge probe | `D>s:prior_initialization` → target `s`, annotation `prior_initialization`, no warning; round-trip through `MarkdownSerializer` preserves `:annotation` |
| CLI strict validate on real bridge artifacts (regression check — the CLI path exercises `gnn.schema`, not `GNNValidator`; it does NOT exercise the provenance keys) | `FepLeanContinuousOU.md` exit 0, `FepLeanSymmetricBool.md` exit 0 |
| GNNValidator STRICT on the same bridge artifacts (this is the path that exercises the provenance keys) | `FepLeanContinuousOU.md` and `FepLeanSymmetricBool.md`: zero provenance errors (executed probe: `GNNValidator().validate_file(..., validation_level=STRICT)`) |
| p4b pipeline re-run (steps 3,5,11,12) on the hardened tree (fresh run, 2026-09-04 12:38:52 UTC, output `/tmp/p4b-hardened-rerun/`) | 4/4 steps success, 0 failed — render `model_kind: continuous`, 5 continuous backends processed (jax, pytorch, numpyro, stan, rxinfer), 4 discrete-only backends `unsupported` (pymdp, activeinference_jl, discopy, bnlearn), 0 failed; step 12: 5 scripts found, 4 success, 1 skipped (pytorch, torch absent) |

## Honest boundary

- The O1 declarations state what each backend's code computes (verified
  by reading `pymdp/control.py` `compute_neg_efe_policy` /
  `compute_expected_utility`, jax `compute_expected_free_energy`,
  pytorch's ambiguity+risk loop). They do not reconcile the conventions —
  that mapping is future certificate work on the fep_lean side.
- The provenance strictness keys off the `FepLean` GNNSection prefix
  only, exactly per contract §4's bridge-convention row; a non-bridge
  document naming itself `FepLean*` without provenance will now fail
  strict validation — that is the intended fail-closed behavior.
- The execution-summary surfacing of the convention rides the existing
  `simulation_data` extraction path (file-based extractors + the pymdp
  results dict); the stdout-based `extract_simulation_data` parsers were
  not extended — file-based extraction is the path the pipeline uses
  (verified in the p4b re-run).
