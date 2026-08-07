# RxInfer Integration Improvement Roadmap

## Current State (2026-08-07)

Verified by execution (see `src/tests/render/test_rxinfer_model_strategies.py`
for the pinned contract):

- 45/45 GNN files render through the real pipeline path; kind taxonomy is
  structural: 40 flat, 2 hierarchical, 2 multi-agent, 1 factored.
- `detect_model_kind` reads typed fields only (GNNSection, per-level/per-agent
  matrix keys, explicit counts) — prose can no longer reroute rendering. It
  raises `ValueError` on a non-mapping `initialparameterization`.
- Strategy dispatch is live: FlatStrategy (canonical), HierarchicalStrategy
  (native two-level @model; 3+ levels render as documented joint composition),
  MultiAgent/FactoredStrategy (documented joint composition stamping their
  true kind), Continuous/LearningStrategy (loud stubs — see A2/D1 below).
- End-to-end executions verified 2026-08-05/07 with `all_valid=true`:
  hierarchical_pomdp (native two-level, ctx posterior + post-hoc propagation),
  temporal_hierarchy (joint), factorized_posterior (joint),
  multi_agent_coordination (256 joint states), actinf_pomdp_agent (flat, E).
- Results JSON echoes `model_parameters.state_factors`, enabling downstream
  per-factor/per-agent marginal recovery (D4) without re-parsing GNN files.
- E vector (habit prior) modulates action selection via
  `softmax(log E − γ·EFE)` in generated scripts; `E` values reported in
  results (D2 closed).
- Belief-entropy validation is diagnostic + combined gate (entropy stats
  reported; failure only when all-degenerate AND below chance-relative
  accuracy) — exact smoothing on high-signal models no longer fails runs.
- Julia module: `pomdp_model`, `continuous_pomdp_model` (validated LGSSM),
  `hierarchical_pomdp_model` (+ mean-field constraints + initialization,
  empirically required on RxInfer 5.5). Precompile workloads have NO
  try/catch — a broken model fails package precompilation loudly.
- Cross-framework comparison (A6): `run_cross_framework_comparison` renders
  one spec to RxInfer/PyMDP/ActiveInference.jl, executes each under correct
  environments, and emits an HTML with per-framework status reasons plus an
  animated belief-trajectory comparison chart. Zero silent fallbacks.

### Corrections to earlier claims (2026-08-05 red-team audit)

Commit `16d3cb25`'s message over-claimed: M6/M7 (docs), D3 (`detect_factors`),
and D4 (`per_agent_beliefs`) described as done were absent from the tree, and
the strategy-pattern refactor regressed 2/45 renders (temporal_hierarchy,
factorized_posterior → NotImplementedError) while the roadmap said 45/45.
All of the above have since been fixed or implemented for real; trust file
evidence over commit messages.

## Open Work

### A2. Continuous state-space rendering (blocked on exemplar data)
`continuous_pomdp_model` (MvNormal LGSSM) is implemented and
precompile-verified, but all 3 continuous exemplars deliberately ship ONLY
discretized A/B/C/D ("discrete POMDP equivalent") — no F/H/Q/R exists in the
repo, and deriving them from discrete matrices would fabricate data.
Task: author dual parameterization into the 3 continuous exemplars
(`F=`, `H=`, `Q=`, `R=`, `prior_mean=`, `prior_cov=` — names must avoid
`[A-E]_` prefixes so the extractor passes them through), then implement
`ContinuousStrategy.generate_model_code` mirroring the flat 4-phase script
with sign-agnostic VFE validation (Gaussian Bethe FE is routinely negative;
do NOT reuse `vfe > 0` or simplex checks; validate posterior covariance PSD
instead). Detection is already structural (F/H/Q/R presence).
Acceptance: 3 continuous exemplars execute with `all_valid=true` and
mean-trajectory beliefs consumable by the analyzer.

### D1. Dirichlet parameter learning (LearningStrategy)
Port DirichletCollection/@constraints/@initialization from the deprecated
`toml_generator.py` (reference code) into a LearningStrategy + Julia model.
Detection: `learning` GNNSection or `dirichlet_[A-E]` parameter keys
(already wired). No exemplar currently declares either — add one.
Acceptance: a learning exemplar executes with softened posteriors and
learned-parameter reporting.

### D3. Native per-factor rendering (FactoredStrategy)
factorized_posterior.md currently renders as the composed joint (documented
interim). Native path: a factored @model keeping s[f] as separate
Categorical chains with per-factor B tensors.
Acceptance: per-factor posteriors emitted directly from inference (not
post-hoc marginalization), joint path retired for FACTORED.

### A1. Online active inference mode
`online_pomdp_model` alias was dropped as dead code; the real work is a
FlatStrategy variant whose generated script runs `infer()` per timestep on
`observations[1:t]` and feeds the filtered posterior into `select_action`
(replacing the hand-rolled forward Bayesian update). Precompile cache
already covers per-T specializations.
Acceptance: an `--online` render option producing filtered (not smoothed)
beliefs, validated on ≥3 exemplars.

### A3+. Native N-level hierarchical rendering
Two-level models render natively. temporal_hierarchy (3 levels) renders as
joint composition — decide level-pairing semantics (top=slow, middle+bottom
composed) and extend `hierarchical_pomdp_model` to chained contexts.

### A5+. Dashboard side-by-side compare + state-size filter
`dashboard.py` groups and filters by category; the roadmap's two-model
side-by-side compare mode and state-space-size filter remain open, as does
a deterministic test (generate against fake manifests in tmp).

### M8. GIF batch + CHANGELOG results
Re-run the 45-model 100-timestep GIF batch with the current animator (old
dark-mode GIFs moved to `~/Downloads/rxinfer_animations/superseded_dark_mode/`;
only 13 current GIFs, none with manifests). Record batch results in
CHANGELOG when complete. Unblocks full A5 data.

### FP-8 follow-up. Wire strategy layout/validation hooks
`generate_graph_layout()` / `get_validation_fields()` have no consumers:
wire `gif_animator` to `get_model_strategy(kind).generate_graph_layout(spec)`
keyed on `runtime_metadata.model_kind`, and the analyzer to
`get_validation_fields()`; until then the animator draws the flat POMDP
graph for non-flat kinds.

## Verification Commands

```bash
PYTHONPATH=src uv run --frozen pytest src/tests/render/test_rxinfer_model_strategies.py -q
PYTHONPATH=src uv run --frozen pytest src/tests/render/ src/tests/analysis/ -q
PYTHONPATH=src uv run --frozen pytest src/tests/test_zero_skip_contracts.py -q
# Live end-to-end (needs julia):
# render via render_gnn_to_rxinfer, then
# julia --startup-file=no --project=src/execute/rxinfer <script>.jl
```
