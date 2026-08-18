# RxInfer Integration Improvement Roadmap

## Current State (2026-08-07, wave 2)

Verified by execution (pinned in `src/tests/render/test_rxinfer_model_strategies.py`;
full suite 2,796 passed / 0 failed):

- **29/29 GNN files render** through the real pipeline path; kind taxonomy is
  structural: 20 flat, 3 continuous, 2 hierarchical, 1 learning, 2 multi-agent,
  1 factored. (Corpus curated 2026-08-07: 17 redundant files pruned — 14
  scaling-study T-variants differing only in a runtime constant, the three
  21 MB generated N128 specs, N2, and factor_graph_model, whose factored
  coverage is owned by factorized_posterior. The scaling sweep regenerates
  its full grid from scripts/pymdp_scaling_config.yaml on demand; a
  representative N4–N64 T100 ladder stays in the corpus. −67 MB.) Detection reads typed fields only (GNNSection, per-level/per-agent
  matrix keys, explicit counts, F/H/Q/R presence, `dirichlet_*` keys).
- **Every ModelKind has a live generator** — the stub base class is gone:
  - FLAT: canonical batch generator, plus an **online mode** (A1) — per-timestep
    `infer()` on the observation prefix, filtered posterior drives EFE+habit
    action selection; selected via ModelParameters `inference_mode: online` or a
    render option. Executed live: `all_valid=true`, accuracy 0.88.
  - HIERARCHICAL (A3): native two-level model (context latent → fast prior via
    `A_level2`, mean-field + initialization); 3+ levels render as the documented
    joint composition. Executed live: `all_valid=true`.
  - FACTORED (D3): native mean-field two-factor model — multi-parent likelihood
    `DiscreteTransition(s1, A_m0, s2)`, per-factor chains and posteriors,
    posterior family = the exemplar's own declared `Q(s_f0)Q(s_f1)`. Executed
    live: `all_valid=true`, per-factor MAP accuracy 1.0.
  - CONTINUOUS (A2): native LGSSM from authored dual parameterization — the 3
    continuous exemplars now carry faithful F/H/Q/R/prior blocks derived from
    their own prose formulations (discrete A/B/C/D retained for the other
    frameworks). Beliefs are posterior means (+ `posterior_cov`), VFE validation
    is sign-agnostic (Gaussian Bethe FE is routinely negative). All 3 executed
    live: `all_valid=true` (nav rmse 0.166).
  - LEARNING (D1): native Dirichlet likelihood learning — `A` is a latent
    `DirichletCollection` from `dirichlet_A` pseudo-counts; environment emits
    through true A while the agent acts through the prior mean. New exemplar
    `learning/dirichlet_likelihood_learning.md`. Executed live:
    `all_valid=true`, learned-A distance 0.178 → 0.052, and
    `a_learning_improved` is a hard gate (FE alone converges happily on
    label-switched optima; only the distance metric catches them).
  - MULTI_AGENT: documented joint composition stamping its true kind;
    per-agent marginals recovered downstream (D4).
- **Timing alignment fix**: `true_states[t]` now records the state that
  EMITTED observation t (was the post-transition state — an off-by-one
  inherited from the original generator, masked by persistent-B exemplars).
  Discrete belief-accuracy metrics are now aligned comparisons (verified 1.0
  on structured exemplars).
- **Strategy hooks live** (FP-8): the GIF animator resolves its graph layout
  via `get_model_strategy(kind).generate_graph_layout()`, and the analyzer
  summarizes strategy-declared validation fields via `get_validation_fields()`.
  Every registered strategy implements both.
- **Dashboard (A5)**: category filter + state-space-size buckets + side-by-side
  compare mode; deterministic test (`test_rxinfer_dashboard.py`).
- **Continuous results contract**: `state_factors`/`observation_modalities`
  echo EMPTY for continuous models — the discrete dual parameterization does
  not describe the continuous latent, and echoing it made per-factor recovery
  raise (caught live, fixed). Template note: the echo is only safe for
  strategies sharing the discrete factorization.
- Cross-framework comparison (A6), E habit prior (D2), per-factor recovery
  (D4), structural detection, and the zero-skip/entropy-gate hardening carried
  over from the 2026-08-05/07 wave-1 work.

### Corrections to earlier claims (2026-08-05 red-team audit)

Commit `16d3cb25`'s message over-claimed: M6/M7 (docs), D3 (`detect_factors`),
and D4 (`per_agent_beliefs`) described as done were absent from the tree, and
the strategy-pattern refactor regressed 2/45 renders while the roadmap said
45/45. All fixed or implemented for real since; trust file evidence over
commit messages.

## Open Work

### Optional T=100 precompile workloads

The completed M8 batch is recorded in the current-state evidence above. If
T=100 batch rendering becomes routine, benchmark extending `@compile_workload`
with representative T=100 entries and record the deliberate
precompile-time/runtime tradeoff before adopting them.

### A3+. Native N-level hierarchical rendering (decision recorded)
Two-level models render natively. For 3+ levels (temporal_hierarchy), the
declared cross-level semantics (per-level C/D modulation at distinct
timescales) do not map onto context-chain prior-coupling without deriving
matrices the files don't declare — deriving them would fabricate data. DECIDED:
3+-level models render as the joint composition (stamped `hierarchical`) until
an exemplar declares explicit composed coupling. Revisit only with new
exemplar data.

### Dashboard browser verification
The A5 compare/filter features are covered by deterministic HTML tests. A
real-browser pass (screenshot + DOM read, accessibility checks) was run
2026-08-18 against the representative fixture GIF set (four models covering
all four size buckets and categories) using headless Chrome for Testing 145:
screenshot captured (1280×900 PNG), all four model cards and every filter /
compare control render in the DOM, and the basic a11y surface is sound
(`lang="en"`, viewport meta, a labelled `<select>` for every control, `alt`
text on every card image, a single `<h1>`). The production 29-GIF set was not
present in this environment; re-run against it when the batch is regenerated.

## Verification Commands

```bash
PYTHONPATH=src uv run --frozen pytest src/tests/render/test_rxinfer_model_strategies.py -q
PYTHONPATH=src uv run --frozen pytest src/tests/render/ src/tests/analysis/ -q
PYTHONPATH=src uv run --frozen pytest src/tests/test_zero_skip_contracts.py -q
# Live end-to-end (needs julia):
# render via render_gnn_to_rxinfer, then
# julia --startup-file=no --project=src/execute/rxinfer <script>.jl
```
