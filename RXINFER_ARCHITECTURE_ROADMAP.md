# RxInfer Pipeline — Architecture, Modularity & Extensibility Roadmap

Review date: 2026-08-04
Scope: `src/render/rxinfer/rxinfer_renderer.py` (900 lines),
`src/execute/rxinfer/src/GnnRxInferModels.jl` (48 lines),
`src/analysis/rxinfer/analyzer.py` (725 lines),
`src/render/rxinfer/toml_generator.py` (1320 lines, deprecated reference),
`src/render/pomdp_contract.py` (285 lines),
`src/render/pomdp_processor.py`, and 47 exemplars under `input/gnn_files/`.

---

## Executive summary

The pipeline is **correct and well-tested for the flat single-modal discrete
POMDP case**, but it is architecturally a **single-flat-model vertical slice**.
Every observed capability gap traces back to one design decision: the renderer
routes **all** model types through `build_canonical_pomdp_spec()` →
`_compose_factored_pomdp()`, which **flattens** every hierarchical / multi-agent
/ factorized exemplar into one joint categorical HMM with a single hidden-state
factor `s`, a single observation modality `y`, discrete `Categorical`/
`DiscreteTransition` nodes, and **no parameter learning**. There is exactly one
hardcoded `@model` (`GnnRxInferModels.pomdp_model`). The richer `@model` with
`DirichletCollection` / `@constraints` / `@initialization` that parameter
learning needs already exists — but only in the **deprecated** `toml_generator.py`.

The single most impactful improvement is **de-flattening the model layer**:
stop collapsing factors/agents/hierarchy into one joint matrix and instead let
the renderer emit structurally-aware `@model` definitions (per-factor,
per-agent, hierarchical, with Dirichlet parameter priors). That one change
upgrades composability, statistics, and visualization simultaneously.

---

## How the current pipeline actually works (ground truth)

1. **`extract_pomdp_from_file`** parses a GNN `.md` into a `POMDPSpace`. For
   factored files (multi-agent, hierarchical, factorized-posterior) this holds
   per-factor matrices `A_agent1`, `B_level1`, `s_joint`, etc.
2. **`POMDPRenderProcessor._pomdp_to_gnn_spec`** → **`_compose_factored_pomdp`**
   tensor-multiplies all `A_*`/`B_*`/`C_*`/`D_*` factors into a **single joint**
   `A/B/C/D`. Multi-agent coordination and hierarchical cross-level structure
   are algebraically collapsed (e.g. the 2-agent model → one 16-state joint
   factor; the 2-level hierarchy → one 8-state joint factor).
3. **`build_canonical_pomdp_spec`** normalizes to `canonical_pomdp_v1` with a
   single flat `A,B,C,D`.
4. **`RxInferRenderer._generate_canonical_rxinfer_code`** builds a ~580-line
   Julia script as **one giant f-string** that calls the single precompiled
   `@model pomdp_model(y, A, B, D, u, T)`.
5. **`GnnRxInferModels.jl`** defines that one flat model + a `PrecompileTools`
   workload.
6. **`analyzer.create_rxinfer_visualizations`** normalizes the resulting flat
   belief/obs/EFE arrays and emits up to 11 plots, all derived from the same
   flattened output.

**Key consequence:** files *named* "continuous", "hierarchical", "multi-agent"
all render/execute through the **identical** flat discrete categorical code
path. The `_extract_declared_rxinfer_agents` + `build_rxinfer_execution_metadata`
functions (renderer lines 799–891) extract agent topology from `toml_generator`
**only to write a `.metadata.json` sidecar** — they do **not** change the
generated model code.

---

## Findings by review axis

### 1. Model composability — currently LOW
- **(a) Multiple state factors** — ✗ not supported. The `@model` has a single
  `s` factor; factored exemplars are flattened. RxInfer.jl supports
  `s[1][t]~...; s[2][t]~...` with per-factor dynamics natively, so this is
  achievable.
- **(b) Hierarchical structure** — ✗ flattened. Two-level dynamics (fast/slow)
  collapse into one joint transition. Native hierarchical RxInfer models
  (slow latent modulating fast-state dynamics) are real and well-supported.
- **(c) Continuous observations / states** — ✗ the "continuous" exemplars are
  hand-discretized POMDPs. No Gaussian/LinearGaussian nodes; the model is
  strictly categorical.
- **(d) Multi-agent coordination** — ✗ agents collapse into one joint state
  factor; per-agent beliefs/policies are unrecoverable downstream. `nr_agents`
  metadata exists but never drives codegen.

**Composability is limited by the flat collapse in step 2, not by RxInfer.jl.**

### 2. Visualization — 11 plots but low information content
The 11 types (`belief_evolution`, `obs_vs_true`, `belief_heatmap`,
`belief_entropy`, `accuracy`, `action_frequencies`, `belief_convergence`,
`belief_trace`, `free_energy`, `observations`, `efe_per_action_heatmap`) all
derive from **one flattened trajectory**. Missing:
- **Posterior predictive checks** (re-draw observations from posterior, overlay
  on true observations).
- **Model comparison** (evidence / free-energy bound across model families).
- **Real convergence diagnostics** — the VFE trace is flattened to a constant
  per step (`renderer` lines ~425–430 push the single final-iteration value for
  every step), so `free_energy` and `belief_convergence` plots show step-wise
  constants, **not** an iteration convergence curve. The per-iteration trace is
  discarded.
- **Uncertainty quantification** — only scalar belief entropy; no credible
  intervals (impossible until parameters are inferred) and no prediction bands.
- **Multi-agent / hierarchical breakout** — lost because data is flattened
  (per-agent belief panels, per-level states).
- No **action/EFE comparison across policies** beyond a per-action heatmap;
  no **preference-fit / goal-reach** summary metric plot.

So the *count* (11) looks sufficient but the *axis coverage* is thin.

### 3. Typing — `Dict[str, Any]` everywhere
`gnn_spec`, the canonical spec, and `initialparameterization` flow as untyped
`Dict[str, Any]` through renderer, contract, metadata, and analyzer. Because the
spec is fundamentally a **JSON document** (deep-copied, mutated, base64-embedded
into the generated script and JSON-parsed back), **`TypedDict` is the right
tool**, not dataclasses — it documents the `canonical_pomdp_v1` shape at zero
conversion cost and stays round-trippable. Dataclasses are the better fit for
internal *renderer configuration* (options, model-family strategy objects) that
never crosses the JSON boundary.

### 4. Statistical extensions — absent as a category
- Parameter learning (unobserved A/B with priors) — not implemented in the live
  model; only in deprecated `toml_generator.py`.
- Posterior predictive checks — absent.
- Model comparison (evidence, BIC/WAIC bayes-factor style) — absent; only the
  single hardcoded model family.
- Convergence — a crude `abs(FE[end]-FE[end-1]) < 1e-4`; full per-iteration
  trace is discarded (see #2).

### 5. Extension points for parameter learning — LOW effort to add (it already exists)
`toml_generator.py` lines ~336–550 already contains a **complete, correct**
reference implementing exactly this:
- `const p_A = DirichletCollection(...); const p_B = DirichletCollection(...)`
- `A ~ p_A; B ~ p_B` inside `@model`
- `@initialization q(A) = DirichletCollection(diageye(num_obs) .+ 0.1)`,
  `q(B) = DirichletCollection(ones(num_states, num_states, num_actions))`
- `@constraints` mean-field factorization separating `q(A)`/`q(B)`.

Bringing this into the **live** renderer is a backport, not green-field work:
add Dirichlet collection helpers (already present at `toml_generator.py`
`_create_dirichlet_prior*`), a `learning`-mode model template, and an
`@initialization`/`@constraints` block in the emitted script. Estimated ~1–2
dev-days plus Julia precompile for the learning model.

### 6. Refactoring — one god function + duplicated Julia logic
- **`_generate_canonical_rxinfer_code`** (renderer lines 139–720, ~580 lines)
  is a god function returning one giant f-string with ~50 `{...}` slots. It is
  untestable in Python, unreviewable, and mixes model-gen + EFE-logic + plotting
  + logging + JSON schema into one template.
- The **generated Julia `run_simulation`** is a ~150-line god function with the
  Bayesian-filter forward pass **written twice** (Phase 1 lines ~341–368 and the
  catch-fallback lines ~453–474) — duplicated logic at high risk of drift.
- The renderer **class** mixes codegen + metadata + file writing + SHA256 +
  agent-topology parsing, and imports `toml_generator`'s *private* helpers
  (`_extract_agent_topology`, `_extract_compact_agents`, etc.).
- Coupling to `POMDPRenderProcessor._pomdp_to_gnn_spec` — a private method used
  cross-module.
- **No templating library** (Jinja2 absent); the entire script is string
  interpolation.
- `analyzer.py` has 10 near-identical try/except/`plt.savefig`/`close` blocks
  with copy-pasted error handling — a prime candidate for a small plot-helper
  (`_emit_figure(title, filename, build_fn)`).

---

## Prioritized improvement roadmap

### P0 — Correctness & internal quality (low effort, high value)
1. **Record the real per-iteration VFE/convergence trace.** Stop flattening the
   final-iteration free energy to a constant per step and emit the actual
   iteration trace. This unblocks *every* convergence diagnostic and fixes a
   currently-misleading plot. *(renderer generated-Julia: run_simulation + main)*
2. **Split the god f-string.** Extract the Julia script into a proper template
   (Jinja2 or a fastapi-style static template) and split the generated source
   into logical Julia functions — model, EFE, plotting, logging. Makes the
   rendered code testable, reviewable, and diffable.
3. **Deduplicate the Bayesian-filter fallback** in generated Julia (one forward
   filter function used by both the main path and the catch path).
4. **Collapse the 10 copy-pasted plot blocks** in `analyzer.py` into one
   `_emit_figure` helper (keeps the 11-plot surface, removes ~50 duplicated
   try/except lines).

### P1 — Typing & contract hardening (small, safe)
5. **Add `TypedDict`s** for `CanonicalPomdpSpec`, `InitialParameterization`,
   and `RxInferSimulationV1` in `pomdp_contract.py`; annotate renderer/analyzer
   parameter lists with them. Use **dataclasses** for internal `RenderOptions`
   and any model-family strategy object (no JSON boundary).
6. **Add a `ModelFamily`/`ModelKind` enum** (FLAT, FACTORED, HIERARCHICAL,
   MULTI_AGENT, CONTINUOUS, LEARNING) detected from the GNN spec and carried in
   metadata, so downstream tools stop guessing.

### P2 — Model composability (the strategic change)
7. **Introduce a strategy pattern** keyed on `ModelKind` in the renderer
   (`ModelTemplate` strategies: flat / factored / hierarchical / multi-agent /
   learning). Each strategy owns its `@model` template + the Julia run/EFE/plot
   sections. Replace the current single hardcoded path. This is the extension
   point that makes everything below additive instead of adversarial.
8. **Backport parameter learning** (Dirichlet `A`, `B` + `@initialization` +
   `@constraints`) from `toml_generator.py` into a `learning` strategy — closes
   the extension gap in the live renderer.
9. **Emit per-factor / per-agent models instead of flattening.** Reuse the joint
   flat path as the default, but let factored/hierarchical/multi-agent exemplars
   render structurally-aware `@model` definitions (multiple `s[i]` factors,
   cross-level couplings, per-agent transition/likelihood). Keep `_compose_factored_pomdp`
   as a compatibility fallback, not the only path.
10. **Add continuous (Gaussian) state/observation support** — a `continuous`
    strategy with `LinearGaussian`/`Normal` nodes so genuinely continuous
    exemplars (not just discretized equivalents) can render.
11. **Support multiple observation modalities** in the `@model` (currently
    single `y`).

### P3 — Visualization & statistics (rides on P2 data)
12. **Posterior predictive checks** — sample from posterior predictive, overlay
    on true observations (needs inference results that carry draws).
13. **Model comparison** — per-exemplar evidence / bound and a cross-exemplar
    comparison plot (needs a model family axis, hence P2).
14. **Convergence diagnostics plots** — per-iteration free-energy curve, stability
    band (needs P0-1 trace) and, once learning exists, parameter-posterior
    credible intervals.
15. **Multi-agent / hierarchical breakout plots** — per-agent beliefs/EFE panels,
    per-level state traces, cross-agent coordination heatmap (needs per-factor
    output that P2-9 preserves).

---

## Suggested sequencing
- **Week 1 (P0):** items 1–4. Low risk, immediately useful, improves the 45-file
  regression surface.
- **Week 2 (P1):** items 5–6. Establishes typed contracts before new strategies.
- **Weeks 3–4 (P2):** items 7–9 first (strategy pattern → learning backport →
  de-flattening). Parameter learning (item 8) is the cheapest win and can ship
  before full de-flattening.
- **P3/P4:** items 10–15 as downstream consumers of P2 data.

## Risks / notes
- Adding strategies changes the precompile workload in `GnnRxInferModels.jl`
  (each new `@model` shape needs a `@compile_workload` entry to avoid ~85s JIT
  per run — see the existing T-loop pattern).
- De-flattening changes the emitted `rxinfer_simulation_v1` schema (per-factor
  `beliefs_by_factor` will actually be populated). Update `analyzer` normalizers
  and the downstream report/LLM steps accordingly — the analyzer is already
  hardened to tolerate dict-shaped keys, so this is partially buffered.
- The deprecated `toml_generator.py` is the authoritative source for the
  Dirichlet/`@constraints`/`@initialization` patterns — do not delete it until
  the learning strategy lands.
