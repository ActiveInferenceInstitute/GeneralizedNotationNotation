# RxInfer Integration Improvement Roadmap

## Current State (2026-08-05)

- 45/45 GNN files render + execute through real RxInfer.jl @model + infer()
- 519 source files, 211 test files, 36 RxInfer-specific tests
- 8 framework renderers: rxinfer, pymdp, activeinference_jl, jax, numpyro, pytorch, stan, discopy
- 0 fallback paths, 0 bare except, 0 mocks
- Model distribution: 37 flat/discrete, 3 continuous, 3 hierarchical, 2 multi-agent
- GIF animations: white publication style with bayesian graph model panel
- Precompile coverage: 6 state-space configs (2,3,4,8,9,16) x 7 T values (3-30)
- Validation: belief accuracy, belief entropy, VFE convergence, per-iteration VFE trace

## MINOR (polish, tests, docs, cleanup)

### M1. GIF animator test suite
Add test_rxinfer_gif_animator.py with 8 tests: file generation, embedded data,
animation structure, different state counts, dict-shaped beliefs, empty beliefs,
white style verification, graph model node presence.

### M2. Clean up old dark-mode GIF files
Remove the first-batch GIFs from ~/Downloads/rxinfer_animations/ that used the
old dark-mode style (superseded by white publication style).

### M3. Export + CHANGELOG
Export generate_gif_animation from analysis.rxinfer.__init__.py (already done)
and add CHANGELOG entry for the GIF animation feature.

### M4. Ruff format on gif_animator.py
Run ruff format to ensure the file is properly formatted.

### M5. Docstrings for gif_animator helpers
Add proper docstrings to _parse_gnn_connections, _node_value, _draw_graph_model.

### M6. Update AGENTS.md for gif_animator
Add gif_animator.py to the analysis/rxinfer/AGENTS.md module structure section.

### M7. Update README for GIF output
Add GIF animation output to analysis/rxinfer/README.md outputs section.

### M8. CHANGELOG batch results
Add the 45-model 100-timestep GIF batch results to CHANGELOG when complete.

## MEDIUM (functional improvements, new capabilities)

### D1. Dirichlet parameter learning
Backport DirichletCollection/@constraints/@initialization from the deprecated
toml_generator.py (1321 lines of reference code). Port into the canonical
renderer as an optional 'learning' mode triggered by ModelKind.LEARNING.
This produces softer posteriors and addresses the degenerate beliefs issue.

### D2. E vector (habit/policy prior)
7/45 GNN files have E vectors but the renderer ignores them. Add E vector
support to the generated Julia code: the E vector should modulate action
selection in the forward pass (combine with EFE via log-add) and be reported
in the results JSON.

### D3. Per-factor rendering
Currently all factors are flattened into one joint HMM via
_compose_factored_pomdp(). Add a factored @model that keeps s[1], s[2], ...
as separate Categorical nodes with per-factor transition matrices. This
preserves factor structure and produces per-factor posteriors.

### D4. Multi-agent per-agent belief recovery
Agents collapse into one joint state (e.g. 256 states for 2-agent model).
Extract per-agent marginals from the joint posterior by summing over the
other agent's states. Animate each agent's beliefs separately in the GIF.

### D5. Convergence diagnostics in analyzer
VFE per-iteration is available but the analyzer doesn't compute convergence
rate, iteration-to-convergence, or VFE slope. Add these as analyzer outputs
and plot them alongside the existing free_energy plot.

### D6. EFE per-action heatmap in GIF
The current 4-panel GIF lacks the EFE landscape. Add a 5th panel or overlay
showing EFE values per action per timestep as a heatmap.

### D7. Integrate GIF generation into pipeline Step 16
Currently GIF generation is a standalone function. Wire it into
process_analysis so every RxInfer execution automatically produces a GIF
alongside the existing PNG plots.

### D8. Policy posterior visualization in GIF
The policy_posterior data is available but not animated. Add it as a
stacked area or heatmap showing how the action distribution evolves.

## MAJOR (architectural changes, new model types)

### A1. Online active inference mode
The current pipeline is offline batch (smoothing). Add an online mode where
infer() runs per-timestep and the posterior feeds back into action selection
at each step. The precompile cache already handles known T values, so the
infrastructure exists — the change is in the generated Julia code structure.

### A2. Continuous state-space models
3/45 GNN files specify continuous dynamics but are discretized. Add
Gaussian/LinearGaussian RxInfer nodes (MvNormal, NormalMeanVariance) for
true continuous inference. Requires a new @model template and a new
ModelKind.CONTINUOUS rendering strategy.

### A3. Hierarchical model support
3/45 GNN files are hierarchical but flattened. Add a hierarchical @model
with slow/fast state coupling: a slow latent modulates the fast-state
transition matrix via a conditional Categorical. Requires a new @model
template and ModelKind.HIERARCHICAL rendering strategy.

### A4. ModelKind-strategy pattern in renderer
Currently one flat @model handles everything. Refactor to a strategy
pattern keyed on ModelKind where each strategy owns its @model template,
its graph layout for animations, and its validation logic. This is the
architectural foundation for A1-A3.

### A5. Interactive HTML dashboard
A single HTML page that loads all 45 GIFs with a model selector dropdown,
grouped by category (discrete/continuous/hierarchical/multiagent/scaling),
with side-by-side comparison views and filtering by state-space size.

### A6. Cross-framework comparison
Render the same GNN file to PyMDP, RxInfer, and ActiveInference.jl, run
all three, and produce a side-by-side comparison animation showing belief
trajectories from all frameworks on the same plot.

### A7. Reproducibility manifest
Every GIF should carry an embedded JSON manifest with the GNN spec hash,
Julia version, RxInfer version, seed, timesteps, inference iterations, and
belief accuracy for full reproducibility. Store as a .json sidecar or embed
in GIF metadata.
