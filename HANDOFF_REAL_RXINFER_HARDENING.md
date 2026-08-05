# Real RxInfer.jl Hardening — Handoff Prompt

> **Hand this to a fresh agent** to fix the 5 critical issues identified by the
> Mahakala red-team review, implement the P0/P1 improvement roadmap, and ensure
> every GNN file runs genuine RxInfer.jl inference with no fallbacks, no mocks,
> no legacy paths, and no silent failures.
>
> **Repo**: `/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`
> (external AII clone — propose changes as working tree edits; the principal
> reviews and commits.)

---

## Your mandate

The Mahakala red-team review found that while real RxInfer.jl `@model` + `infer()`
is genuinely running and producing real VFE, five critical issues undermine the
scientific validity of the results. You must fix all five, implement the P0/P1
improvement roadmap, and ensure zero fallbacks exist in the codebase.

**Absolute rule: NO MOCKS, NO FAKE, NO LEGACY, NO FALLBACK.**
- The `catch e` fallback to the hand-rolled Bayesian filter must be **removed
  entirely**. If `infer()` fails, the script must **fail** (exit non-zero) and
  report the error. Never silently substitute a hand-rolled filter for real
  RxInfer inference.
- `uses_real_rxinfer` must reflect the **actual** execution, not be hardcoded.
- No `try/catch` around `infer()` that swallows errors and produces fake results.
- The deprecated `toml_generator.py` must not be called from any live path.

---

## Ground truth — verified facts (starting state, trust these)

### What works (verified by Mahakala probes + direct execution)

1. Real `@model function pomdp_model(y, A, B, D, u, T)` with `Categorical` /
   `DiscreteTransition` nodes — **CONFIRMED running**
2. Real `infer()` with `free_energy=true` — **CONFIRMED** (non-zero VFE = 6.11
   for simple_mdp, no fallback message in stdout)
3. 45/45 GNN files render and execute — **CONFIRMED** (all return rc=0)
4. Committed `Project.toml` + `Manifest.toml` pinning RxInfer 5.5.0 — **CONFIRMED**
5. Runner passes `--project=src/execute/rxinfer` — **CONFIRMED**
6. Precompilation cache (`GnnRxInferModels.jl`) reduces runtime 25-47% — **CONFIRMED**
7. Reproducible across same-seed runs — **CONFIRMED** (byte-identical)
8. Schema `rxinfer_simulation_v1` preserved — **CONFIRMED**
9. Structured logging + PNG visualizations — **CONFIRMED**

### What is broken (the 5 critical issues)

**C1. VFE trace is a fabricated constant.**
`rxinfer_renderer.py` line ~427: `final_vfe = Float64(result.free_energy[end])`
then line ~429: `push!(vfe_trace, final_vfe)` inside a `for t in 1:TIME_STEPS`
loop. RxInfer returns one VFE scalar per *iteration* (for the whole model), not
per *timestep*. The code takes the final iteration's scalar and replicates it T
times, producing a flat "time series" that plots as a constant line.

**C2. `uses_real_rxinfer` is hardcoded `true` even on fallback.**
`rxinfer_renderer.py` — the `runtime_metadata` dict contains
`"uses_real_rxinfer" => true` unconditionally. When `infer()` throws and the
catch block runs (Bayesian filter fallback), `uses_real_rxinfer` is still `true`.

**C3. Batch `infer()` is post-hoc smoothing, not active inference.**
The generated `run_simulation()` runs a hand-rolled forward filter to collect
observations/actions, then runs `infer()` on the full pre-committed sequence.
The RxInfer posteriors never feed back into action selection.

**C4. `all_valid` is tautological.**
Checks only `beliefs ∈ [0,1]`, `sum(beliefs) = 1`, `actions ∈ range` — all
guaranteed by construction. `inference_converged` and `vfe_present` are computed
but excluded from `all_valid`. Degenerate beliefs pass.

**C5. Beliefs are degenerate.**
All probability mass collapses to a single state even with non-identity A
matrices (confirmed on actinf_pomdp_agent with A = 0.9/0.05/0.05).

### Architecture state

- The renderer flattens all model types (hierarchical, multi-agent, continuous)
  into one joint categorical HMM via `_compose_factored_pomdp()`.
- The deprecated `toml_generator.py` (lines 336-550) has a complete reference
  for parameter learning: `DirichletCollection`, `@constraints`,
  `@initialization`.
- The precompile workload covers T ∈ {3,5,10,15,20,25,30,40,50,100} at 4-state
  config only — but exemplars use up to 128 states.

---

## Implementation plan

### Phase 1: Remove fallback, make infer() authoritative (CRITICAL)

**In `src/render/rxinfer/rxinfer_renderer.py`**, rewrite the `run_simulation()`
function in the generated Julia code:

1. **Remove the `try/catch` around `infer()`.** If `infer()` fails, the script
   must crash with a clear error message. Never silently fall back.

2. **Make `uses_real_rxinfer` conditional:**
   ```julia
   # After successful infer():
   "uses_real_rxinfer" => true
   # If infer() throws (no catch — script exits):
   #   the script crashes, no results JSON is written
   ```

3. **Remove the duplicated Bayesian-filter fallback** (the catch block at lines
   ~453-474 that re-runs the forward filter). This eliminates ~30 lines of
   duplicated logic.

4. **Keep the forward pass** (Phase 1: collect observations/actions using the
   hand-rolled EFE). This is necessary because the batch `infer()` needs the
   full observation/action sequence. But label it accurately in comments as
   "forward simulation for data collection" not "Bayesian filter".

### Phase 2: Fix the VFE trace (CRITICAL)

**In the generated Julia code**, change the VFE recording:

1. **Record the full per-iteration VFE vector** from `result.free_energy`:
   ```julia
   # BEFORE (wrong — one scalar replicated):
   final_vfe = Float64(result.free_energy[end])
   for t in 1:TIME_STEPS
       push!(vfe_trace, final_vfe)
   end

   # AFTER (correct — full per-iteration trace):
   vfe_per_iteration = Float64.(result.free_energy)  # length = INFERENCE_ITERATIONS
   ```

2. **Add `vfe_per_iteration` as a new field** in the results dict (separate from
   the per-step `variational_free_energy` which the analyzer expects). The
   per-iteration trace is the real convergence diagnostic.

3. **For `variational_free_energy` (per-step, consumed by the analyzer):** keep
   it as a list of length `TIME_STEPS`, but populate it with the **per-timestep
   contribution** to the free energy, not a replicated constant. If RxInfer
   doesn't provide per-timestep VFE directly, compute the marginal contribution:
   for each timestep `t`, the negative log-marginal `−log p(y[t] | y[1:t−1])`
   approximated from the posterior. If this is not tractable, set
   `variational_free_energy` to the full per-iteration vector (length =
   `INFERENCE_ITERATIONS`) and document clearly that it is per-iteration, not
   per-step. Update the analyzer to handle either length.

4. **Fix the convergence check** to use the real per-iteration trace:
   ```julia
   # Check if the last 5 iterations are within tolerance
   if length(vfe_per_iteration) >= 5
       last_5 = vfe_per_iteration[end-4:end]
       converged = (maximum(last_5) - minimum(last_5)) < 1e-4
   else
       converged = false  # too few iterations to assess
   end
   ```

### Phase 3: Strengthen validation (CRITICAL)

**In the generated Julia code**, make `all_valid` actually meaningful:

1. **Include `inference_converged` in `all_valid`:**
   ```julia
   validation["all_valid"] = validation["all_beliefs_valid"] &&
       validation["beliefs_sum_to_one"] &&
       validation["actions_in_range"] &&
       validation["inference_converged"] &&
       validation["vfe_present"]
   ```

2. **Add belief entropy check** — reject degenerate beliefs for non-identity A:
   ```julia
   function belief_entropy(belief)
       safe = max.(belief, 1e-16)
       return -sum(safe .* log.(safe))
   end

   # In validation:
   # For non-identity A matrices, check beliefs aren't degenerate
   is_identity_A = all(abs(A[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                       for i in 1:size(A,1), j in 1:size(A,2))
   min_entropy = is_identity_A ? 0.0 : 0.1  # skip for fully observable
   "belief_entropy_ok" => all(b -> belief_entropy(b) >= min_entropy, beliefs)
   ```

3. **Add `vfe_present` to validation:**
   ```julia
   "vfe_present" => !isempty(vfe_per_iteration) && all(v -> v > 0, vfe_per_iteration)
   ```

4. **Make the script exit non-zero on validation failure** (already done, but
   now validation actually has teeth).

### Phase 4: Fix the batch inference approach (HIGH)

**In the generated Julia code**, restructure `run_simulation()`:

1. **Phase 1 (forward simulation):** Run the environment forward using the
   hand-rolled EFE to collect observations, actions, and true states. This
   is the "data collection" phase. Label it clearly.

2. **Phase 2 (real RxInfer inference):** Run `infer()` on the collected data.
   No try/catch — if it fails, the script crashes.

3. **Phase 3 (posterior extraction):** Extract per-timestep posteriors from
   `result.posteriors[:s]`. The posteriors are from the joint (smoothing)
   inference — label them as "smoothed posteriors" not "filtered beliefs".

4. **Phase 4 (EFE/policy from posteriors):** Compute EFE and policy from the
   smoothed posteriors. These are post-hoc policy evaluations, not online
   control. Label accurately.

5. **Update all comments and docstrings** to accurately describe the pipeline
   as "offline batch inference (smoothing) with post-hoc policy evaluation"
   rather than "active inference".

### Phase 5: Update docs to be accurate (HIGH)

Update these files to accurately describe the pipeline:

1. **`doc/rxinfer/README.md`** — replace "active inference" with "offline batch
   inference (Bayesian smoothing) with post-hoc EFE policy evaluation"
2. **`src/render/rxinfer/AGENTS.md`** — update to describe the two-phase pipeline
3. **`src/execute/rxinfer/AGENTS.md`** — update execution description
4. **`src/analysis/rxinfer/README.md`** — note that VFE is per-iteration, not
   per-step; beliefs are smoothed posteriors
5. **`CHANGELOG.md`** — add entries for the 5 critical fixes
6. **`MAHAKALA_REVIEW.md`** — update to note which findings are now fixed

### Phase 6: Expand precompile coverage (MEDIUM)

**In `src/execute/rxinfer/src/GnnRxInferModels.jl`**:

1. **Add precompile workloads for different state-space sizes** — not just 4×4×4.
   Add 2-state, 3-state, 8-state, 9-state, 16-state configurations to cover the
   common GNN exemplar dimensions.

2. **Log precompile success/failure** — remove the bare `try/catch` in the
   `@compile_workload` block and replace with logged attempts:
   ```julia
   @compile_workload begin
       for (n_states, n_obs, n_actions) in [(2,2,2), (3,3,3), (4,4,4), (8,8,4), (9,9,5), (16,16,4)]
           # ... precompile each config
       end
   end
   ```

3. **Document that the precompile cache is machine-local** (not portable) in
   the module docstring and in `setup_environment.jl`.

### Phase 7: Typing improvements (P1)

1. **In `src/render/pomdp_contract.py`**, add `TypedDict` definitions for:
   - `CanonicalPomdpSpec` — the output of `build_canonical_pomdp_spec()`
   - `InitialParameterization` — the A/B/C/D/E matrices
   - `RxInferSimulationV1` — the output schema

2. **In `src/render/rxinfer/rxinfer_renderer.py`**, annotate the renderer methods
   with these types.

3. **Add a `ModelKind` enum** (FLAT, FACTORED, HIERARCHICAL, MULTI_AGENT,
   CONTINUOUS, LEARNING) detected from the GNN spec and carried in metadata.

---

## Gates (required before claiming done)

```bash
cd /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
export PYTHONPATH=src

# 1. Python gates
uv run --extra dev ruff check src/render/rxinfer/ src/analysis/rxinfer/ src/execute/rxinfer/
uv run --extra dev ruff format --check src/render/rxinfer/ src/analysis/rxinfer/ src/execute/rxinfer/
uv run --extra dev python -m mypy src/render/rxinfer/rxinfer_renderer.py --ignore-missing-imports

# 2. Analyzer tests (must work WITHOUT Julia — pure Python, 11 plots)
uv run --extra dev python -m pytest src/tests/analysis/test_rxinfer_analyzer_comprehensive.py -v --tb=short

# 3. Render contract tests (source inspection)
uv run --extra dev python -m pytest src/tests/render/test_rxinfer_viz_log_contract.py::test_generated_source_contains_viz_and_log_blocks -v --tb=short

# 4. Julia execution test (requires env var + Julia)
export RANDOM_SIMULATION_ENABLED=1
uv run --extra dev python -m pytest src/tests/render/test_rxinfer_viz_log_contract.py::test_rendered_script_emits_results_log_and_png -v --tb=long

# 5. Verify NO fallback exists in generated code
PYTHONPATH=src uv run --extra dev python -c "
from pathlib import Path
from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
import tempfile
gnn_file = Path('input/gnn_files/discrete/simple_mdp.md')
pomdp_space = extract_pomdp_from_file(gnn_file, strict_validation=True)
tmp = Path(tempfile.mkdtemp())
gnn_spec = POMDPRenderProcessor(tmp)._pomdp_to_gnn_spec(pomdp_space)
output = tmp / 'test.jl'
render_gnn_to_rxinfer(gnn_spec, output)
source = output.read_text()
assert 'falling back' not in source, 'FALLBACK STILL PRESENT'
assert 'catch e' not in source or 'Bayesian filter' not in source, 'FALLBACK CATCH BLOCK STILL PRESENT'
assert 'uses_real_rxinfer' in source, 'uses_real_rxinfer missing'
print('NO FALLBACK: confirmed')
"

# 6. Verify real VFE (non-constant, per-iteration)
PYTHONPATH=src uv run --extra dev python -c "
from pathlib import Path
from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
import tempfile, subprocess, json
gnn_file = Path('input/gnn_files/discrete/simple_mdp.md')
pomdp_space = extract_pomdp_from_file(gnn_file, strict_validation=True)
tmp = Path(tempfile.mkdtemp())
gnn_spec = POMDPRenderProcessor(tmp)._pomdp_to_gnn_spec(pomdp_space)
output = tmp / 'test.jl'
render_gnn_to_rxinfer(gnn_spec, output)
project = Path('src/execute/rxinfer').resolve()
r = subprocess.run(['julia', '--startup-file=no', f'--project={project}', str(output)],
    cwd=str(tmp), capture_output=True, text=True, timeout=600)
assert r.returncode == 0, f'Julia failed: {r.stderr[-300:]}'
res = json.loads((tmp / 'simulation_results.json').read_text())
vfe = res['variational_free_energy']
assert len(vfe) > 0, 'VFE empty'
assert all(v > 0 for v in vfe), f'VFE has zeros: {vfe[:5]}'
assert res['validation']['all_valid'] == True
assert res['validation']['inference_converged'] == True
assert res['validation']['vfe_present'] == True
assert res['runtime_metadata']['uses_real_rxinfer'] == True
print(f'VFE: {vfe[:5]}')
print(f'converged: {res[\"validation\"][\"inference_converged\"]}')
print(f'uses_real_rxinfer: {res[\"runtime_metadata\"][\"uses_real_rxinfer\"]}')
print('REAL RXINFER: confirmed')
"

# 7. Zero-skip contract
uv run --extra dev python -m pytest src/tests/test_zero_skip_contracts.py -q --tb=short

# 8. Doc gates
uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

# 9. Execute 3 representative GNN files end-to-end
PYTHONPATH=src uv run --extra dev python -c "
from pathlib import Path
from gnn.pomdp_extractor import extract_pomdp_from_file
from render.pomdp_processor import POMDPRenderProcessor
from render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer
import tempfile, subprocess, json

gnn_dir = Path('input/gnn_files')
test_files = ['discrete/simple_mdp.md', 'discrete/actinf_pomdp_agent.md', 'discrete/hmm_baseline.md']
tmp = Path(tempfile.mkdtemp())
project = Path('src/execute/rxinfer').resolve()

for tf in test_files:
    gnn_file = gnn_dir / tf
    pomdp_space = extract_pomdp_from_file(gnn_file, strict_validation=True)
    gnn_spec = POMDPRenderProcessor(tempfile.mkdtemp())._pomdp_to_gnn_spec(pomdp_space)
    run_dir = tmp / gnn_file.stem
    run_dir.mkdir(exist_ok=True)
    output = run_dir / f'{gnn_file.stem}_rxinfer.jl'
    render_gnn_to_rxinfer(gnn_spec, output)
    r = subprocess.run(['julia', '--startup-file=no', f'--project={project}', str(output)],
        cwd=str(run_dir), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f'{tf} failed: {r.stderr[-300:]}'
    res = json.loads((run_dir / 'simulation_results.json').read_text())
    vfe = res['variational_free_energy']
    assert all(v > 0 for v in vfe), f'{tf} VFE has zeros'
    assert res['validation']['all_valid'] == True
    assert res['runtime_metadata']['uses_real_rxinfer'] == True
    assert 'falling back' not in r.stdout, f'{tf} used fallback!'
    print(f'PASS: {tf} VFE={len(vfe)} valid={res[\"validation\"][\"all_valid\"]} rx={res[\"runtime_metadata\"][\"uses_real_rxinfer\"]}')
"
```

---

## Key files reference

| File | Role |
|------|------|
| `src/render/rxinfer/rxinfer_renderer.py` | **Canonical renderer** — fix C1-C5 here |
| `src/execute/rxinfer/src/GnnRxInferModels.jl` | **Precompiled model** — expand precompile coverage |
| `src/execute/rxinfer/Project.toml` | **Julia environment** — may need new deps for learning |
| `src/execute/rxinfer/Manifest.toml` | **Pinned manifest** — regenerate if deps change |
| `src/analysis/rxinfer/analyzer.py` | **Step-16 analyzer** — update for per-iteration VFE |
| `src/render/pomdp_contract.py` | **POMDP spec** — add TypedDicts |
| `src/render/rxinfer/toml_generator.py` | **Reference for Dirichlet/learning** — do NOT call, read only |
| `src/tests/render/test_rxinfer_viz_log_contract.py` | **Tests** — update assertions |
| `src/tests/analysis/test_rxinfer_analyzer_comprehensive.py` | **Analyzer tests** — may need VFE length updates |
| `MAHAKALA_REVIEW.md` | **Red-team report** — update with fix status |
| `RXINFER_ARCHITECTURE_ROADMAP.md` | **Architecture roadmap** — reference |
| `RXINFER_PREMORTEM_REVIEW.md` | **Premortem report** — reference |

---

## Deliverable

The agent must produce:

1. **No fallback** — `infer()` failure crashes the script, no silent substitution
2. **`uses_real_rxinfer`** conditional on actual `infer()` success
3. **Per-iteration VFE trace** — not a replicated constant
4. **Strengthened `all_valid`** — includes convergence, VFE presence, belief entropy
5. **Accurate docs** — "offline batch inference (smoothing)" not "active inference"
6. **Expanded precompile** — covers common state-space sizes
7. **TypedDicts** for `CanonicalPomdpSpec`, `RxInferSimulationV1`
8. **All tests passing** including Julia execution
9. **3 representative GNN files verified** end-to-end with real VFE
10. **A brief diff summary** of what changed and why

**Do not** keep any fallback path anywhere in the codebase. The point is to
ship real RxInfer or ship nothing. If `infer()` fails, the script fails, the
test fails, and the principal investigates — never silently substitute.