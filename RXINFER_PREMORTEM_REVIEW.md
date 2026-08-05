# Premortem + Assumption-Surfacing Review — GNN × RxInfer.jl Integration

**Date:** 2026-08-04 · **Scope:** repo `GeneralizedNotationNotation` RxInfer integration,
files: `src/render/rxinfer/rxinfer_renderer.py`, `src/execute/rxinfer/{rxinfer_runner.jl,
rxinfer_runner.py, Project.toml, Manifest.toml, src/GnnRxInferModels.jl}`, committed
artifacts under `output/11_render_output/.../rxinfer` and `output/12_execute_output/.../rxinfer`,
and the acceptance gates `src/tests/pipeline/test_pomdp_gridworld_cross_framework.py` +
`scripts/check_pomdp_gridworld_outputs.py`.

**Framing premise honored:** we assumed the project FAILED in production (silently —
no crash, just wrong/meaningless results) and worked backward. Every finding below was
verified against the actual code/artifacts, not inferred.

---

## Part 1 — Verified findings (the "already-failed" reality)

These are observable facts, not hypotheses. They are why production failed silently.

1. **The committed "RxInfer" artifact never runs RxInfer.** The git-tracked rendered
   script `output/11_render_output/pomdp_gridworld_3x3/rxinfer/POMDP GridWorld 3x3_rxinfer.jl`
   contains **no `@model` and no `infer()` call**. It is a hand-rolled Bayesian filter
   + hand-rolled EFE in pure Julia. Its `simulation_results.json` has
   `variational_free_energy = []`, no `uses_real_rxinfer`, no `script_sha256`. The *current*
   renderer (`rxinfer_renderer.py`) DOES emit a genuine `@model`/`infer()` script (added in a
   later commit), so **the committed evidence is stale relative to the claimed capability** —
   the docs claim "genuine @model + infer() VMP inference" but the committed proof-of-execution
   is the old simulator. Only **1** of the 47 exemplars has committed rxinfer output; the other
   44 are gitignored runtime artifacts, so the "45/45 render + execute" claim is evidenced for 1.

2. **`all_valid` is vacuously true — not load-bearing.** In both the hand-rolled and genuine
   versions: `all_valid = beliefs∈[0,1] ∧ sum(beliefs)=1 ∧ actions∈range`. Each of these is
   *guaranteed by construction* (beliefs are normalized, actions are `Categorical` samples
   decremented). A **degenerate point-mass belief** (all mass on one, possibly *wrong*, state)
   passes. Belief *correctness vs. the true state* is never checked in the genuine path.
   `inference_converged` and `vfe_present` are computed but **excluded from `all_valid`**.

3. **Fallback is indistinguishable from success; metadata lies.** In the genuine renderer the
   `catch` block (RxInfer failing) fills beliefs from the forward filter, sets `vfe_trace=NaN`,
   `inference_converged=false` — yet `all_valid` stays `true`. Worse, `runtime_metadata
   ["uses_real_rxinfer"]` is **hardcoded to `true` unconditionally**, so even a full fallback run
   reports "real RxInfer ran." The only tell is manual inspection of NaN VFE / `converged=false`;
   no gate checks it.

4. **The acceptance test never executes inference.** `_assert_julia_parse` only runs
   `Meta.parseall(...)` (syntax check) in a throwaway `/tmp/julia_test_env`; `_assert_julia_packages`
   only `using`-loads packages. The gridworld gate (`check_pomdp_gridworld_outputs.py`) asserts
   only: `success is True`, `num_timesteps==15`, `B_shape==[9,9,5]`, `all_valid is True`. **No
   test runs real `infer()` on any of the 45/47 scripts.** A hand-rolled filter — or a fully
   fallback run — passes every gate.

5. **Batch `infer()` is post-hoc offline smoothing, not online control.** The `pomdp_model`
   takes the *full action sequence `u`* as known inputs. Actions were chosen by the hand-rolled
   EFE forward pass; `infer()` then smooths the already-recorded trajectory. It never selects an
   action and never closes the control loop. The RxInfer posteriors are **decorative** relative to
   control.

6. **The VFE "trace" is a fabricated constant.** RxInfer returns one `free_energy` scalar *per
   iteration for the whole model*. The code takes `free_energy[end]` and **replicates it across
   all TIME_STEPS** as `variational_free_energy` — producing a flat, meaningless "time series"
   that plots as a constant line. Plausible-looking, wrong.

7. **EFE formula is unvalidated.** `compute_efe` = ambiguity + risk(KL(predicted_obs ‖ C_pref)),
   single-step myopic, min-selected via `softmax(-prec·EFE)`. Sign convention is standard, but there
   is **no cross-framework reference check (e.g., vs pymdp) and no test asserting numeric
   correctness**. A sign/scale error yields finite plausible decimals and passes silently.

8. **Precompilation cache mis-invested and silently fragile.** `GnnRxInferModels.jl` precompiles
   `pomdp_model` only for T∈{3,5,10,15,20,25,30,40,50,100} at 4-state config. Its `@compile_workload`
   loop wraps every `infer()` in a bare `try/catch` (no logging), so a **failed precompile is
   silent** and the advertised "~80s saved per run" disappears undetected. PrecompileTools caches are
   machine- and version-keyed and **never committed** → the "cache" is not portable; each machine
   rebuilds (5–10 min). A Julia/RxInfer upgrade or a T-value outside the list silently invalidates it
   and JIT recompiles per run. Currently the precompile is **dead code w.r.t. the committed artifact**
   (which never loads `GnnRxInferModels`).

9. **Committed Manifest is machine-pinned, not portable.** `Manifest.toml` records
   `julia_version="1.12.6"` with stdlibs pinned `1.11.0`/`1.12.6`, while `Project.toml` compat says
   `julia="1.10"`. On a machine with Julia 1.10/1.11 the pinned stdlib versions don't exist and Pkg
   re-resolves or fails; JLL binary artifacts are OS/arch-specific. The runner's `safe_require`
   would silently `Pkg.add` (network, mutates env). Not portable across Julia versions or OS.

10. **Stale/deprecated runner divergence.** `rxinfer_runner.jl` (TOML path) defines its own
    simplified `@model` that **ignores actions** (`s[t] ~ Categorical(B_mat[:,1])`, Gaussian
    `MvNormal` obs) — inconsistent with the genuine model — and its fallback repeats the prior `D`
    with **no marker** while reporting "accuracy" as `maximum(belief)` (near-1 regardless). A second,
    divergent "RxInfer" path that also looks successful in the worst case.

---

## Part 2 — Ranked failure-mode catalog (plausibility × impact)

Severity = P × I where `P` plausibility (evidence), `I` impact (silent-wrong vs crash vs cosmetic).
Scale 1–5 each; **rank = product**.

| # | Failure mode | How it manifests | Evidence | P | I | Rank | Detection |
|---|---|---|---|---|---|---|---|
| 1 | No real `infer()` ever runs (hand-rolled / fallback only) | Outputs are a stale or fallback Bayesian filter presented as RxInfer; VFE empty | Committed `.jl` has no `infer()`; `vfe=[]`; `uses_real_rxinfer` absent then hardcoded `true` | 5 | 5 | **25** | **None in gate** — must grep for `infer(` + check VFE |
| 2 | Validation is tautological → degenerate/wrong beliefs pass | Point-mass (possibly wrong) beliefs report `all_valid=True`, high "confidence" | `all_valid` only checks range/sum/action-range, all guaranteed by construction | 5 | 5 | **25** | None — no belief-vs-truth assert |
| 3 | Fallback never tagged; metadata always says real RxInfer | Full silent degradation with no signal; users trust results | Catch sets NaN VFE but `uses_real_rxinfer=true` unconditional | 4 | 5 | **20** | Only by spotting NaN VFE / `converged=false` |
| 4 | Docs claim features committed evidence doesn't demonstrate | "45/45 render+execute", "genuine infer()", "real VFE" overclaim | 1/47 exemplars committed; commit artifact is old simulator; test never runs infer() | 5 | 4 | **20** | Doc-vs-artifact audit |
| 5 | Batch infer is post-hoc — no active-inference control loop | "Active Inference agent" is really passive smoothing of a pre-committed trajectory | Model takes full `u`; actions chosen by hand-rolled forward EFE; infer never affects actions | 5 | 4 | **20** | Conceptual / architectural review |
| 6 | VFE "trace" fabricated (constant replicated) | Misleading per-step free-energy curve; wrong analysis/plots | `push!(vfe_trace, final_vfe)` inside a TIME_STEPS loop | 5 | 3 | **15** | Inspect `variational_free_energy` for constancy |
| 7 | EFE formula wrong-but-plausible | Actions produced by incorrect one-step EFE; no reference check | No cross-framework numeric assert; myopic single-step | 4 | 4 | **16** | Add reference (pymdp) equality test |
| 8 | Precompile cache silent staleness / T-mismatch | Advertised speedup silently lost; per-run JIT blowup | Bare `try/catch`; T list incomplete; cache not portable | 4 | 3 | **12** | Log precompile success; benchmark T coverage |
| 9 | Manifest/Julia-version lock breaks other machines | Resolve failure or `Pkg.add` network mutation on 1.10/1.11/other OS | `julia_version=1.12.6`, stdlib 1.12.6, compat julia=1.10 | 3 | 4 | **12** | CI matrix over Julia versions/OS |
| 10 | Only gridworld checked in gates; rest unverified at commit | Non-gridworld exemplars may crash/misrender and never block | 1 committed artifact; gate hardcodes gridworld shapes | 4 | 3 | **12** | Gate all 45 committed outputs |
| 11 | `rxinfer_runner.jl` deprecated path diverges & auto-installs | Second, different (action-ignoring) model surface; env mutation | Simplified `@model`; `Pkg.add` in `safe_require` | 3 | 3 | 9 | Delete or align |
| 12 | Cross-machine result non-reproducibility | Seed fixed, but fallback/VFE/version state unflagged; runtime outputs gitignored | `generated_at` stamps; builds differ; manifests pinned | 2 | 4 | 8 | Pin Julia+artifacts; flag fallback |

**Top-3 in production (the actual killers):** (1) real RxInfer may never have run — committed
proof is the hand-rolled simulator; (2) the acceptance gate validates only tautologies, so any
fallback/degenerate run passes; (3) the control claim is false — infer() is post-hoc smoothing,
so "Active Inference" results are not what they claim even when the pipeline "passes."

---

## Part 3 — Assumption table with load-bearing risk

Each row: the implicit assumption, why it's load-bearing, and its risk given the evidence.

| # | Implicit assumption | Load-bearing for | Risk assessment |
|---|---|---|---|
| A1 | Forward-pass Bayesian filter yields the same observations/actions a *real* Active Inference agent would | The whole "simulation" claim | **HIGH (violated).** Filter + EFE are hand-rolled; RxInfer `infer()` is post-hoc and never selects actions. There is no active-inference policy loop. |
| A2 | Batch `infer()` adds information beyond the forward filter | Justifying the expensive RxInfer step at all | **MEDIUM-HIGH.** True only in the weak sense of Bayesian *smoothing* (future-observation conditioning). Not used for control; the added info is recorded, not acted on. |
| A3 | VFE from `infer()` is meaningful for Active Inference | Gradients/plots/convergence reporting | **HIGH (violated).** `variational_free_energy` is one scalar replicated PER STEP — not a per-step VFE. Never fed back; convergence check is on this replicated value. |
| A4 | EFE formula is correct for POMDPs | Every action chosen in every run | **MEDIUM-HIGH (unverified).** Single-step ambiguity+risk is defensible, but has no reference cross-check; a wrong-but-finite formula passes every suite. |
| A5 | Convergence check is meaningful | "converged" telemetry, `inference_converged` flag | **MEDIUM-HIGH (misleading).** Measures last-iteration Δ of a whole-model scalar; set `true` when trace length <2; excluded from `all_valid`; fabricated per-step. |
| A6 | Precompilation cache is portable | Runtime ~80s×N savings; reproducibility of timings | **HIGH (false).** PrecompileTools caches are machine/version-keyed, never committed; silent `try/catch` hides failures. Benefit evaporates per machine/upgrade. |
| A7 | Results reproducible across machines | Scientific validity of outputs | **MEDIUM-HIGH.** Seed 42 fixed, but fallback/VFE/version state is unflagged or mislabeled; Manifest pinned to Julia 1.12.6; only 1 exemplar's output committed. |
| A8 | `all_valid=True` ⇒ simulation is correct | Every PASS gate & CI | **HIGH (false).** Checks only construction-guaranteed range/sum/action invariants; degenerate & wrong beliefs, missing VFE, full fallback all pass. |
| A9 | Committed Manifest resolves on any OS/Julia | Cross-platform execution | **MEDIUM-HIGH.** Pinned to Julia 1.12.6 stdlibs vs compat `julia="1.10"`; JLL binaries platform-specific; runner would `Pkg.add` on failure. |
| A10 | "45/45 render + execute" is evidenced | Docs, release claims, user trust | **HIGH (not demonstrated).** Only 1 committed rxinfer artifact; gate runs parse-only on the rest; 47 exemplars vs "45" mismatch. |
| A11 | Julia/CI actually executes the generated scripts | The "execute" half of 45/45 | **HIGH (not exercised).** Test is `Meta.parseall` syntax check in a temp env; no script is run end-to-end in tests. |
| A12 | `uses_real_rxinfer=true` reflects whether RxInfer truly ran | Runtime provenance metadata | **HIGH (violated).** Hardcoded `true` even on full fallback. |

---

## Part 4 — Minimal hardening recommendations (highest leverage)

1. **Make fallback & provenance explicit and gate on it.** Emit `uses_real_rxinfer` from the
   *actual* branch, not hardcoded; add `vfe_present` and `inference_converged` INTO `all_valid`;
   fail the run (exit≠0) when `infer()` throws instead of silently substituting the filter.
2. **Re-generate and *commit* all 45 rendered scripts + results**, and change `_assert_julia_parse`
   to a real end-to-end run (or a checked-in blessed output diff). Assert belief correctness vs.
   true state and a non-degenerate VFE.
3. **Give the EFE a reference.** Add a numeric equality test vs the pymdp `expected_free_energy`
   over the same matrices; add a degeneracy/mass-entropy check to validation.
4. **Stop mislabeling VFE.** Report the true per-iteration whole-model VFE vector, or drop the
   per-step claim; never replicate one scalar across T and call it a trace.
5. **Precompile transparently.** Log which T values precompiled vs failed; cover every T the
   renderer can emit; document that caches are per-machine (they are not portable) or drop the
   portability claim.
6. **Pin/reproduce Julia+Manifest in CI** across ≥2 Julia versions (1.10, 1.12) and ≥2 OS so
   portability claims are true or the docs are corrected.

---

*Method: ~20 terminal/grep/python probes; all claims traced to specific files/lines. No code
changed; report only.*
