#!/usr/bin/env python3
"""ModelKind-strategy pattern for the RxInfer.jl renderer.

Dispatch model-code generation, animation graph layout, and validation-field
selection to per-``ModelKind`` strategy classes so the renderer no longer
hard-codes the flat POMDP generator.

Each strategy exposes:

- ``generate_model_code(gnn_spec, model_name) -> str`` — the Julia script
  (``@model`` + ``infer()``) for this model kind.
- ``generate_graph_layout(gnn_spec=None) -> dict`` — node positions for
  animations (``{node_name: (x, y)}``).
- ``get_validation_fields() -> list`` — extra validation fields.

``FlatStrategy`` is the canonical flat-POMDP generator. ``FactoredStrategy``
and ``MultiAgentStrategy`` deliberately render the extractor's composed
joint POMDP through the flat generator while stamping their true kind
(per-factor recovery happens downstream from the ``state_factors`` echo).
``HierarchicalStrategy`` renders two-level models natively (slow context
coupled into the fast-state prior) and 3+-level models as the joint
composition. ``ContinuousStrategy`` and ``LearningStrategy`` remain loud
``NotImplementedError`` stubs (roadmap A2/D1) — no exemplar currently
declares the data they need.
"""

from __future__ import annotations

import base64
import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from render.pomdp_contract import ModelKind

__all__ = [
    "ModelStrategy",
    "FlatStrategy",
    "FactoredStrategy",
    "HierarchicalStrategy",
    "MultiAgentStrategy",
    "ContinuousStrategy",
    "LearningStrategy",
    "get_model_strategy",
    "STRATEGY_REGISTRY",
]


def _now() -> str:
    """Return a timestamp string for generated-script headers."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ModelStrategy(ABC):
    """Base class for per-``ModelKind`` RxInfer generation strategies."""

    kind: ModelKind

    # --- hooks ---------------------------------------------------------

    @abstractmethod
    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Return the Julia ``@model`` code for this model kind."""
        raise NotImplementedError(
            f"{type(self).__name__}.generate_model_code is not implemented"
        )

    def generate_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return node positions for animations: ``{name: (x, y)}``."""
        return self._default_graph_layout(gnn_spec)

    def get_validation_fields(self) -> List[str]:
        """Return extra validation fields this strategy contributes."""
        return []

    def _default_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Canonical left-to-right POMDP plate layout."""
        return {
            "D": (0.00, 0.85),
            "A": (0.00, 0.55),
            "B": (0.00, 0.25),
            "C": (0.00, -0.05),
            "s": (0.33, 0.85),
            "u": (0.33, 0.25),
            "o": (0.66, 0.55),
            "s'": (0.66, 0.85),
            "G": (1.00, 0.55),
        }


class FlatStrategy(ModelStrategy):
    """Single-factor POMDP (the common case) — the existing canonical generator."""

    kind = ModelKind.FLAT

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate a genuine RxInfer.jl script with @model + infer() from canonical POMDP matrices.

        The generated script implements offline batch inference (Bayesian
        smoothing) with post-hoc EFE policy evaluation — NOT online active
        inference. The pipeline is:

        - Phase 1: Forward simulation for data collection (hand-rolled EFE)
        - Phase 2: Real RxInfer ``infer()`` with ``free_energy=true`` — if
          this fails, the script crashes (no fallback)
        - Phase 3: Smoothed posterior extraction from ``result.posteriors[:s]``
        - Phase 4: Post-hoc EFE and policy from smoothed posteriors

        The per-iteration VFE trace (``vfe_per_iteration``) is the real
        convergence diagnostic. ``variational_free_energy`` is reported as the
        per-iteration vector (length = INFERENCE_ITERATIONS), not a per-step
        constant.
        """
        model_display_name = (
            gnn_spec.get("model_name") or gnn_spec.get("name") or model_name
        )
        model_params = gnn_spec.get("model_parameters", {})
        num_states = int(model_params["num_hidden_states"])
        num_observations = int(model_params["num_obs"])
        num_actions = int(model_params["num_actions"])
        num_timesteps = int(model_params.get("num_timesteps", 20))
        seed = int(model_params.get("random_seed", model_params.get("seed", 42)))
        action_precision = float(
            model_params.get("action_precision", model_params.get("gamma", 4.0))
        )
        inference_iterations = int(model_params.get("inference_iterations", 20))
        # Stamp the strategy's own kind — the dispatcher already detected it
        # once; re-detecting here could disagree with the strategy that
        # actually generated the code (and lets joint-composition subclasses
        # stamp their true kind).
        model_kind_str = self.kind.value
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation — genuine @model + infer() pipeline
# Generated from GNN Model: {model_display_name}
# Generated: {_now()}
#
# This script uses real RxInfer.jl variational message-passing inference:
#   - @model defines the generative POMDP with Categorical / DiscreteTransition nodes
#   - infer() with free_energy=true returns posteriors over hidden states
#     and real variational free energy traces
#   - EFE and policy selection remain custom (not RxInfer's domain)

using Pkg
using RxInfer
using Distributions
using LinearAlgebra
using Random
using SHA
using StatsBase
using JSON
using Base64
using Dates

# --- Optional Julia-native plotting via Plots.jl (matplotlib-free PNGs).
# Guarded so a missing Plots installation/backend degrades gracefully and the
# script NEVER fails to run because of plotting.
const PLOTS_READY = try
    @eval using Plots
    true
catch e
    println("⚠️ Plots unavailable; PNG plotting disabled: $e")
    false
end

const SCHEMA_VERSION = "rxinfer_simulation_v1"
const MODEL_NAME = {model_name_literal}
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = {num_timesteps}
const RANDOM_SEED = {seed}
const ACTION_PRECISION = {action_precision}
const INFERENCE_ITERATIONS = {inference_iterations}
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "{model_kind_str}"
const GNN_SPEC_JSON_B64 = "{spec_json_b64}"
const GNN_SPEC = JSON.parse(String(base64decode(GNN_SPEC_JSON_B64)))

function package_version(name::String)
    for (_, dep) in Pkg.dependencies()
        if dep.name == name
            return string(dep.version)
        end
    end
    return "unknown"
end

# --- Real RxInfer.jl generative model ---
# The @model definition is precompiled in the GnnRxInferModels package module.
# Using `using` loads the precompiled cache (built once via PrecompileTools.jl),
# eliminating ~85s of JIT compilation on every run.
#
# The model is a generative POMDP: hidden states evolve via
# DiscreteTransition conditioned on the previous state and selected action;
# observations are emitted via DiscreteTransition through the likelihood
# matrix A.

using GnnRxInferModels: pomdp_model

# --- Custom EFE computation (Active Inference domain, not RxInfer's) ---

function softmax(values)
    shifted = values .- maximum(values)
    weights = exp.(shifted)
    return weights ./ sum(weights)
end

function categorical_index(probabilities)
    safe_probs = max.(probabilities, 1e-16)
    safe_probs ./= sum(safe_probs)
    return rand(Categorical(safe_probs))
end

function compute_efe(belief, action, A, B, C_pref)
    predicted_state = B[:, :, action] * belief
    predicted_state = max.(predicted_state, 1e-16)
    predicted_state ./= sum(predicted_state)
    predicted_obs = A * predicted_state
    predicted_obs = max.(predicted_obs, 1e-16)
    predicted_obs ./= sum(predicted_obs)

    ambiguity = 0.0
    for state in eachindex(predicted_state)
        likelihood = max.(A[:, state], 1e-16)
        ambiguity -= predicted_state[state] * sum(likelihood .* log.(likelihood))
    end

    preferred = max.(C_pref, 1e-16)
    risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(preferred)))
    return ambiguity + risk
end

# Policy: softmax(log E - gamma * EFE). The habit prior E enters via
# log-add (Active Inference habit term); with the uniform default E the
# log-term is constant and cancels inside softmax, preserving the
# E-less behavior exactly.
function select_action(belief, A, B, C_pref, E_prior)
    efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
    policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
    action = categorical_index(policy)
    return action, efe_values, policy
end

function compute_efe_and_policy(belief, A, B, C_pref, E_prior)
    efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
    policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
    return efe_values, policy
end

function validate_dimensions(A, B, C, D)
    if size(A) != (NUM_OBSERVATIONS, NUM_STATES)
        error("A shape $(size(A)) does not match expected ($NUM_OBSERVATIONS, $NUM_STATES)")
    end
    if size(B) != (NUM_STATES, NUM_STATES, NUM_ACTIONS)
        error("B shape $(size(B)) does not match expected ($NUM_STATES, $NUM_STATES, $NUM_ACTIONS)")
    end
    if length(C) != NUM_OBSERVATIONS
        error("C length $(length(C)) does not match expected $NUM_OBSERVATIONS")
    end
    if length(D) != NUM_STATES
        error("D length $(length(D)) does not match expected $NUM_STATES")
    end
end

# --- Offline batch inference (Bayesian smoothing) with post-hoc EFE policy
# evaluation.
#
# This is NOT online active inference. The pipeline is:
#   Phase 1 — Forward simulation for data collection: run the environment
#     forward using the hand-rolled EFE to collect observations, actions,
#     and true states. (The hand-rolled forward filter here is a data
#     collection mechanism, not a substitute for RxInfer inference.)
#   Phase 2 — Real RxInfer batch inference: run infer() with
#     free_energy=true on the collected data. If infer() fails, the script
#     crashes (exit non-zero). There is NO fallback.
#   Phase 3 — Posterior extraction: extract per-timestep smoothed posteriors
#     from result.posteriors[:s].
#   Phase 4 — Post-hoc EFE/policy from posteriors: compute EFE and policy
#     from the smoothed posteriors. These are post-hoc policy evaluations,
#     not online control.

function belief_entropy(belief)
    # Shannon entropy in nats. Returns 0 for a degenerate point-mass.
    safe = max.(belief, 1e-16)
    return -sum(safe .* log.(safe))
end

function run_simulation()
    Random.seed!(RANDOM_SEED)
    initial = GNN_SPEC["initialparameterization"]
    A = zeros(Float64, NUM_OBSERVATIONS, NUM_STATES)
    raw_A = initial["A"]
    for obs in 1:NUM_OBSERVATIONS
        row = collect(raw_A[obs])
        for state in 1:NUM_STATES
            A[obs, state] = Float64(row[state])
        end
    end
    # B is stored as (next_state, previous_state, action)
    raw_B = initial["B"]
    B = zeros(Float64, NUM_STATES, NUM_STATES, NUM_ACTIONS)
    for ns in 1:NUM_STATES
        for ps in 1:NUM_STATES
            for a in 1:NUM_ACTIONS
                B[ns, ps, a] = Float64(raw_B[ns][ps][a])
            end
        end
    end
    C = Float64.(collect(initial["C"]))
    D = Float64.(collect(initial["D"]))
    E = haskey(initial, "E") ? Float64.(collect(initial["E"])) : fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    if length(E) != NUM_ACTIONS
        error("E length $(length(E)) does not match expected $NUM_ACTIONS")
    end
    E = E ./ sum(E)  # normalize the habit prior
    validate_dimensions(A, B, C, D)

    C_pref = softmax(C)

    # --- Phase 1: Forward simulation for data collection ---
    # Uses a hand-rolled EFE-based forward filter to collect the observation
    # and action sequence. This is NOT the inference step — it is data
    # collection for the subsequent RxInfer batch inference.
    current_state = categorical_index(D)
    current_belief = copy(D)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    action_seq_full = Int[]  # 1-indexed actions for the model

    for step in 1:TIME_STEPS
        observation = categorical_index(A[:, current_state])

        # Simple Bayesian update for the forward-pass belief
        obs_onehot = [i == observation ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS]
        likelihood = A[observation, :]
        updated = current_belief .* likelihood
        if sum(updated) <= 0
            error("belief update produced zero mass at step $step")
        end
        current_belief = updated ./ sum(updated)

        # Action selection via EFE + habit prior E (forward-pass policy)
        action, efe_values, policy = select_action(current_belief, A, B, C_pref, E)

        # Environment transition
        next_probs = B[:, current_state, action]
        current_state = categorical_index(next_probs)

        # Predict next belief
        predicted = B[:, :, action] * current_belief
        current_belief = predicted ./ sum(predicted)

        push!(observations, observation - 1)  # 0-indexed for JSON
        push!(true_states, current_state - 1)
        push!(actions, action - 1)  # 0-indexed for JSON
        push!(action_seq_full, action)  # 1-indexed for model
    end

    # --- Phase 2: Real RxInfer batch inference (no fallback) ---
    # Build one-hot observation sequence for the model
    obs_seq = [[i == (obs + 1) ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS] for obs in observations]

    # The model needs u[1:T-1] for transitions, plus a dummy u[T]
    model_actions = copy(action_seq_full)
    while length(model_actions) < TIME_STEPS
        push!(model_actions, 1)
    end

    # NO try/catch — if infer() fails, the script crashes with a clear error.
    # This is deliberate: real RxInfer inference or nothing.
    result = infer(
        model = pomdp_model(A=A, B=B, D=D, u=model_actions, T=TIME_STEPS),
        data = (y = obs_seq,),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true
    )

    uses_real_rxinfer = true  # only reached if infer() succeeded

    # --- Phase 3: Posterior extraction (smoothed posteriors) ---
    # RxInfer returns posteriors[:s] as Vector of Vector of Categorical.
    # Outer index = iteration, inner index = time step.
    # We take the final iteration's posteriors — these are smoothed
    # (joint) posteriors from batch inference, not filtered (online) beliefs.
    posteriors_s = result.posteriors[:s]
    final_iter = posteriors_s[end]
    if isa(final_iter, Vector)
        posterior_per_step = final_iter
    else
        # Single Categorical (T=1 case)
        posterior_per_step = [final_iter]
    end

    beliefs = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()

    for t in 1:TIME_STEPS
        cat_dist = posterior_per_step[t]
        belief = copy(cat_dist.p)
        belief = max.(belief, 1e-16)
        belief ./= sum(belief)
        push!(beliefs, belief)

        # Phase 4: Post-hoc EFE and policy from the smoothed posterior
        efe_vals, pol = compute_efe_and_policy(belief, A, B, C_pref, E)
        push!(efe_per_action, efe_vals)
        push!(selected_efe, efe_vals[action_seq_full[t]])
        push!(policy_posterior, pol)
    end

    # --- VFE recording: per-iteration trace (the real convergence diagnostic) ---
    # RxInfer returns one VFE scalar per inference iteration (for the whole
    # model), NOT per timestep. We record the full per-iteration vector.
    vfe_per_iteration = Float64.(result.free_energy)  # length = INFERENCE_ITERATIONS

    # variational_free_energy (consumed by the analyzer): report the
    # per-iteration trace directly. This is per-iteration, not per-step.
    # Documented clearly in the results dict and the analyzer.
    variational_free_energy = copy(vfe_per_iteration)

    # Convergence check using the real per-iteration trace
    if length(vfe_per_iteration) >= 5
        last_5 = vfe_per_iteration[end-4:end]
        inference_converged = (maximum(last_5) - minimum(last_5)) < 1e-4
    elseif length(vfe_per_iteration) >= 2
        inference_converged = abs(vfe_per_iteration[end] - vfe_per_iteration[end-1]) < 1e-4
    else
        inference_converged = false  # too few iterations to assess
    end

    # --- Strengthened validation ---
    vfe_present = !isempty(vfe_per_iteration) && all(v -> v > 0, vfe_per_iteration)

    # Belief-entropy diagnostics. Exact Bayesian smoothing legitimately
    # produces near-zero-entropy marginals in high-signal regimes (each
    # marginal conditions on the WHOLE observation sequence), so low entropy
    # is not a failure by itself — systematic collapse only signals failure
    # when the beliefs also point at the WRONG states, which the
    # chance-relative accuracy gate below catches. belief_entropy_ok
    # therefore flags only the pathological combination: every timestep
    # degenerate AND accuracy below the gate. Raw entropy stats are
    # reported alongside for diagnosis.
    is_identity_A = all(abs(A[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                        for i in 1:size(A,1), j in 1:size(A,2))
    min_entropy = is_identity_A ? 0.0 : 0.1  # collapse threshold (nats)
    belief_entropies = [belief_entropy(b) for b in beliefs]
    all_beliefs_degenerate = !isempty(belief_entropies) &&
        maximum(belief_entropies) < min_entropy

    # Belief accuracy: check that argmax(belief) matches the true state
    # for a majority of timesteps. This catches systematic inference failures
    # where beliefs are valid distributions but point at the wrong state.
    belief_accuracy = 0.0
    if length(beliefs) == length(true_states) && length(beliefs) > 0
        correct = 0
        for t in 1:length(beliefs)
            if argmax(beliefs[t]) == (true_states[t] + 1)  # true_states are 0-indexed
                correct += 1
            end
        end
        belief_accuracy = Float64(correct) / length(beliefs)
    end
    # Identity A (fully observable): expect high accuracy. Non-identity A:
    # require accuracy meaningfully above chance (the old 0.0 threshold was
    # vacuously true) — twice chance, capped at 0.5.
    min_accuracy = is_identity_A ? 0.5 : min(0.5, 2.0 / NUM_STATES)
    belief_accuracy_ok = belief_accuracy >= min_accuracy
    belief_entropy_ok = !(all_beliefs_degenerate && !belief_accuracy_ok)

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
        "actions_in_range" => all(a -> 0 <= a < NUM_ACTIONS, actions),
        "inference_converged" => inference_converged,
        "vfe_present" => vfe_present,
        "belief_entropy_ok" => belief_entropy_ok,
        "belief_entropy_min" => isempty(belief_entropies) ? 0.0 : minimum(belief_entropies),
        "belief_entropy_mean" => isempty(belief_entropies) ? 0.0 : sum(belief_entropies) / length(belief_entropies),
        "belief_entropy_max" => isempty(belief_entropies) ? 0.0 : maximum(belief_entropies),
        "belief_accuracy" => belief_accuracy,
        "belief_accuracy_ok" => belief_accuracy_ok
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"] &&
        validation["inference_converged"] &&
        validation["vfe_present"] &&
        validation["belief_entropy_ok"] &&
        validation["belief_accuracy_ok"]

    # Compute script SHA256 for reproducibility tracking
    script_sha = try
        script_path = PROGRAM_FILE
        if isfile(script_path)
            open(script_path) do f
                bytes2hex(sha256(read(f)))
            end
        else
            "unknown"
        end
    catch
        "unknown"
    end

    return Dict(
        "schema_version" => SCHEMA_VERSION,
        "success" => true,
        "framework" => "RxInfer.jl",
        "model_name" => MODEL_NAME,
        "num_timesteps" => TIME_STEPS,
        "observations_by_modality" => Dict("joint_observation" => observations),
        "hidden_states_by_factor" => Dict("joint_state" => true_states),
        "actions_by_control_factor" => Dict("joint_action" => actions),
        "beliefs_by_factor" => Dict("joint_state" => beliefs),
        "expected_free_energy" => selected_efe,
        "efe_per_action" => efe_per_action,
        "variational_free_energy" => variational_free_energy,
        "vfe_per_iteration" => vfe_per_iteration,
        "policy_posterior" => policy_posterior,
        "observations" => observations,
        "true_states" => true_states,
        "actions" => actions,
        "beliefs" => beliefs,
        "model_parameters" => Dict(
            "A_shape" => collect(size(A)),
            "B_shape" => collect(size(B)),
            "C_shape" => [length(C)],
            "D_shape" => [length(D)],
            "E_shape" => [length(E)],
            "E" => E,
            "num_states" => NUM_STATES,
            "num_observations" => NUM_OBSERVATIONS,
            "num_actions" => NUM_ACTIONS,
            "inference_iterations" => INFERENCE_ITERATIONS,
            # Per-factor structure echoed from the GNN spec so downstream
            # analysis can un-flatten joint posteriors into per-factor
            # (per-agent) marginals without re-parsing the GNN file.
            "state_factors" => get(get(GNN_SPEC, "model_parameters", Dict()), "state_factors", []),
            "observation_modalities" => get(get(GNN_SPEC, "model_parameters", Dict()), "observation_modalities", [])
        ),
        "matrix_provenance" => get(GNN_SPEC, "matrix_provenance", Dict()),
        "runtime_metadata" => Dict(
            "random_seed" => RANDOM_SEED,
            "schema_version" => SCHEMA_VERSION,
            "generated_at" => string(now()),
            "rxinfer_version" => package_version("RxInfer"),
            "julia_version" => string(VERSION),
            "script_sha256" => script_sha,
            "inference_converged" => inference_converged,
            "uses_real_rxinfer" => uses_real_rxinfer,
            "model_kind" => MODEL_KIND,
            "b_tensor_order" => B_TENSOR_ORDER,
            "belief_accuracy" => belief_accuracy
        ),
        "metrics" => Dict(
            "expected_free_energy" => selected_efe,
            "policy_posterior" => policy_posterior,
            "belief_confidence" => [maximum(b) for b in beliefs],
            "variational_free_energy" => variational_free_energy
        ),
        "validation" => validation
    )
end

# --- Structured per-step execution log (JSON Lines: one record per step).
# Captures per-step beliefs / action / EFE / policy posterior / validation,
# written alongside simulation_results.json. Pure JSON + Base stdlib, and
# guarded so logging can never crash the simulation.
function write_execution_log(results)
    log_path = "simulation.log"
    beliefs = get(get(results, "beliefs_by_factor", Dict()), "joint_state", results["beliefs"])
    actions = results["actions"]
    efe = results["expected_free_energy"]
    efe_per_action = results["efe_per_action"]
    policy = results["policy_posterior"]
    validation = get(results, "validation", Dict())

    open(log_path, "w") do file
        for step in 1:TIME_STEPS
            record = Dict(
                "event" => "step",
                "step" => step,
                "model_name" => MODEL_NAME,
                "schema_version" => SCHEMA_VERSION,
                "belief" => beliefs[step],
                "action" => actions[step],
                "expected_free_energy" => efe[step],
                "efe_per_action" => efe_per_action[step],
                "policy_posterior" => policy[step],
                "validation" => validation
            )
            JSON.print(file, record)
            println(file)
        end
        summary = Dict(
            "event" => "summary",
            "schema_version" => SCHEMA_VERSION,
            "model_name" => MODEL_NAME,
            "num_steps" => TIME_STEPS,
            "validation" => validation
        )
        JSON.print(file, summary)
        println(file)
    end

    # Complete structured JSON sidecar for downstream tooling that prefers a
    # single document over JSONL.
    full_log = Dict(
        "schema_version" => SCHEMA_VERSION,
        "model_name" => MODEL_NAME,
        "format" => "jsonl",
        "num_steps" => TIME_STEPS,
        "validation" => validation,
        "log_file" => log_path
    )
    open("simulation_log.json", "w") do file
        JSON.print(file, full_log, 2)
    end

    println("RxInfer.jl simulation wrote $log_path and simulation_log.json")
    return log_path
end

# --- Julia-native visualization via Plots.jl (matplotlib-free PNGs).
# Everything is wrapped in try/catch so a missing Plots backend degrades to a
# warning and NEVER prevents the simulation from running to completion.
function write_plots(results)
    if !PLOTS_READY
        println("⚠️ Skipping PNG plots (Plots backend not available)")
        return
    end
    try
        beliefs = get(get(results, "beliefs_by_factor", Dict()), "joint_state", results["beliefs"])
        efe = results["expected_free_energy"]
        policy = results["policy_posterior"]

        if !isempty(beliefs)
            belief_mat = hcat(beliefs...)
            steps = 1:size(belief_mat, 2)
            p1 = plot(
                title = "Belief Evolution over Time",
                xlabel = "Time step",
                ylabel = "Belief mass",
                legend = :outertopright,
                size = (900, 450),
                titlefontsize = 12,
                guidefontsize = 10,
                legendfontsize = 8,
                tickfontsize = 8,
                linewidth = 2
            )
            for state in 1:size(belief_mat, 1)
                plot!(p1, steps, belief_mat[state, :], label = "State $state")
            end
            savefig(p1, "belief_evolution.png")
        end

        if !isempty(efe)
            p2 = plot(
                1:length(efe), efe,
                title = "Expected Free Energy over Time",
                xlabel = "Time step",
                ylabel = "Action EFE",
                label = "selected EFE",
                legend = :topright,
                size = (900, 400),
                titlefontsize = 12,
                guidefontsize = 10,
                legendfontsize = 8,
                tickfontsize = 8,
                linewidth = 2
            )
            savefig(p2, "efe_over_time.png")
        end

        if !isempty(policy)
            policy_mat = hcat(policy...)
            p3 = heatmap(policy_mat,
                title = "Policy Posterior over Time",
                xlabel = "Time step",
                ylabel = "Action",
                color = :viridis,
                colorbar = :right,
                size = (900, 400),
                titlefontsize = 12,
                guidefontsize = 10,
                tickfontsize = 8
            )
            savefig(p3, "policy_posterior.png")
        end

        println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png, efe_over_time.png, policy_posterior.png)")
    catch e
        println("⚠️ Plotting skipped (Plots backend unavailable): $e")
    end
end

function main()
    results = run_simulation()
    # Sanitize NaN/Inf values before JSON serialization (JSON.jl rejects them by default)
    function sanitize!(x)
        if isa(x, Float64)
            if isnan(x) || isinf(x)
                return 0.0
            end
            return x
        elseif isa(x, Vector)
            return [sanitize!(v) for v in x]
        elseif isa(x, Dict)
            for (k, v) in x
                x[k] = sanitize!(v)
            end
            return x
        end
        return x
    end
    results = sanitize!(results)
    open("simulation_results.json", "w") do file
        JSON.print(file, results, 2)
    end
    println("RxInfer.jl simulation wrote simulation_results.json")
    write_execution_log(results)
    write_plots(results)
    return results["validation"]["all_valid"] ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''
        return code

    def generate_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Flat model: single-observation / single-state-factor plate.
        return self._default_graph_layout(gnn_spec)

    def get_validation_fields(self) -> List[str]:
        return [
            "all_beliefs_valid",
            "beliefs_sum_to_one",
            "actions_in_range",
            "inference_converged",
            "vfe_present",
            "belief_entropy_ok",
            "belief_accuracy",
            "belief_accuracy_ok",
        ]


class _NotImplementedStrategy(ModelStrategy):
    """Base for model kinds whose generators are not yet implemented."""

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        raise NotImplementedError(
            f"RxInfer {self.kind.value} model generation is not yet implemented"
        )

    def generate_graph_layout(
        self, gnn_spec: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"RxInfer {self.kind.value} graph layout is not yet implemented"
        )

    def get_validation_fields(self) -> List[str]:
        raise NotImplementedError(
            f"RxInfer {self.kind.value} validation fields are not yet implemented"
        )


class _JointCompositionStrategy(FlatStrategy):
    """Render via the extractor's joint composition (the flat ``@model``).

    The extractor already composes multiple factors into one joint POMDP
    (C-order ``itertools.product`` over ``state_factors``), and that joint
    model renders and executes through the flat generator. Subclasses stamp
    their true detected kind into the generated script's ``model_kind``
    metadata, and the results JSON echoes ``state_factors`` so downstream
    analysis can recover per-factor marginals from the joint posterior.

    This is a deliberate, documented rendering decision (the
    pre-strategy-pattern behavior), not a silent fallback.
    """


class FactoredStrategy(_JointCompositionStrategy):
    """Multiple independent hidden state factors.

    Rendered as the composed joint POMDP; native per-factor ``@model``
    rendering (separate Categorical chains per factor, roadmap D3) is open.
    """

    kind = ModelKind.FACTORED


class MultiAgentStrategy(_JointCompositionStrategy):
    """Multiple coordinated agents.

    Rendered as the composed joint POMDP; per-agent beliefs are recovered
    downstream by marginalizing the joint posterior over the other agents'
    factors (roadmap D4), using the ``state_factors`` echo in the results.
    """

    kind = ModelKind.MULTI_AGENT


class HierarchicalStrategy(FlatStrategy):
    """Two-level slow/fast hierarchical POMDP rendering (roadmap A3).

    Two-level exemplars (``A_level1``/``B_level1``/... plus
    ``A_level2``/``D_level2``) render to the native hierarchical ``@model``
    in ``GnnRxInferModels``: a single Categorical context ``z`` couples into
    the fast-state prior via the column-normalized ``A_level2``, and the
    fast chain is driven by observed actions over ``B_level1``. Inference
    requires the mean-field constraint + uniform marginal initialization
    shipped with the model (Bethe free-energy scoring of the non-square
    coupling hits ReactiveMP's square-matrix ``mul_trace`` assertion
    otherwise — verified empirically against RxInfer 5.5). Context dynamics
    (``B_level2``) are applied post-hoc as deterministic prior propagation
    and reported as the slow factor's belief trajectory.

    Models declaring 3+ levels render as the extractor's joint composition
    (the pre-strategy-pattern behavior) with ``model_kind`` stamped
    ``hierarchical`` — a deliberate, documented interim decision recorded in
    the roadmap (native N-level chain rendering is open), not a silent
    fallback.
    """

    kind = ModelKind.HIERARCHICAL

    _REQUIRED_TWO_LEVEL = (
        "A_level1",
        "B_level1",
        "C_level1",
        "A_level2",
        "D_level2",
    )

    @staticmethod
    def _declared_levels(gnn_spec: Dict[str, Any]) -> set:
        matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
        levels = set()
        for key in matrices:
            match = re.match(r"^[ABCDE]_level(\d+)$", str(key))
            if match:
                levels.add(int(match.group(1)))
        return levels

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        levels = self._declared_levels(gnn_spec)
        if levels == {1, 2}:
            matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
            missing = [key for key in self._REQUIRED_TWO_LEVEL if key not in matrices]
            if missing:
                raise ValueError(
                    f"hierarchical model {model_name} is missing per-level "
                    f"matrices required for two-level rendering: {missing}"
                )
            return self._generate_two_level_code(gnn_spec, model_name)
        # 3+ declared levels: joint composition (see class docstring).
        return super().generate_model_code(gnn_spec, model_name)

    def get_validation_fields(self) -> List[str]:
        return super().get_validation_fields() + [
            "context_beliefs_valid",
            "context_beliefs_sum_to_one",
        ]

    def _generate_two_level_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """Generate the native two-level hierarchical RxInfer.jl script."""
        matrices = gnn_spec["structured_pomdp"]["matrices"]
        a_fast = matrices["A_level1"]
        b_fast = matrices["B_level1"]
        a_ctx = matrices["A_level2"]
        d_slow = matrices["D_level2"]
        num_obs_fast = len(a_fast)
        num_fast = len(a_fast[0])
        num_actions = len(b_fast)  # parsed layout: [action][next][prev]
        num_slow = len(d_slow)
        if len(a_ctx) != num_fast or len(a_ctx[0]) != num_slow:
            raise ValueError(
                f"hierarchical model {model_name}: A_level2 shape "
                f"({len(a_ctx)}, {len(a_ctx[0])}) does not match "
                f"(n_fast={num_fast}, n_slow={num_slow})"
            )

        model_display_name = (
            gnn_spec.get("model_name") or gnn_spec.get("name") or model_name
        )
        model_params = gnn_spec.get("model_parameters", {})
        num_timesteps = int(model_params.get("num_timesteps", 20))
        seed = int(model_params.get("random_seed", model_params.get("seed", 42)))
        action_precision = float(
            model_params.get("action_precision", model_params.get("gamma", 4.0))
        )
        inference_iterations = int(model_params.get("inference_iterations", 20))
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl two-level hierarchical POMDP simulation — genuine @model + infer()
# Generated from GNN Model: {model_display_name}
# Generated: {_now()}
#
# Structure (matches the GNN file's declared semantics):
#   z (slow context, {num_slow} states) --A_level2--> fast-state prior
#   s[t] (fast, {num_fast} states) driven by actions over B_level1
#   y[t] ({num_obs_fast} obs) emitted through A_level1
# Context dynamics (B_level2) are applied POST-HOC as deterministic prior
# propagation of q(z) — labeled as such, never presented as inference.

using Pkg
using RxInfer
using Distributions
using LinearAlgebra
using Random
using SHA
using StatsBase
using JSON
using Base64
using Dates

const PLOTS_READY = try
    @eval using Plots
    true
catch e
    println("⚠️ Plots unavailable; PNG plotting disabled: $e")
    false
end

const SCHEMA_VERSION = "rxinfer_simulation_v1"
const MODEL_NAME = {model_name_literal}
const NUM_FAST = {num_fast}
const NUM_SLOW = {num_slow}
const NUM_OBSERVATIONS = {num_obs_fast}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = {num_timesteps}
const RANDOM_SEED = {seed}
const ACTION_PRECISION = {action_precision}
const INFERENCE_ITERATIONS = {inference_iterations}
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "{self.kind.value}"
const GNN_SPEC_JSON_B64 = "{spec_json_b64}"
const GNN_SPEC = JSON.parse(String(base64decode(GNN_SPEC_JSON_B64)))

function package_version(name::String)
    for (_, dep) in Pkg.dependencies()
        if dep.name == name
            return string(dep.version)
        end
    end
    return "unknown"
end

# The hierarchical @model, its mean-field constraints, and its marginal
# initialization are precompiled in the GnnRxInferModels package module.
using GnnRxInferModels:
    hierarchical_pomdp_model, hierarchical_constraints, hierarchical_initialization

# --- Custom EFE computation on the FAST level (Active Inference domain) ---

function softmax(values)
    shifted = values .- maximum(values)
    weights = exp.(shifted)
    return weights ./ sum(weights)
end

function categorical_index(probabilities)
    safe_probs = max.(probabilities, 1e-16)
    safe_probs ./= sum(safe_probs)
    return rand(Categorical(safe_probs))
end

function compute_efe(belief, action, A, B, C_pref)
    predicted_state = B[:, :, action] * belief
    predicted_state = max.(predicted_state, 1e-16)
    predicted_state ./= sum(predicted_state)
    predicted_obs = A * predicted_state
    predicted_obs = max.(predicted_obs, 1e-16)
    predicted_obs ./= sum(predicted_obs)

    ambiguity = 0.0
    for state in eachindex(predicted_state)
        likelihood = max.(A[:, state], 1e-16)
        ambiguity -= predicted_state[state] * sum(likelihood .* log.(likelihood))
    end

    preferred = max.(C_pref, 1e-16)
    risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(preferred)))
    return ambiguity + risk
end

function select_action(belief, A, B, C_pref)
    efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
    policy = softmax(-ACTION_PRECISION .* efe_values)
    action = categorical_index(policy)
    return action, efe_values, policy
end

function compute_efe_and_policy(belief, A, B, C_pref)
    efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
    policy = softmax(-ACTION_PRECISION .* efe_values)
    return efe_values, policy
end

function belief_entropy(belief)
    safe = max.(belief, 1e-16)
    return -sum(safe .* log.(safe))
end

function load_level_matrices()
    matrices = GNN_SPEC["structured_pomdp"]["matrices"]

    raw_A = matrices["A_level1"]
    A = zeros(Float64, NUM_OBSERVATIONS, NUM_FAST)
    for o in 1:NUM_OBSERVATIONS, s in 1:NUM_FAST
        A[o, s] = Float64(raw_A[o][s])
    end
    A = A ./ sum(A, dims = 1)  # column-normalize likelihood

    # Parsed layout of B_level1 is [action][next][prev]
    raw_B = matrices["B_level1"]
    if length(raw_B) != NUM_ACTIONS
        error("B_level1 action count $(length(raw_B)) != expected $NUM_ACTIONS")
    end
    B = zeros(Float64, NUM_FAST, NUM_FAST, NUM_ACTIONS)
    for a in 1:NUM_ACTIONS, ns in 1:NUM_FAST, ps in 1:NUM_FAST
        B[ns, ps, a] = Float64(raw_B[a][ns][ps])
    end

    C = Float64.(collect(matrices["C_level1"]))
    if length(C) != NUM_OBSERVATIONS
        error("C_level1 length $(length(C)) != expected $NUM_OBSERVATIONS")
    end

    raw_ctx = matrices["A_level2"]
    A_ctx = zeros(Float64, NUM_FAST, NUM_SLOW)
    for s in 1:NUM_FAST, k in 1:NUM_SLOW
        A_ctx[s, k] = Float64(raw_ctx[s][k])
    end
    A_ctx = A_ctx ./ sum(A_ctx, dims = 1)  # columns are P(s1 | z=k)

    D_slow = Float64.(collect(matrices["D_level2"]))
    D_slow = D_slow ./ sum(D_slow)

    # B_level2 (context dynamics) is used only for post-hoc propagation.
    B_slow = Matrix{{Float64}}(I, NUM_SLOW, NUM_SLOW)
    if haskey(matrices, "B_level2")
        raw_slow = matrices["B_level2"]
        for ns in 1:NUM_SLOW, ps in 1:NUM_SLOW
            B_slow[ns, ps] = Float64(raw_slow[ns][ps])
        end
        B_slow = B_slow ./ sum(B_slow, dims = 1)
    end

    return A, B, C, A_ctx, D_slow, B_slow
end

function run_simulation()
    Random.seed!(RANDOM_SEED)
    A, B, C, A_ctx, D_slow, B_slow = load_level_matrices()
    C_pref = softmax(C)

    # --- Phase 1: Forward simulation for data collection ---
    # Sample the true context, derive the context-modulated fast prior,
    # then run the same EFE-driven forward pass as the flat generator.
    z_true = categorical_index(D_slow)
    fast_prior = copy(A_ctx[:, z_true])
    current_state = categorical_index(fast_prior)
    current_belief = copy(fast_prior)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    action_seq_full = Int[]

    for step in 1:TIME_STEPS
        observation = categorical_index(A[:, current_state])

        likelihood = A[observation, :]
        updated = current_belief .* likelihood
        if sum(updated) <= 0
            error("belief update produced zero mass at step $step")
        end
        current_belief = updated ./ sum(updated)

        action, efe_values, policy = select_action(current_belief, A, B, C_pref)

        next_probs = B[:, current_state, action]
        current_state = categorical_index(next_probs)

        predicted = B[:, :, action] * current_belief
        current_belief = predicted ./ sum(predicted)

        push!(observations, observation - 1)
        push!(true_states, current_state - 1)
        push!(actions, action - 1)
        push!(action_seq_full, action)
    end

    # --- Phase 2: Real RxInfer hierarchical inference (no fallback) ---
    obs_seq = [[i == (obs + 1) ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS] for obs in observations]
    model_actions = copy(action_seq_full)
    while length(model_actions) < TIME_STEPS
        push!(model_actions, 1)
    end

    # NO try/catch — if infer() fails, the script crashes with a clear error.
    result = infer(
        model = hierarchical_pomdp_model(A=A, B=B, A_ctx=A_ctx, D_slow=D_slow,
                                         u=model_actions, T=TIME_STEPS),
        data = (y = obs_seq,),
        constraints = hierarchical_constraints(),
        initialization = hierarchical_initialization(NUM_FAST, NUM_SLOW),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true
    )

    uses_real_rxinfer = true

    # --- Phase 3: Posterior extraction (fast chain + context) ---
    posteriors_s = result.posteriors[:s]
    final_iter = posteriors_s[end]
    posterior_per_step = isa(final_iter, Vector) ? final_iter : [final_iter]

    posteriors_z = result.posteriors[:z]
    q_z = posteriors_z[end]
    context_posterior = copy(q_z.p)
    context_posterior = max.(context_posterior, 1e-16)
    context_posterior ./= sum(context_posterior)

    beliefs = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()

    for t in 1:TIME_STEPS
        cat_dist = posterior_per_step[t]
        belief = copy(cat_dist.p)
        belief = max.(belief, 1e-16)
        belief ./= sum(belief)
        push!(beliefs, belief)

        efe_vals, pol = compute_efe_and_policy(belief, A, B, C_pref)
        push!(efe_per_action, efe_vals)
        push!(selected_efe, efe_vals[action_seq_full[t]])
        push!(policy_posterior, pol)
    end

    # --- Phase 4: Post-hoc context trajectory (deterministic propagation) ---
    # q(z) is inferred ONCE per episode (the only declared evidence channel
    # is the fast-state prior at episode start). The per-timestep slow
    # trajectory reported below is q(z) propagated by the declared context
    # dynamics B_level2 — deterministic prior propagation, NOT inference.
    context_beliefs = Vector{{Vector{{Float64}}}}()
    push!(context_beliefs, copy(context_posterior))
    for t in 2:TIME_STEPS
        propagated = B_slow * context_beliefs[end]
        propagated = max.(propagated, 1e-16)
        propagated ./= sum(propagated)
        push!(context_beliefs, propagated)
    end

    vfe_per_iteration = Float64.(result.free_energy)
    variational_free_energy = copy(vfe_per_iteration)

    if length(vfe_per_iteration) >= 5
        last_5 = vfe_per_iteration[end-4:end]
        inference_converged = (maximum(last_5) - minimum(last_5)) < 1e-4
    elseif length(vfe_per_iteration) >= 2
        inference_converged = abs(vfe_per_iteration[end] - vfe_per_iteration[end-1]) < 1e-4
    else
        inference_converged = false
    end

    vfe_present = !isempty(vfe_per_iteration) && all(v -> v > 0, vfe_per_iteration)

    # Same entropy/accuracy semantics as the flat generator: entropy is a
    # diagnostic; only (all-degenerate AND below-chance-gate accuracy) fails.
    is_identity_A = all(abs(A[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                        for i in 1:size(A,1), j in 1:size(A,2))
    min_entropy = is_identity_A ? 0.0 : 0.1
    belief_entropies = [belief_entropy(b) for b in beliefs]
    all_beliefs_degenerate = !isempty(belief_entropies) &&
        maximum(belief_entropies) < min_entropy

    belief_accuracy = 0.0
    if length(beliefs) == length(true_states) && length(beliefs) > 0
        correct = 0
        for t in 1:length(beliefs)
            if argmax(beliefs[t]) == (true_states[t] + 1)
                correct += 1
            end
        end
        belief_accuracy = Float64(correct) / length(beliefs)
    end
    min_accuracy = is_identity_A ? 0.5 : min(0.5, 2.0 / NUM_FAST)
    belief_accuracy_ok = belief_accuracy >= min_accuracy
    belief_entropy_ok = !(all_beliefs_degenerate && !belief_accuracy_ok)

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
        "actions_in_range" => all(a -> 0 <= a < NUM_ACTIONS, actions),
        "inference_converged" => inference_converged,
        "vfe_present" => vfe_present,
        "belief_entropy_ok" => belief_entropy_ok,
        "belief_entropy_min" => isempty(belief_entropies) ? 0.0 : minimum(belief_entropies),
        "belief_entropy_mean" => isempty(belief_entropies) ? 0.0 : sum(belief_entropies) / length(belief_entropies),
        "belief_entropy_max" => isempty(belief_entropies) ? 0.0 : maximum(belief_entropies),
        "belief_accuracy" => belief_accuracy,
        "belief_accuracy_ok" => belief_accuracy_ok,
        "context_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), context_beliefs),
        "context_beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), context_beliefs)
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"] &&
        validation["inference_converged"] &&
        validation["vfe_present"] &&
        validation["belief_entropy_ok"] &&
        validation["belief_accuracy_ok"] &&
        validation["context_beliefs_valid"] &&
        validation["context_beliefs_sum_to_one"]

    script_sha = try
        script_path = PROGRAM_FILE
        if isfile(script_path)
            open(script_path) do f
                bytes2hex(sha256(read(f)))
            end
        else
            "unknown"
        end
    catch
        "unknown"
    end

    return Dict(
        "schema_version" => SCHEMA_VERSION,
        "success" => true,
        "framework" => "RxInfer.jl",
        "model_name" => MODEL_NAME,
        "num_timesteps" => TIME_STEPS,
        "observations_by_modality" => Dict("fast_observation" => observations),
        "hidden_states_by_factor" => Dict(
            "fast_state" => true_states,
            "slow_context" => fill(z_true - 1, TIME_STEPS)
        ),
        "actions_by_control_factor" => Dict("fast_action" => actions),
        "beliefs_by_factor" => Dict(
            "fast_state" => beliefs,
            "slow_context" => context_beliefs
        ),
        "expected_free_energy" => selected_efe,
        "efe_per_action" => efe_per_action,
        "variational_free_energy" => variational_free_energy,
        "vfe_per_iteration" => vfe_per_iteration,
        "policy_posterior" => policy_posterior,
        "observations" => observations,
        "true_states" => true_states,
        "actions" => actions,
        "beliefs" => beliefs,
        "context_posterior" => context_posterior,
        "model_parameters" => Dict(
            "A_level1_shape" => collect(size(A)),
            "B_level1_shape" => collect(size(B)),
            "A_level2_shape" => collect(size(A_ctx)),
            "num_states" => NUM_FAST,
            "num_fast_states" => NUM_FAST,
            "num_slow_states" => NUM_SLOW,
            "num_observations" => NUM_OBSERVATIONS,
            "num_actions" => NUM_ACTIONS,
            "inference_iterations" => INFERENCE_ITERATIONS,
            "state_factors" => get(get(GNN_SPEC, "model_parameters", Dict()), "state_factors", []),
            "observation_modalities" => get(get(GNN_SPEC, "model_parameters", Dict()), "observation_modalities", [])
        ),
        "matrix_provenance" => get(GNN_SPEC, "matrix_provenance", Dict()),
        "runtime_metadata" => Dict(
            "random_seed" => RANDOM_SEED,
            "schema_version" => SCHEMA_VERSION,
            "generated_at" => string(now()),
            "rxinfer_version" => package_version("RxInfer"),
            "julia_version" => string(VERSION),
            "script_sha256" => script_sha,
            "inference_converged" => inference_converged,
            "uses_real_rxinfer" => uses_real_rxinfer,
            "model_kind" => MODEL_KIND,
            "b_tensor_order" => B_TENSOR_ORDER,
            "hierarchical_rendering" => "native_two_level",
            "context_trajectory" => "posthoc_prior_propagation",
            "belief_accuracy" => belief_accuracy
        ),
        "metrics" => Dict(
            "expected_free_energy" => selected_efe,
            "policy_posterior" => policy_posterior,
            "belief_confidence" => [maximum(b) for b in beliefs],
            "variational_free_energy" => variational_free_energy
        ),
        "validation" => validation
    )
end

# --- Structured per-step execution log (JSON Lines) ---
function write_execution_log(results)
    log_path = "simulation.log"
    beliefs = get(get(results, "beliefs_by_factor", Dict()), "fast_state", results["beliefs"])
    actions = results["actions"]
    efe = results["expected_free_energy"]
    efe_per_action = results["efe_per_action"]
    policy = results["policy_posterior"]
    validation = get(results, "validation", Dict())

    open(log_path, "w") do file
        for step in 1:TIME_STEPS
            record = Dict(
                "event" => "step",
                "step" => step,
                "model_name" => MODEL_NAME,
                "schema_version" => SCHEMA_VERSION,
                "belief" => beliefs[step],
                "action" => actions[step],
                "expected_free_energy" => efe[step],
                "efe_per_action" => efe_per_action[step],
                "policy_posterior" => policy[step],
                "validation" => validation
            )
            JSON.print(file, record)
            println(file)
        end
        summary = Dict(
            "event" => "summary",
            "schema_version" => SCHEMA_VERSION,
            "model_name" => MODEL_NAME,
            "num_steps" => TIME_STEPS,
            "validation" => validation
        )
        JSON.print(file, summary)
        println(file)
    end

    full_log = Dict(
        "schema_version" => SCHEMA_VERSION,
        "model_name" => MODEL_NAME,
        "format" => "jsonl",
        "num_steps" => TIME_STEPS,
        "validation" => validation,
        "log_file" => log_path
    )
    open("simulation_log.json", "w") do file
        JSON.print(file, full_log, 2)
    end

    println("RxInfer.jl simulation wrote $log_path and simulation_log.json")
    return log_path
end

# --- Julia-native visualization (fast beliefs + context trajectory) ---
function write_plots(results)
    if !PLOTS_READY
        println("⚠️ Skipping PNG plots (Plots backend not available)")
        return
    end
    try
        beliefs = get(get(results, "beliefs_by_factor", Dict()), "fast_state", results["beliefs"])
        context = get(get(results, "beliefs_by_factor", Dict()), "slow_context", [])

        if !isempty(beliefs)
            belief_mat = hcat(beliefs...)
            steps = 1:size(belief_mat, 2)
            p1 = plot(
                title = "Fast-State Belief Evolution",
                xlabel = "Time step",
                ylabel = "Belief mass",
                legend = :outertopright,
                size = (900, 450),
                linewidth = 2
            )
            for state in 1:size(belief_mat, 1)
                plot!(p1, steps, belief_mat[state, :], label = "State $state")
            end
            savefig(p1, "belief_evolution.png")
        end

        if !isempty(context)
            ctx_mat = hcat(context...)
            steps = 1:size(ctx_mat, 2)
            p2 = plot(
                title = "Context Belief (post-hoc propagation)",
                xlabel = "Time step",
                ylabel = "Belief mass",
                legend = :outertopright,
                size = (900, 400),
                linewidth = 2
            )
            for k in 1:size(ctx_mat, 1)
                plot!(p2, steps, ctx_mat[k, :], label = "Context $k")
            end
            savefig(p2, "context_evolution.png")
        end

        println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png, context_evolution.png)")
    catch e
        println("⚠️ Plotting skipped (Plots backend unavailable): $e")
    end
end

function main()
    results = run_simulation()
    function sanitize!(x)
        if isa(x, Float64)
            if isnan(x) || isinf(x)
                return 0.0
            end
            return x
        elseif isa(x, Vector)
            return [sanitize!(v) for v in x]
        elseif isa(x, Dict)
            for (k, v) in x
                x[k] = sanitize!(v)
            end
            return x
        end
        return x
    end
    results = sanitize!(results)
    open("simulation_results.json", "w") do file
        JSON.print(file, results, 2)
    end
    println("RxInfer.jl simulation wrote simulation_results.json")
    write_execution_log(results)
    write_plots(results)
    return results["validation"]["all_valid"] ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''
        return code


class ContinuousStrategy(_NotImplementedStrategy):
    """Continuous state/observation spaces.

    The Julia-side linear-Gaussian ``@model`` (``continuous_pomdp_model``)
    exists and is precompile-verified, but no GNN exemplar declares the
    continuous parameterization (F/H/Q/R + Gaussian prior) it needs — the
    continuous exemplars deliberately ship discretized POMDP equivalents
    and correctly detect FLAT. Deriving F/H/Q/R from discrete A/B/C/D would
    fabricate data, so this strategy raises until exemplars carry
    authored continuous sections (roadmap A2).
    """

    kind = ModelKind.CONTINUOUS


class LearningStrategy(_NotImplementedStrategy):
    """Parameter learning (Dirichlet priors, etc.) — roadmap D1 open."""

    kind = ModelKind.LEARNING


STRATEGY_REGISTRY: Dict[ModelKind, ModelStrategy] = {
    ModelKind.FLAT: FlatStrategy(),
    ModelKind.FACTORED: FactoredStrategy(),
    ModelKind.HIERARCHICAL: HierarchicalStrategy(),
    ModelKind.MULTI_AGENT: MultiAgentStrategy(),
    ModelKind.CONTINUOUS: ContinuousStrategy(),
    ModelKind.LEARNING: LearningStrategy(),
}


def get_model_strategy(model_kind: ModelKind) -> ModelStrategy:
    """Return the registered strategy for a ``ModelKind``."""
    return STRATEGY_REGISTRY[model_kind]
