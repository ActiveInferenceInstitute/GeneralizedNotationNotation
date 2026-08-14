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

``FlatStrategy`` is the canonical flat-POMDP generator.
``MultiAgentStrategy`` deliberately renders the extractor's composed joint
POMDP through the flat generator while stamping its true kind (per-agent
recovery happens downstream from the ``state_factors`` echo).
``HierarchicalStrategy`` renders two-level models natively (slow context
coupled into the fast-state prior) and 3+-level models as the joint
composition. ``FactoredStrategy`` (roadmap D3), ``ContinuousStrategy``
(A2) and ``LearningStrategy`` (D1) render natively against the
``factored_pomdp_model``, ``continuous_pomdp_model`` and
``learning_pomdp_model`` definitions in ``GnnRxInferModels``; each raises
``ValueError`` naming the missing parameterization when a spec reaches it
without the matrices its ``@model`` requires.
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
        ...

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

        Two inference modes, selected by ``model_parameters.inference_mode``
        (``"batch"`` default, or ``"online"`` — settable from a GNN file's
        ModelParameters or the renderer's options):

        BATCH (default) — offline Bayesian smoothing with post-hoc EFE policy
        evaluation:

        - Phase 1: Forward simulation for data collection (hand-rolled EFE)
        - Phase 2: Real RxInfer ``infer()`` with ``free_energy=true`` — if
          this fails, the script crashes (no fallback)
        - Phase 3: Smoothed posterior extraction from ``result.posteriors[:s]``
        - Phase 4: Post-hoc EFE and policy from smoothed posteriors

        ONLINE (roadmap A1) — genuine online active inference: at every
        timestep ``infer()`` runs on the observation prefix ``y[1:t]`` and the
        FILTERED posterior at ``t`` feeds action selection for the next step.
        Beliefs in the results are filtered (not smoothed) posteriors.

        The per-iteration VFE trace (``vfe_per_iteration``) is the real
        convergence diagnostic. ``variational_free_energy`` is reported as the
        per-iteration vector (length = INFERENCE_ITERATIONS), not a per-step
        constant.
        """
        mode = str(
            (gnn_spec.get("model_parameters") or {}).get("inference_mode", "batch")
        ).lower()
        if mode not in ("batch", "online"):
            raise ValueError(
                f"inference_mode must be 'batch' or 'online', got {mode!r}"
            )
        if mode == "online":
            return self._generate_online_code(gnn_spec, model_name)
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
        emitting_state = current_state  # the state that generated this observation

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
        push!(true_states, emitting_state - 1)  # state that emitted observation t (matches beliefs[t])
        push!(actions, action - 1)  # 0-indexed for JSON
        push!(action_seq_full, action)  # 1-indexed for model
    end

    # --- Phase 2: Real RxInfer batch inference (no fallback) ---
    # Build one-hot observation sequence for the model
    obs_seq = [[i == (obs + 1) ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS] for obs in observations]

        # The model needs u[1:T-1] for transitions, plus a padding u[T]
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

    def _generate_online_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the online (filtering) active-inference variant (A1).

        Per timestep the script runs ``infer()`` on the observation prefix
        ``y[1:t]`` and feeds the FILTERED posterior at ``t`` into EFE action
        selection — a genuine perception→action loop, unlike the batch
        variant's hand-rolled forward filter. The reported beliefs are the
        filtered posteriors; the VFE trace comes from the final full-sequence
        run (t = T), so its length is still INFERENCE_ITERATIONS.
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
        model_kind_str = self.kind.value
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl ONLINE active-inference POMDP simulation — per-timestep infer()
# Generated from GNN Model: {model_display_name}
# Generated: {_now()}
#
# Online mode (roadmap A1): at every timestep t, infer() runs on the
# observation prefix y[1:t]; the FILTERED posterior at t drives EFE action
# selection for the next step. Beliefs below are filtered (not smoothed).

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
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = {num_timesteps}
const RANDOM_SEED = {seed}
const ACTION_PRECISION = {action_precision}
const INFERENCE_ITERATIONS = {inference_iterations}
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "{model_kind_str}"
const INFERENCE_MODE = "online"
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

using GnnRxInferModels: pomdp_model

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

function select_action(belief, A, B, C_pref, E_prior)
    efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
    policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
    action = categorical_index(policy)
    return action, efe_values, policy
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

function belief_entropy(belief)
    safe = max.(belief, 1e-16)
    return -sum(safe .* log.(safe))
end

# Run infer() on the observation prefix and return (filtered belief at the
# last step, the run's per-iteration free-energy trace). NO try/catch — if
# infer() fails the script crashes (no fallback).
function filtered_posterior(obs_prefix, actions_prefix, A, B, D)
    t = length(obs_prefix)
    u = t > 1 ? actions_prefix[1:(t - 1)] : [1]  # u is never indexed when t == 1
    result = infer(
        model = pomdp_model(A=A, B=B, D=D, u=u, T=t),
        data = (y = obs_prefix,),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true
    )
    posteriors_s = result.posteriors[:s]
    final_iter = posteriors_s[end]
    last_marginal = isa(final_iter, Vector) ? final_iter[end] : final_iter
    belief = copy(last_marginal.p)
    belief = max.(belief, 1e-16)
    belief ./= sum(belief)
    return belief, Float64.(result.free_energy)
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
    E = E ./ sum(E)
    validate_dimensions(A, B, C, D)

    C_pref = softmax(C)

    # --- Online perception→action loop (the inference IS the loop) ---
    current_state = categorical_index(D)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    action_seq_full = Int[]
    obs_onehot_seq = Vector{{Vector{{Float64}}}}()

    beliefs = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()
    vfe_per_iteration = Float64[]

    for step in 1:TIME_STEPS
        observation = categorical_index(A[:, current_state])
        emitting_state = current_state  # the state that generated this observation
        push!(obs_onehot_seq, [i == observation ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS])

        # Real RxInfer filtering on the prefix y[1:step]
        belief, fe_trace = filtered_posterior(obs_onehot_seq, action_seq_full, A, B, D)
        push!(beliefs, belief)
        if step == TIME_STEPS
            vfe_per_iteration = fe_trace  # full-sequence trace
        end

        # Action selection from the FILTERED posterior (habit prior E + EFE)
        action, efe_values, policy = select_action(belief, A, B, C_pref, E)
        push!(efe_per_action, efe_values)
        push!(selected_efe, efe_values[action])
        push!(policy_posterior, policy)

        # Environment transition
        next_probs = B[:, current_state, action]
        current_state = categorical_index(next_probs)

        push!(observations, observation - 1)
        push!(true_states, emitting_state - 1)  # state that emitted observation t (matches beliefs[t])
        push!(actions, action - 1)
        push!(action_seq_full, action)
    end

    uses_real_rxinfer = true  # every belief above came from infer()

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

    is_identity_A = all(abs(A[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                        for i in 1:size(A,1), j in 1:size(A,2))
    min_entropy = is_identity_A ? 0.0 : 0.1
    belief_entropies = [belief_entropy(b) for b in beliefs]
    all_beliefs_degenerate = !isempty(belief_entropies) &&
        maximum(belief_entropies) < min_entropy

    # Filtered beliefs at step t condition on y[1:t]; true_states[t] records
    # the state that EMITTED observation t, so this is an aligned comparison.
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
            "inference_mode" => INFERENCE_MODE,
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

function write_execution_log(results)
    log_path = "simulation.log"
    beliefs = results["beliefs"]
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

function write_plots(results)
    if !PLOTS_READY
        println("⚠️ Skipping PNG plots (Plots backend not available)")
        return
    end
    try
        beliefs = results["beliefs"]
        if !isempty(beliefs)
            belief_mat = hcat(beliefs...)
            steps = 1:size(belief_mat, 2)
            p1 = plot(
                title = "Filtered Belief Evolution (online)",
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
        println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png)")
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

    # Parsed layout of B_level1 is [action][prev][next] per the repo's
    # canonical B contract (pomdp_contract.canonicalise_b_matrix and
    # _canonicalise_factored_B both apply transpose(2,1,0) to action-first
    # raw). The exemplar's own "next x prev" comment disagrees with the
    # contract, but the contract feeds the cross-framework joint
    # composition, so it is authoritative here. (Numerically identical for
    # the shipped exemplars — their action blocks are symmetric.)
    raw_B = matrices["B_level1"]
    if length(raw_B) != NUM_ACTIONS
        error("B_level1 action count $(length(raw_B)) != expected $NUM_ACTIONS")
    end
    B = zeros(Float64, NUM_FAST, NUM_FAST, NUM_ACTIONS)
    for a in 1:NUM_ACTIONS, ns in 1:NUM_FAST, ps in 1:NUM_FAST
        B[ns, ps, a] = Float64(raw_B[a][ps][ns])
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
        emitting_state = current_state  # the state that generated this observation

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
        push!(true_states, emitting_state - 1)  # state that emitted observation t (matches beliefs[t])
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


def _descriptor_name(descriptors: Any, index: int, default: str) -> str:
    """Return ``descriptors[index]["name"]`` from a GNN descriptor list."""
    if isinstance(descriptors, list) and len(descriptors) > index:
        entry = descriptors[index]
        if isinstance(entry, dict) and entry.get("name"):
            return str(entry["name"])
    return default


class FactoredStrategy(ModelStrategy):
    """Two-factor / two-modality POMDP rendered natively (roadmap D3).

    The exemplar's ``## Equations`` declare the mean-field factorization
    ``Q(s_f0, s_f1) = Q(s_f0) Q(s_f1)``, and ``factored_constraints()``
    states exactly that cut — so the posterior family IS the declared
    model, not an approximation bolted on afterwards. Factor 0 is the
    action-driven (controllable) chain over ``B_f0``; factor 1 is the
    passive chain over the static ``B_f1``. Modality 0 depends on BOTH
    factors through the 3-tensor ``A_m0``; modality 1 depends on factor 0
    alone through ``A_m1``.

    The native path REQUIRES the per-factor matrices. When they are absent
    this raises ``ValueError`` rather than dropping to the extractor's
    joint composition: a model detected FACTORED without per-factor
    matrices is a contract violation, not a case for a quieter render.
    """

    kind = ModelKind.FACTORED

    _REQUIRED_MATRICES = ("A_m0", "A_m1", "B_f0", "B_f1", "D_f0", "D_f1")

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
            "factor1_beliefs_valid",
            "factor1_beliefs_sum_to_one",
        ]

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the native two-factor mean-field RxInfer.jl script."""
        matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
        model_params = gnn_spec.get("model_parameters", {}) or {}
        missing = [key for key in self._REQUIRED_MATRICES if key not in matrices]
        num_factors = int(model_params.get("num_factors", 0))
        num_modalities = int(model_params.get("num_modalities", 0))

        problems = []
        if missing:
            problems.append(f"missing per-factor matrices {missing}")
        if num_factors != 2:
            problems.append(f"num_factors is {num_factors}, native path requires 2")
        if num_modalities != 2:
            problems.append(
                f"num_modalities is {num_modalities}, native path requires 2"
            )
        if problems:
            raise ValueError(
                f"factored model {model_name} cannot render natively: "
                + "; ".join(problems)
                + ". The joint composition is not a fallback for this kind — a "
                "factored model without per-factor matrices is a contract "
                "violation."
            )

        n_f0 = int(model_params["num_hidden_states_factor0"])
        n_f1 = int(model_params["num_hidden_states_factor1"])
        n_obs_m0 = int(model_params["num_obs_modality0"])
        n_obs_m1 = int(model_params["num_obs_modality1"])
        num_actions = int(model_params["num_actions"])
        num_timesteps = int(model_params.get("num_timesteps", 20))
        seed = int(model_params.get("random_seed", model_params.get("seed", 42)))
        action_precision = float(
            model_params.get("action_precision", model_params.get("gamma", 4.0))
        )
        inference_iterations = int(model_params.get("inference_iterations", 20))

        factor0 = _descriptor_name(model_params.get("state_factors"), 0, "s_f0")
        factor1 = _descriptor_name(model_params.get("state_factors"), 1, "s_f1")
        modality0 = _descriptor_name(
            model_params.get("observation_modalities"), 0, "o_m0"
        )
        modality1 = _descriptor_name(
            model_params.get("observation_modalities"), 1, "o_m1"
        )
        control0 = _descriptor_name(model_params.get("control_factors"), 0, "u")

        model_display_name = (
            gnn_spec.get("model_name") or gnn_spec.get("name") or model_name
        )
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl two-factor mean-field POMDP simulation — genuine @model + infer()
# Generated from GNN Model: {model_display_name}
# Generated: {_now()}
#
# Structure (matches the GNN file's declared semantics):
#   s1[t] — state factor 0 "{factor0}" ({n_f0} states), action-driven over B_f0
#   s2[t] — state factor 1 "{factor1}" ({n_f1} states), passive chain over B_f1
#   y_m0[t] — modality 0 "{modality0}" ({n_obs_m0} outcomes) depends on BOTH
#     factors through the 3-tensor A_m0 (n_obs_m0 x n_f0 x n_f1)
#   y_m1[t] — modality 1 "{modality1}" ({n_obs_m1} outcomes) depends on s1 only
#
# The exemplar's ## Equations declare Q(s_f0, s_f1) = Q(s_f0) Q(s_f1), and
# factored_constraints() states exactly that cut — the posterior family IS
# the declared model. infer() REQUIRES both that constraint and the uniform
# marginal initialization (verified against RxInfer 5.5: with neither, and
# with initialization alone, infer() dies with "Variables [ s1, s2 ] have
# not been updated after an update event").

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
const N_F0 = {n_f0}
const N_F1 = {n_f1}
const N_OBS_M0 = {n_obs_m0}
const N_OBS_M1 = {n_obs_m1}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = {num_timesteps}
const RANDOM_SEED = {seed}
const ACTION_PRECISION = {action_precision}
const INFERENCE_ITERATIONS = {inference_iterations}
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "{self.kind.value}"
const FACTOR0_NAME = {json.dumps(factor0)}
const FACTOR1_NAME = {json.dumps(factor1)}
const MODALITY0_NAME = {json.dumps(modality0)}
const MODALITY1_NAME = {json.dumps(modality1)}
const CONTROL0_NAME = {json.dumps(control0)}
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

# The factored @model, its mean-field constraints, and its marginal
# initialization are precompiled in the GnnRxInferModels package module.
using GnnRxInferModels:
    factored_pomdp_model, factored_constraints, factored_initialization

# --- Custom EFE computation (Active Inference domain, not RxInfer's) ---
# EFE is computed on FACTOR 0 — the controllable (location) factor — using
# modality 1, the modality that depends on factor 0 alone. Factor 1 is
# passive (static B_f1) so no policy acts on it and an EFE over it would be
# constant across actions.

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

# Policy: softmax(log E - gamma * EFE). With the uniform default habit prior
# E the log-term is constant and cancels inside softmax.
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

function belief_entropy(belief)
    safe = max.(belief, 1e-16)
    return -sum(safe .* log.(safe))
end

# --- Per-factor matrix loading -------------------------------------------
# Nestings below are the EXACT parsed layouts produced by
# gnn.pomdp_extractor -> render.pomdp_processor.pomdp_to_gnn_spec for
# per-factor matrices; the B_f0 transpose follows
# render/pomdp_contract.py::canonicalise_b_matrix (raw is action-first
# [action][prev][next]; canonical is [next, prev, action]).
function load_factored_matrices()
    matrices = GNN_SPEC["structured_pomdp"]["matrices"]

    # A_m0: parsed [obs][f0][f1]. That IS DiscreteTransition's (out, in, T1)
    # tensor order, so it loads with NO permutation.
    raw_A_m0 = matrices["A_m0"]
    A_m0 = zeros(Float64, N_OBS_M0, N_F0, N_F1)
    for o in 1:N_OBS_M0, i in 1:N_F0, j in 1:N_F1
        A_m0[o, i, j] = Float64(raw_A_m0[o][i][j])
    end
    for i in 1:N_F0, j in 1:N_F1
        A_m0[:, i, j] ./= sum(A_m0[:, i, j])
    end

    # A_m1: parsed [obs][f0]
    raw_A_m1 = matrices["A_m1"]
    A_m1 = zeros(Float64, N_OBS_M1, N_F0)
    for o in 1:N_OBS_M1, i in 1:N_F0
        A_m1[o, i] = Float64(raw_A_m1[o][i])
    end
    A_m1 = A_m1 ./ sum(A_m1, dims = 1)

    # B_f0: parsed [action][prev][next] -> canonical (next, prev, action)
    raw_B_f0 = matrices["B_f0"]
    if length(raw_B_f0) != NUM_ACTIONS
        error("B_f0 action count $(length(raw_B_f0)) != expected $NUM_ACTIONS")
    end
    B_f0 = zeros(Float64, N_F0, N_F0, NUM_ACTIONS)
    for a in 1:NUM_ACTIONS, p in 1:N_F0, n in 1:N_F0
        B_f0[n, p, a] = Float64(raw_B_f0[a][p][n])
    end
    for a in 1:NUM_ACTIONS, p in 1:N_F0
        B_f0[:, p, a] ./= sum(B_f0[:, p, a])
    end

    # B_f1: parsed [next][prev], static/passive factor
    raw_B_f1 = matrices["B_f1"]
    B_f1 = zeros(Float64, N_F1, N_F1)
    for n in 1:N_F1, p in 1:N_F1
        B_f1[n, p] = Float64(raw_B_f1[n][p])
    end
    B_f1 = B_f1 ./ sum(B_f1, dims = 1)

    D_f0 = Float64.(collect(matrices["D_f0"]))
    if length(D_f0) != N_F0
        error("D_f0 length $(length(D_f0)) != expected $N_F0")
    end
    D_f0 ./= sum(D_f0)

    D_f1 = Float64.(collect(matrices["D_f1"]))
    if length(D_f1) != N_F1
        error("D_f1 length $(length(D_f1)) != expected $N_F1")
    end
    D_f1 ./= sum(D_f1)

    # Preferences are not part of inference — they drive the post-hoc EFE.
    C_m0 = haskey(matrices, "C_m0") ? Float64.(collect(matrices["C_m0"])) : zeros(Float64, N_OBS_M0)
    C_m1 = haskey(matrices, "C_m1") ? Float64.(collect(matrices["C_m1"])) : zeros(Float64, N_OBS_M1)
    if length(C_m0) != N_OBS_M0
        error("C_m0 length $(length(C_m0)) != expected $N_OBS_M0")
    end
    if length(C_m1) != N_OBS_M1
        error("C_m1 length $(length(C_m1)) != expected $N_OBS_M1")
    end

    return A_m0, A_m1, B_f0, B_f1, D_f0, D_f1, C_m0, C_m1
end

# Take the LAST variational iteration's per-timestep marginals and fail loud
# if the chain length does not match the simulated horizon.
function last_iteration_marginals(posteriors, label)
    final_iter = posteriors[end]
    per_step = isa(final_iter, Vector) ? final_iter : [final_iter]
    if length(per_step) != TIME_STEPS
        error("$label posterior has $(length(per_step)) marginals, expected $TIME_STEPS")
    end
    return per_step
end

function run_simulation()
    Random.seed!(RANDOM_SEED)
    A_m0, A_m1, B_f0, B_f1, D_f0, D_f1, C_m0, C_m1 = load_factored_matrices()

    # Habit prior over the factor-0 control: uniform (the exemplar declares
    # no per-factor E).
    E = fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    C_pref = softmax(C_m1)

    # --- Phase 1: Forward simulation for data collection ---
    # The environment is sampled PER FACTOR: factor 0 is action-driven via
    # B_f0, factor 1 evolves passively via the static B_f1. Observations come
    # from A_m0[:, s1, s2] (both factors) and A_m1[:, s1] (factor 0 only).
    # The agent's forward beliefs are a mean-field pair updated by simple
    # Bayes on each modality's likelihood slice, marginalizing the OTHER
    # factor's current belief — the same Q(s_f0)Q(s_f1) family the model
    # declares. This is data collection, not the inference step.
    s1_true = categorical_index(D_f0)
    s2_true = categorical_index(D_f1)
    b_f0 = copy(D_f0)
    b_f1 = copy(D_f1)

    obs_m0 = Int[]
    obs_m1 = Int[]
    true_f0 = Int[]
    true_f1 = Int[]
    actions = Int[]
    action_seq_full = Int[]

    for step in 1:TIME_STEPS
        o0 = categorical_index(A_m0[:, s1_true, s2_true])
        o1 = categorical_index(A_m1[:, s1_true])
        emitting_s1 = s1_true  # factor states that generated these observations
        emitting_s2 = s2_true

        # Both likelihood slices are formed from the PRE-update beliefs so the
        # two factors are updated simultaneously (a mean-field sweep).
        lik_f0 = [
            sum(A_m0[o0, i, j] * b_f1[j] for j in 1:N_F1) * A_m1[o1, i]
            for i in 1:N_F0
        ]
        lik_f1 = [sum(A_m0[o0, i, j] * b_f0[i] for i in 1:N_F0) for j in 1:N_F1]

        updated_f0 = b_f0 .* lik_f0
        if sum(updated_f0) <= 0
            error("factor-0 belief update produced zero mass at step $step")
        end
        b_f0 = updated_f0 ./ sum(updated_f0)

        updated_f1 = b_f1 .* lik_f1
        if sum(updated_f1) <= 0
            error("factor-1 belief update produced zero mass at step $step")
        end
        b_f1 = updated_f1 ./ sum(updated_f1)

        # Action selection acts on the controllable factor only.
        action, efe_values, policy = select_action(b_f0, A_m1, B_f0, C_pref, E)

        s1_true = categorical_index(B_f0[:, s1_true, action])
        s2_true = categorical_index(B_f1[:, s2_true])

        predicted_f0 = B_f0[:, :, action] * b_f0
        b_f0 = predicted_f0 ./ sum(predicted_f0)
        predicted_f1 = B_f1 * b_f1
        b_f1 = predicted_f1 ./ sum(predicted_f1)

        push!(obs_m0, o0 - 1)  # 0-indexed for JSON
        push!(obs_m1, o1 - 1)
        push!(true_f0, emitting_s1 - 1)  # states that emitted observation t (match beliefs[t])
        push!(true_f1, emitting_s2 - 1)
        push!(actions, action - 1)
        push!(action_seq_full, action)  # 1-indexed for the model
    end

    # --- Phase 2: Real RxInfer factored inference (no fallback) ---
    y_m0_onehot = [[i == (o + 1) ? 1.0 : 0.0 for i in 1:N_OBS_M0] for o in obs_m0]
    y_m1_onehot = [[i == (o + 1) ? 1.0 : 0.0 for i in 1:N_OBS_M1] for o in obs_m1]
    model_actions = copy(action_seq_full)
    while length(model_actions) < TIME_STEPS
        push!(model_actions, 1)
    end

    # NO try/catch — if infer() fails, the script crashes with a clear error.
    result = infer(
        model = factored_pomdp_model(A_m0 = A_m0, A_m1 = A_m1, B_f0 = B_f0,
                                     B_f1 = B_f1, D_f0 = D_f0, D_f1 = D_f1,
                                     u = model_actions, T = TIME_STEPS),
        data = (y_m0 = y_m0_onehot, y_m1 = y_m1_onehot),
        constraints = factored_constraints(),
        initialization = factored_initialization(N_F0, N_F1),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true
    )

    uses_real_rxinfer = true

    # --- Phase 3: Per-factor smoothed marginal extraction ---
    q_s1 = last_iteration_marginals(result.posteriors[:s1], "s1")
    q_s2 = last_iteration_marginals(result.posteriors[:s2], "s2")

    beliefs_f0 = Vector{{Vector{{Float64}}}}()
    beliefs_f1 = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()

    for t in 1:TIME_STEPS
        b0 = copy(q_s1[t].p)
        b0 = max.(b0, 1e-16)
        b0 ./= sum(b0)
        push!(beliefs_f0, b0)

        b1 = copy(q_s2[t].p)
        b1 = max.(b1, 1e-16)
        b1 ./= sum(b1)
        push!(beliefs_f1, b1)

        # Phase 4: post-hoc EFE/policy from the smoothed factor-0 posterior.
        efe_vals, pol = compute_efe_and_policy(b0, A_m1, B_f0, C_pref, E)
        push!(efe_per_action, efe_vals)
        push!(selected_efe, efe_vals[action_seq_full[t]])
        push!(policy_posterior, pol)
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

    # Entropy/accuracy semantics identical to the flat generator, applied to
    # the primary (controllable) factor. Factor 1 is passive, so it only gets
    # simplex/validity checks — an accuracy gate on a static factor measures
    # the prior, not the inference.
    is_identity_A = all(abs(A_m1[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                        for i in 1:size(A_m1,1), j in 1:size(A_m1,2))
    min_entropy = is_identity_A ? 0.0 : 0.1
    belief_entropies = [belief_entropy(b) for b in beliefs_f0]
    all_beliefs_degenerate = !isempty(belief_entropies) &&
        maximum(belief_entropies) < min_entropy

    belief_accuracy = 0.0
    if length(beliefs_f0) == length(true_f0) && length(beliefs_f0) > 0
        correct = 0
        for t in 1:length(beliefs_f0)
            if argmax(beliefs_f0[t]) == (true_f0[t] + 1)
                correct += 1
            end
        end
        belief_accuracy = Float64(correct) / length(beliefs_f0)
    end
    min_accuracy = is_identity_A ? 0.5 : min(0.5, 2.0 / N_F0)
    belief_accuracy_ok = belief_accuracy >= min_accuracy
    belief_entropy_ok = !(all_beliefs_degenerate && !belief_accuracy_ok)

    factor1_accuracy = 0.0
    if length(beliefs_f1) == length(true_f1) && length(beliefs_f1) > 0
        correct = 0
        for t in 1:length(beliefs_f1)
            if argmax(beliefs_f1[t]) == (true_f1[t] + 1)
                correct += 1
            end
        end
        factor1_accuracy = Float64(correct) / length(beliefs_f1)
    end

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs_f0),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs_f0),
        "actions_in_range" => all(a -> 0 <= a < NUM_ACTIONS, actions),
        "inference_converged" => inference_converged,
        "vfe_present" => vfe_present,
        "belief_entropy_ok" => belief_entropy_ok,
        "belief_entropy_min" => isempty(belief_entropies) ? 0.0 : minimum(belief_entropies),
        "belief_entropy_mean" => isempty(belief_entropies) ? 0.0 : sum(belief_entropies) / length(belief_entropies),
        "belief_entropy_max" => isempty(belief_entropies) ? 0.0 : maximum(belief_entropies),
        "belief_accuracy" => belief_accuracy,
        "belief_accuracy_ok" => belief_accuracy_ok,
        "factor1_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs_f1),
        "factor1_beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs_f1),
        "factor1_belief_accuracy" => factor1_accuracy
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"] &&
        validation["inference_converged"] &&
        validation["vfe_present"] &&
        validation["belief_entropy_ok"] &&
        validation["belief_accuracy_ok"] &&
        validation["factor1_beliefs_valid"] &&
        validation["factor1_beliefs_sum_to_one"]

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
        "observations_by_modality" => Dict(
            MODALITY0_NAME => obs_m0,
            MODALITY1_NAME => obs_m1
        ),
        "hidden_states_by_factor" => Dict(
            FACTOR0_NAME => true_f0,
            FACTOR1_NAME => true_f1
        ),
        "actions_by_control_factor" => Dict(CONTROL0_NAME => actions),
        "beliefs_by_factor" => Dict(
            FACTOR0_NAME => beliefs_f0,
            FACTOR1_NAME => beliefs_f1
        ),
        "expected_free_energy" => selected_efe,
        "efe_per_action" => efe_per_action,
        "variational_free_energy" => variational_free_energy,
        "vfe_per_iteration" => vfe_per_iteration,
        "policy_posterior" => policy_posterior,
        # Flat-schema aliases: "observations"/"true_states"/"beliefs" carry the
        # PRIMARY (controllable) factor 0 so schema-generic consumers keep
        # working; per-factor data lives in the *_by_factor maps above.
        "observations" => obs_m1,
        "true_states" => true_f0,
        "actions" => actions,
        "beliefs" => beliefs_f0,
        "model_parameters" => Dict(
            "A_m0_shape" => collect(size(A_m0)),
            "A_m1_shape" => collect(size(A_m1)),
            "B_f0_shape" => collect(size(B_f0)),
            "B_f1_shape" => collect(size(B_f1)),
            "C_m0_shape" => [length(C_m0)],
            "C_m1_shape" => [length(C_m1)],
            "D_f0_shape" => [length(D_f0)],
            "D_f1_shape" => [length(D_f1)],
            "E_shape" => [length(E)],
            "E" => E,
            "num_factors" => 2,
            "num_modalities" => 2,
            "num_hidden_states_factor0" => N_F0,
            "num_hidden_states_factor1" => N_F1,
            "num_obs_modality0" => N_OBS_M0,
            "num_obs_modality1" => N_OBS_M1,
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
            "posterior_family" => "mean_field_factorized",
            "primary_factor" => FACTOR0_NAME,
            "efe_factor" => FACTOR0_NAME,
            "belief_accuracy" => belief_accuracy
        ),
        "metrics" => Dict(
            "expected_free_energy" => selected_efe,
            "policy_posterior" => policy_posterior,
            "belief_confidence" => [maximum(b) for b in beliefs_f0],
            "variational_free_energy" => variational_free_energy
        ),
        "validation" => validation
    )
end

# --- Structured per-step execution log (JSON Lines) ---
function write_execution_log(results)
    log_path = "simulation.log"
    by_factor = get(results, "beliefs_by_factor", Dict())
    beliefs = get(by_factor, FACTOR0_NAME, results["beliefs"])
    beliefs_f1 = get(by_factor, FACTOR1_NAME, [])
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
                "factor1_belief" => isempty(beliefs_f1) ? Float64[] : beliefs_f1[step],
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

# --- Julia-native visualization (per-factor beliefs) ---
function write_plots(results)
    if !PLOTS_READY
        println("⚠️ Skipping PNG plots (Plots backend not available)")
        return
    end
    try
        by_factor = get(results, "beliefs_by_factor", Dict())
        beliefs = get(by_factor, FACTOR0_NAME, results["beliefs"])
        beliefs_f1 = get(by_factor, FACTOR1_NAME, [])
        efe = results["expected_free_energy"]

        if !isempty(beliefs)
            belief_mat = hcat(beliefs...)
            steps = 1:size(belief_mat, 2)
            p1 = plot(
                title = "Belief Evolution — $FACTOR0_NAME (controllable factor)",
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

        if !isempty(beliefs_f1)
            f1_mat = hcat(beliefs_f1...)
            steps = 1:size(f1_mat, 2)
            p2 = plot(
                title = "Belief Evolution — $FACTOR1_NAME (passive factor)",
                xlabel = "Time step",
                ylabel = "Belief mass",
                legend = :outertopright,
                size = (900, 400),
                linewidth = 2
            )
            for state in 1:size(f1_mat, 1)
                plot!(p2, steps, f1_mat[state, :], label = "State $state")
            end
            savefig(p2, "factor1_belief_evolution.png")
        end

        if !isempty(efe)
            p3 = plot(
                1:length(efe), efe,
                title = "Expected Free Energy over Time",
                xlabel = "Time step",
                ylabel = "Action EFE",
                label = "selected EFE",
                legend = :topright,
                size = (900, 400),
                linewidth = 2
            )
            savefig(p3, "efe_over_time.png")
        end

        println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png, factor1_belief_evolution.png, efe_over_time.png)")
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


class ContinuousStrategy(ModelStrategy):
    """Linear-Gaussian state-space rendering for continuous specs (A2).

    Continuous exemplars carry an authored continuous parameterization
    (``F``/``H``/``Q``/``R`` plus ``prior_mean``/``prior_cov``) alongside
    their discretized POMDP stand-in. This strategy renders those keys onto
    ``continuous_pomdp_model`` — the LGSSM ``@model`` in
    ``GnnRxInferModels`` — which needs neither constraints nor
    initialization (it is fully conjugate; belief propagation converges in
    one sweep).

    Deriving F/H/Q/R from discrete A/B/C/D would fabricate data, so a spec
    reaching this strategy without them raises ``ValueError`` naming the
    missing keys.
    """

    kind = ModelKind.CONTINUOUS

    _REQUIRED_KEYS = ("F", "H", "Q", "R", "prior_mean", "prior_cov")

    def get_validation_fields(self) -> List[str]:
        return [
            "vfe_finite",
            "means_finite",
            "posterior_cov_psd",
            "inference_converged",
            "rmse_vs_true",
            "rmse_finite",
        ]

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the native linear-Gaussian RxInfer.jl script."""
        initial = gnn_spec.get("initialparameterization") or {}
        missing = [key for key in self._REQUIRED_KEYS if key not in initial]
        if missing:
            raise ValueError(
                f"continuous model {model_name} is missing the continuous "
                f"parameterization required for linear-Gaussian rendering: "
                f"{missing}. Deriving these from the discrete A/B/C/D "
                f"stand-in would fabricate data."
            )

        f_matrix = initial["F"]
        h_matrix = initial["H"]
        num_states = len(f_matrix)
        num_obs = len(h_matrix)
        if num_states == 0 or num_obs == 0:
            raise ValueError(
                f"continuous model {model_name}: F has {num_states} rows and H "
                f"has {num_obs} rows; both must be non-empty matrices"
            )

        model_params = gnn_spec.get("model_parameters", {}) or {}
        num_timesteps = int(model_params.get("num_timesteps", 20))
        seed = int(model_params.get("random_seed", model_params.get("seed", 42)))
        inference_iterations = int(model_params.get("inference_iterations", 20))

        model_display_name = (
            gnn_spec.get("model_name") or gnn_spec.get("name") or model_name
        )
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl linear-Gaussian state-space simulation — genuine @model + infer()
# Generated from GNN Model: {model_display_name}
# Generated: {_now()}
#
# Structure (the continuous parameterization the GNN file declares):
#   x[1]  ~ MvNormal(prior_mean, prior_cov)
#   x[t]  = F * x[t-1] + u[t-1] + N(0, Q)     ({num_states}-dim latent state)
#   y[t]  = H * x[t]           + N(0, R)      ({num_obs}-dim observation)
#
# Control input u is a sequence of ZERO vectors: the continuous formulations
# in these exemplars declare no continuous control policy (their discrete
# stand-ins carry the action set), so the continuous dynamics run passively.
        # That is a declared property of the model, not a stand-in — no EFE,
# policy posterior, or action trace is emitted, because none is defined here.
#
# continuous_pomdp_model is fully conjugate, so infer() needs neither
# constraints nor initialization and the free-energy trace is flat after one
# sweep (empirically verified against RxInfer 5.5).

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
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_obs}
const TIME_STEPS = {num_timesteps}
const RANDOM_SEED = {seed}
const INFERENCE_ITERATIONS = {inference_iterations}
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

# The linear-Gaussian @model is precompiled in the GnnRxInferModels package.
using GnnRxInferModels: continuous_pomdp_model

# --- Continuous parameterization loading ---------------------------------
# InitialParameterization matrices parse as [row][col]; vectors parse flat.

function read_matrix(raw, rows, cols, label)
    if length(raw) != rows
        error("$label has $(length(raw)) rows, expected $rows")
    end
    M = zeros(Float64, rows, cols)
    for r in 1:rows
        row = collect(raw[r])
        if length(row) != cols
            error("$label row $r has $(length(row)) entries, expected $cols")
        end
        for c in 1:cols
            M[r, c] = Float64(row[c])
        end
    end
    return M
end

function matrix_rows(M)
    return [Float64.(collect(M[r, :])) for r in 1:size(M, 1)]
end

function load_continuous_parameters()
    initial = GNN_SPEC["initialparameterization"]
    F = read_matrix(initial["F"], NUM_STATES, NUM_STATES, "F")
    H = read_matrix(initial["H"], NUM_OBSERVATIONS, NUM_STATES, "H")
    Q = read_matrix(initial["Q"], NUM_STATES, NUM_STATES, "Q")
    R = read_matrix(initial["R"], NUM_OBSERVATIONS, NUM_OBSERVATIONS, "R")
    D_cov = read_matrix(initial["prior_cov"], NUM_STATES, NUM_STATES, "prior_cov")
    D_mean = Float64.(collect(initial["prior_mean"]))
    if length(D_mean) != NUM_STATES
        error("prior_mean length $(length(D_mean)) does not match expected $NUM_STATES")
    end
    return F, H, Q, R, D_mean, D_cov
end

# Take the LAST variational iteration's per-timestep marginals and fail loud
# if the chain length does not match the simulated horizon.
function last_iteration_marginals(posteriors, label)
    final_iter = posteriors[end]
    per_step = isa(final_iter, Vector) ? final_iter : [final_iter]
    if length(per_step) != TIME_STEPS
        error("$label posterior has $(length(per_step)) marginals, expected $TIME_STEPS")
    end
    return per_step
end

function run_simulation()
    Random.seed!(RANDOM_SEED)
    F, H, Q, R, D_mean, D_cov = load_continuous_parameters()

    # --- Phase 1: Forward simulation of the true continuous trajectory ---
    # Sampled from the generative model itself, so the posterior means can be
    # scored against a known ground truth (rmse_vs_true below).
    u_seq = [zeros(Float64, NUM_STATES) for _ in 1:TIME_STEPS]
    process_noise = MvNormal(zeros(NUM_STATES), Q)
    observation_noise = MvNormal(zeros(NUM_OBSERVATIONS), R)

    true_states_continuous = Vector{{Vector{{Float64}}}}()
    observations_continuous = Vector{{Vector{{Float64}}}}()

    x = rand(MvNormal(D_mean, D_cov))
    push!(true_states_continuous, copy(x))
    push!(observations_continuous, H * x + rand(observation_noise))
    for t in 2:TIME_STEPS
        x = F * x + u_seq[t-1] + rand(process_noise)
        push!(true_states_continuous, copy(x))
        push!(observations_continuous, H * x + rand(observation_noise))
    end

    # --- Phase 2: Real RxInfer inference (no fallback, no constraints) ---
    # NO try/catch — if infer() fails, the script crashes with a clear error.
    result = infer(
        model = continuous_pomdp_model(F = F, H = H, Q = Q, R = R,
                                       D_mean = D_mean, D_cov = D_cov,
                                       u = u_seq, T = TIME_STEPS),
        data = (y = observations_continuous,),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true
    )

    uses_real_rxinfer = true

    # --- Phase 3: Posterior mean/covariance extraction ---
    q_s = last_iteration_marginals(result.posteriors[:s], "s")
    posterior_means = [Float64.(collect(mean(m))) for m in q_s]
    posterior_cov = [matrix_rows(Matrix(cov(m))) for m in q_s]

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

    # --- Validation ---
    # Bethe free energy for a continuous model is routinely NEGATIVE (it
    # carries differential-entropy terms), so the discrete generators'
    # "vfe > 0" check is WRONG here. The correct invariant is finiteness.
    vfe_finite = !isempty(vfe_per_iteration) && all(isfinite, vfe_per_iteration)
    means_finite = all(m -> all(isfinite, m), posterior_means)

    # Positive-definiteness via Cholesky with a 1e-12 jitter: smoothed
    # covariances can be numerically singular to machine precision without
    # being invalid.
    posterior_cov_psd = all(
        m -> isposdef(Symmetric(Matrix(cov(m)) + 1e-12 * I)), q_s
    )

    squared_error = 0.0
    element_count = 0
    for t in 1:TIME_STEPS
        residual = posterior_means[t] .- true_states_continuous[t]
        squared_error += sum(residual .^ 2)
        element_count += length(residual)
    end
    rmse_vs_true = element_count > 0 ? sqrt(squared_error / element_count) : 0.0
    rmse_finite = isfinite(rmse_vs_true)

    validation = Dict(
        "vfe_finite" => vfe_finite,
        "means_finite" => means_finite,
        "posterior_cov_psd" => posterior_cov_psd,
        "inference_converged" => inference_converged,
        "rmse_vs_true" => rmse_vs_true,
        "rmse_finite" => rmse_finite
    )
    validation["all_valid"] = validation["vfe_finite"] &&
        validation["means_finite"] &&
        validation["posterior_cov_psd"] &&
        validation["inference_converged"] &&
        validation["rmse_finite"]

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
        # "beliefs" carries the per-timestep posterior MEANS (T x n_states),
        # the continuous analogue of a belief trajectory. Uncertainty lives in
        # posterior_cov; the two together are the full Gaussian marginal.
        "beliefs" => posterior_means,
        "posterior_cov" => posterior_cov,
        "true_states_continuous" => true_states_continuous,
        "observations_continuous" => observations_continuous,
        # Discrete-schema slots stay EMPTY: this model declares no discrete
        # observation indices, no action set, and therefore no expected free
        # energy or policy posterior. Emitting zeros here would be fabricated
        # data. The continuous data lives in the *_continuous keys above.
        "observations" => Int[],
        "true_states" => Int[],
        "actions" => Int[],
        "expected_free_energy" => Float64[],
        "efe_per_action" => Vector{{Vector{{Float64}}}}(),
        "policy_posterior" => Vector{{Vector{{Float64}}}}(),
        "variational_free_energy" => variational_free_energy,
        "vfe_per_iteration" => vfe_per_iteration,
        "model_parameters" => Dict(
            "F_shape" => collect(size(F)),
            "H_shape" => collect(size(H)),
            "Q_shape" => collect(size(Q)),
            "R_shape" => collect(size(R)),
            "prior_mean_shape" => [length(D_mean)],
            "prior_cov_shape" => collect(size(D_cov)),
            "num_continuous_states" => NUM_STATES,
            "num_continuous_observations" => NUM_OBSERVATIONS,
            "inference_iterations" => INFERENCE_ITERATIONS,
            # Deliberately EMPTY: the GNN spec's state_factors describe the
            # exemplar's DISCRETE dual parameterization, not the continuous
            # state — "beliefs" here are posterior mean vectors, so echoing
            # the discrete factorization would make downstream per-factor
            # recovery raise on a renderer/analyzer contract violation.
            "state_factors" => [],
            "observation_modalities" => []
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
            "parameterization" => "linear_gaussian_state_space",
            "control_input" => "passive_zero_vector",
            "rmse_vs_true" => rmse_vs_true
        ),
        "metrics" => Dict(
            "expected_free_energy" => Float64[],
            "policy_posterior" => Vector{{Vector{{Float64}}}}(),
            "belief_confidence" => Float64[],
            "variational_free_energy" => variational_free_energy
        ),
        "validation" => validation
    )
end

# --- Structured per-step execution log (JSON Lines) ---
function write_execution_log(results)
    log_path = "simulation.log"
    means = results["beliefs"]
    covariances = results["posterior_cov"]
    true_states = results["true_states_continuous"]
    observations = results["observations_continuous"]
    validation = get(results, "validation", Dict())

    open(log_path, "w") do file
        for step in 1:TIME_STEPS
            record = Dict(
                "event" => "step",
                "step" => step,
                "model_name" => MODEL_NAME,
                "schema_version" => SCHEMA_VERSION,
                "posterior_mean" => means[step],
                "posterior_cov" => covariances[step],
                "true_state" => true_states[step],
                "observation" => observations[step],
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

# --- Julia-native visualization (posterior mean vs true trajectory) ---
function write_plots(results)
    if !PLOTS_READY
        println("⚠️ Skipping PNG plots (Plots backend not available)")
        return
    end
    try
        means = results["beliefs"]
        true_states = results["true_states_continuous"]
        vfe = results["vfe_per_iteration"]

        if !isempty(means)
            mean_mat = hcat(means...)
            true_mat = hcat(true_states...)
            steps = 1:size(mean_mat, 2)
            p1 = plot(
                title = "Posterior Mean vs True Continuous State",
                xlabel = "Time step",
                ylabel = "State value",
                legend = :outertopright,
                size = (900, 450),
                linewidth = 2
            )
            for dim in 1:size(mean_mat, 1)
                plot!(p1, steps, mean_mat[dim, :], label = "posterior x$dim")
                plot!(p1, steps, true_mat[dim, :], label = "true x$dim", linestyle = :dash)
            end
            savefig(p1, "belief_evolution.png")
        end

        if !isempty(vfe)
            p2 = plot(
                1:length(vfe), vfe,
                title = "Variational Free Energy per Iteration",
                xlabel = "Iteration",
                ylabel = "Bethe free energy (nats)",
                label = "VFE",
                legend = :topright,
                size = (900, 400),
                linewidth = 2
            )
            savefig(p2, "free_energy.png")
        end

        println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png, free_energy.png)")
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


class LearningStrategy(ModelStrategy):
    """Dirichlet likelihood learning rendered natively (roadmap D1).

    Specs carrying ``dirichlet_A`` pseudo-counts render onto
    ``learning_pomdp_model``, where the likelihood ``A`` is a latent
    ``DirichletCollection`` rather than a fixed constant. The structured
    mean-field cut ``q(s, A) = q(s)q(A)`` plus marginal initialization of
    both sides is required by RxInfer 5.5, and the counts must break
    column-permutation symmetry (a uniform prior converges to a
    label-switched optimum that the free energy alone will not catch —
    which is why ``a_distance_posterior`` is a hard gate).
    """

    kind = ModelKind.LEARNING

    _REQUIRED_KEYS = ("dirichlet_A", "A", "B", "C", "D")

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
            "a_learning_improved",
            "a_posterior_columns_normalized",
        ]

    def generate_model_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate the native Dirichlet-likelihood-learning RxInfer.jl script."""
        initial = gnn_spec.get("initialparameterization") or {}
        missing = [key for key in self._REQUIRED_KEYS if key not in initial]
        if missing:
            raise ValueError(
                f"learning model {model_name} is missing the parameterization "
                f"required for Dirichlet likelihood learning: {missing}"
            )

        model_params = gnn_spec.get("model_parameters", {}) or {}
        num_states = int(model_params["num_hidden_states"])
        num_observations = int(model_params["num_obs"])
        num_actions = int(model_params["num_actions"])
        num_timesteps = int(model_params.get("num_timesteps", 20))
        seed = int(model_params.get("random_seed", model_params.get("seed", 42)))
        action_precision = float(
            model_params.get("action_precision", model_params.get("gamma", 4.0))
        )
        inference_iterations = int(model_params.get("inference_iterations", 20))

        model_display_name = (
            gnn_spec.get("model_name") or gnn_spec.get("name") or model_name
        )
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl Dirichlet likelihood-learning POMDP — genuine @model + infer()
# Generated from GNN Model: {model_display_name}
# Generated: {_now()}
#
# Same state chain as the flat POMDP, but the likelihood A is a LATENT
# DirichletCollection instead of a fixed constant. dirichlet_A holds the
# prior pseudo-counts (n_obs x n_states); DirichletCollection stores
# independent Dirichlets along the FIRST dimension, so column
# dirichlet_A[:, s] is a Dirichlet over observations for state s — exactly
# the A[obs, state] orientation DiscreteTransition(s, A) expects.
#
# Two distinct likelihoods are in play, and conflating them would be the
# central modelling error here:
#   * TRUE A (the exemplar's InitialParameterization A) generates the
#     observations in Phase 1. It is the environment, and the agent never
#     sees it.
#   * The agent's WORKING likelihood is the prior mean of dirichlet_A
#     (column-normalized counts) during the forward pass, and the learned
#     posterior mean of q(A) for the post-hoc EFE. A learning agent acts on
#     what it has inferred, not on ground truth.
# a_distance_prior / a_distance_posterior measure both against true A, and
# a_learning_improved gates the run on the posterior being no worse.

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
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
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

# The learning @model, its structured mean-field constraints, and its
# marginal initialization are precompiled in the GnnRxInferModels package.
using GnnRxInferModels:
    learning_pomdp_model, learning_constraints, learning_initialization

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

function belief_entropy(belief)
    safe = max.(belief, 1e-16)
    return -sum(safe .* log.(safe))
end

function matrix_rows(M)
    return [Float64.(collect(M[r, :])) for r in 1:size(M, 1)]
end

function load_matrices()
    initial = GNN_SPEC["initialparameterization"]

    A_true = zeros(Float64, NUM_OBSERVATIONS, NUM_STATES)
    raw_A = initial["A"]
    for obs in 1:NUM_OBSERVATIONS
        row = collect(raw_A[obs])
        for state in 1:NUM_STATES
            A_true[obs, state] = Float64(row[state])
        end
    end

    # B is stored as (next_state, previous_state, action)
    raw_B = initial["B"]
    B = zeros(Float64, NUM_STATES, NUM_STATES, NUM_ACTIONS)
    for ns in 1:NUM_STATES, ps in 1:NUM_STATES, a in 1:NUM_ACTIONS
        B[ns, ps, a] = Float64(raw_B[ns][ps][a])
    end

    C = Float64.(collect(initial["C"]))
    D = Float64.(collect(initial["D"]))
    E = haskey(initial, "E") ? Float64.(collect(initial["E"])) : fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    if length(E) != NUM_ACTIONS
        error("E length $(length(E)) does not match expected $NUM_ACTIONS")
    end
    E = E ./ sum(E)

    # dirichlet_A holds RAW pseudo-counts (never normalized by the contract).
    raw_dA = initial["dirichlet_A"]
    prior_counts = zeros(Float64, NUM_OBSERVATIONS, NUM_STATES)
    for obs in 1:NUM_OBSERVATIONS
        row = collect(raw_dA[obs])
        if length(row) != NUM_STATES
            error("dirichlet_A row $obs has $(length(row)) entries, expected $NUM_STATES")
        end
        for state in 1:NUM_STATES
            prior_counts[obs, state] = Float64(row[state])
        end
    end
    if any(prior_counts .<= 0.0)
        error("dirichlet_A pseudo-counts must be strictly positive")
    end

    if size(A_true) != (NUM_OBSERVATIONS, NUM_STATES)
        error("A shape $(size(A_true)) does not match expected ($NUM_OBSERVATIONS, $NUM_STATES)")
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

    return A_true, B, C, D, E, prior_counts
end

function last_iteration_marginals(posteriors, label)
    final_iter = posteriors[end]
    per_step = isa(final_iter, Vector) ? final_iter : [final_iter]
    if length(per_step) != TIME_STEPS
        error("$label posterior has $(length(per_step)) marginals, expected $TIME_STEPS")
    end
    return per_step
end

function run_simulation()
    Random.seed!(RANDOM_SEED)
    A_true, B, C, D, E, prior_counts = load_matrices()
    C_pref = softmax(C)

    # The agent's belief about A before seeing any data: the Dirichlet prior
    # MEAN, i.e. column-normalized pseudo-counts.
    A_prior_mean = prior_counts ./ sum(prior_counts, dims = 1)

    # --- Phase 1: Forward simulation for data collection ---
    # Environment emits through TRUE A; the agent filters and selects actions
    # through A_prior_mean, because it has not learned A yet.
    current_state = categorical_index(D)
    current_belief = copy(D)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    action_seq_full = Int[]

    for step in 1:TIME_STEPS
        observation = categorical_index(A_true[:, current_state])
        emitting_state = current_state  # the state that generated this observation

        likelihood = A_prior_mean[observation, :]
        updated = current_belief .* likelihood
        if sum(updated) <= 0
            error("belief update produced zero mass at step $step")
        end
        current_belief = updated ./ sum(updated)

        action, efe_values, policy = select_action(current_belief, A_prior_mean, B, C_pref, E)

        next_probs = B[:, current_state, action]
        current_state = categorical_index(next_probs)

        predicted = B[:, :, action] * current_belief
        current_belief = predicted ./ sum(predicted)

        push!(observations, observation - 1)  # 0-indexed for JSON
        push!(true_states, emitting_state - 1)  # state that emitted observation t (matches beliefs[t])
        push!(actions, action - 1)
        push!(action_seq_full, action)  # 1-indexed for the model
    end

    # --- Phase 2: Real RxInfer inference over states AND the likelihood ---
    obs_seq = [[i == (obs + 1) ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS] for obs in observations]
    model_actions = copy(action_seq_full)
    while length(model_actions) < TIME_STEPS
        push!(model_actions, 1)
    end

    # NO try/catch — if infer() fails, the script crashes with a clear error.
    result = infer(
        model = learning_pomdp_model(dirichlet_A = prior_counts, B = B, D = D,
                                     u = model_actions, T = TIME_STEPS),
        data = (y = obs_seq,),
        constraints = learning_constraints(),
        initialization = learning_initialization(prior_counts, NUM_STATES),
        iterations = INFERENCE_ITERATIONS,
        free_energy = true,
        options = (limit_stack_depth = 500,)
    )

    uses_real_rxinfer = true

    # --- Phase 3: Posterior extraction — states AND learned likelihood ---
    posterior_per_step = last_iteration_marginals(result.posteriors[:s], "s")
    A_posterior = last(result.posteriors[:A])
    A_learned_mean = Matrix(mean(A_posterior))
    A_posterior_counts = Matrix(A_posterior.α)
    if size(A_learned_mean) != (NUM_OBSERVATIONS, NUM_STATES)
        error("learned A shape $(size(A_learned_mean)) does not match expected ($NUM_OBSERVATIONS, $NUM_STATES)")
    end

    beliefs = Vector{{Vector{{Float64}}}}()
    efe_per_action = Vector{{Vector{{Float64}}}}()
    selected_efe = Float64[]
    policy_posterior = Vector{{Vector{{Float64}}}}()

    for t in 1:TIME_STEPS
        belief = copy(posterior_per_step[t].p)
        belief = max.(belief, 1e-16)
        belief ./= sum(belief)
        push!(beliefs, belief)

        # Phase 4: post-hoc EFE/policy under the LEARNED likelihood — what the
        # agent believes after inference, not the environment's true A.
        efe_vals, pol = compute_efe_and_policy(belief, A_learned_mean, B, C_pref, E)
        push!(efe_per_action, efe_vals)
        push!(selected_efe, efe_vals[action_seq_full[t]])
        push!(policy_posterior, pol)
    end

    # --- Learning diagnostics: did q(A) move toward the true likelihood? ---
    a_distance_prior = sum(abs.(A_prior_mean .- A_true)) / length(A_true)
    a_distance_posterior = sum(abs.(A_learned_mean .- A_true)) / length(A_true)
    a_learning_improved = a_distance_posterior <= a_distance_prior
    a_posterior_columns_normalized = all(
        isapprox(sum(A_learned_mean[:, s]), 1.0; atol=1e-6) for s in 1:NUM_STATES
    )

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

    # vfe_present keeps its schema name but means FINITE here: the free energy
    # of a model with a latent Dirichlet carries differential-entropy terms,
    # so the discrete generators' "vfe > 0" check does not hold.
    vfe_present = !isempty(vfe_per_iteration) && all(isfinite, vfe_per_iteration)

    # Entropy/accuracy semantics identical to the flat generator. The identity
    # check reads the TRUE A — it asks how observable the environment is.
    is_identity_A = all(abs(A_true[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                        for i in 1:size(A_true,1), j in 1:size(A_true,2))
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
        "belief_accuracy_ok" => belief_accuracy_ok,
        "a_learning_improved" => a_learning_improved,
        "a_posterior_columns_normalized" => a_posterior_columns_normalized
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"] &&
        validation["inference_converged"] &&
        validation["vfe_present"] &&
        validation["belief_entropy_ok"] &&
        validation["belief_accuracy_ok"] &&
        validation["a_learning_improved"] &&
        validation["a_posterior_columns_normalized"]

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
        "learned_A_mean" => matrix_rows(A_learned_mean),
        "dirichlet_A_prior_counts" => matrix_rows(prior_counts),
        "dirichlet_A_posterior_counts" => matrix_rows(A_posterior_counts),
        "true_A" => matrix_rows(A_true),
        "a_distance_prior" => a_distance_prior,
        "a_distance_posterior" => a_distance_posterior,
        "model_parameters" => Dict(
            "A_shape" => collect(size(A_true)),
            "B_shape" => collect(size(B)),
            "C_shape" => [length(C)],
            "D_shape" => [length(D)],
            "E_shape" => [length(E)],
            "E" => E,
            "dirichlet_A_shape" => collect(size(prior_counts)),
            "num_states" => NUM_STATES,
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
            "learned_parameters" => ["A"],
            "belief_accuracy" => belief_accuracy
        ),
        "metrics" => Dict(
            "expected_free_energy" => selected_efe,
            "policy_posterior" => policy_posterior,
            "belief_confidence" => [maximum(b) for b in beliefs],
            "variational_free_energy" => variational_free_energy,
            "a_distance_prior" => a_distance_prior,
            "a_distance_posterior" => a_distance_posterior
        ),
        "validation" => validation
    )
end

# --- Structured per-step execution log (JSON Lines) ---
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
            "learned_A_mean" => results["learned_A_mean"],
            "a_distance_prior" => results["a_distance_prior"],
            "a_distance_posterior" => results["a_distance_posterior"],
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

# --- Julia-native visualization (beliefs + learned likelihood) ---
function write_plots(results)
    if !PLOTS_READY
        println("⚠️ Skipping PNG plots (Plots backend not available)")
        return
    end
    try
        beliefs = get(get(results, "beliefs_by_factor", Dict()), "joint_state", results["beliefs"])
        learned = results["learned_A_mean"]
        truth = results["true_A"]

        if !isempty(beliefs)
            belief_mat = hcat(beliefs...)
            steps = 1:size(belief_mat, 2)
            p1 = plot(
                title = "Belief Evolution over Time",
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

        if !isempty(learned)
            learned_mat = permutedims(hcat(learned...))
            p2 = heatmap(learned_mat,
                title = "Learned Likelihood q(A) mean",
                xlabel = "Hidden state",
                ylabel = "Observation",
                color = :viridis,
                colorbar = :right,
                size = (700, 500)
            )
            savefig(p2, "learned_likelihood.png")
        end

        if !isempty(truth)
            truth_mat = permutedims(hcat(truth...))
            learned_mat = permutedims(hcat(learned...))
            p3 = heatmap(abs.(learned_mat .- truth_mat),
                title = "|q(A) mean - true A|",
                xlabel = "Hidden state",
                ylabel = "Observation",
                color = :magma,
                colorbar = :right,
                size = (700, 500)
            )
            savefig(p3, "likelihood_error.png")
        end

        println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png, learned_likelihood.png, likelihood_error.png)")
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
