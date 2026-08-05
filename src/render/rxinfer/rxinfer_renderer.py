#!/usr/bin/env python3
"""
RxInfer.jl Renderer

Renders GNN specifications to RxInfer.jl simulation code using probabilistic programming.
This renderer creates executable RxInfer.jl simulations configured from parsed GNN POMDP specifications.

Features:
- GNN-to-RxInfer parameter extraction
- Julia probabilistic programming code generation
- Bayesian Active Inference model specification
- Pipeline integration support

Author: GNN RxInfer Integration
Date: 2024
"""

import base64
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from render.pomdp_contract import (
    CanonicalPomdpSpec,
    InitialParameterization,
    ModelKind,
    RxInferSimulationV1,
    build_canonical_pomdp_spec,
    detect_model_kind,
)


class RxInferRenderer:
    """
    RxInfer.jl renderer for generating Julia probabilistic programming code from GNN specifications.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize RxInfer renderer.

        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.logger = logging.getLogger(__name__)

    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to RxInfer.jl simulation code.

        Args:
            gnn_file_path: Path to GNN file
            output_path: Path for output RxInfer script

        Returns:
            Tuple of (success, message)
        """
        try:
            from gnn.pomdp_extractor import extract_pomdp_from_file
            from render.pomdp_processor import POMDPRenderProcessor

            pomdp_space = extract_pomdp_from_file(gnn_file_path, strict_validation=True)
            if pomdp_space is None:
                raise ValueError(f"No valid POMDP matrices found in {gnn_file_path}")
            gnn_spec = POMDPRenderProcessor(output_path.parent)._pomdp_to_gnn_spec(
                pomdp_space
            )
            rxinfer_code = self._generate_rxinfer_simulation_code(
                gnn_spec, gnn_file_path.stem
            )

            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rxinfer_code)
            _write_rxinfer_execution_metadata(output_path, gnn_spec)

            self.logger.info(f"Generated RxInfer.jl simulation: {output_path}")
            return True, "Successfully generated RxInfer.jl simulation code"

        except Exception as e:
            error_msg = f"Error rendering {gnn_file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    def _parse_gnn_content(self, content: str, model_name: str) -> Dict[str, Any]:
        """Parse GNN content into a structured dictionary (simplified parser)."""
        gnn_spec: dict[str, Any] = {
            "model_name": model_name,
            "variables": [],
            "model_parameters": {},
            "initial_parameterization": {},
        }

        # Simple parser for key sections
        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                current_section = line[3:].strip()
            elif current_section == "ModelParameters" and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        gnn_spec["model_parameters"][key] = float(value)
                    else:
                        gnn_spec["model_parameters"][key] = int(value)
                except ValueError:
                    gnn_spec["model_parameters"][key] = value

        return gnn_spec

    def _generate_rxinfer_simulation_code_simple(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """Require the canonical renderer for file-based RxInfer rendering."""
        raise ValueError(
            f"RxInfer rendering for {model_name} requires explicit POMDP matrices"
        )

    def _generate_rxinfer_simulation_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """
        Generate executable RxInfer.jl simulation code from GNN specification.

        Args:
            gnn_spec: Parsed GNN specification
            model_name: Name of the model

        Returns:
            Generated Julia code string
        """
        canonical_spec = build_canonical_pomdp_spec(gnn_spec)
        return self._generate_canonical_rxinfer_code(canonical_spec, model_name)

    def _generate_canonical_rxinfer_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
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
        model_kind = detect_model_kind(gnn_spec)
        model_kind_str = model_kind.value
        spec_json = json.dumps(gnn_spec, sort_keys=True)
        spec_json_b64 = base64.b64encode(spec_json.encode("utf-8")).decode("ascii")
        model_name_literal = json.dumps(str(model_display_name))

        code = f'''#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation — genuine @model + infer() pipeline
# Generated from GNN Model: {model_display_name}
# Generated: {self._get_timestamp()}
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

        # Action selection via EFE (forward-pass policy)
        action, efe_values, policy = select_action(current_belief, A, B, C_pref)

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
        efe_vals, pol = compute_efe_and_policy(belief, A, B, C_pref)
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

    # Belief entropy check — reject degenerate beliefs for non-identity A
    is_identity_A = all(abs(A[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                        for i in 1:size(A,1), j in 1:size(A,2))
    min_entropy = is_identity_A ? 0.0 : 0.1  # skip for fully observable
    belief_entropy_ok = all(b -> belief_entropy(b) >= min_entropy, beliefs)

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
        "actions_in_range" => all(a -> 0 <= a < NUM_ACTIONS, actions),
        "inference_converged" => inference_converged,
        "vfe_present" => vfe_present,
        "belief_entropy_ok" => belief_entropy_ok
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"] &&
        validation["inference_converged"] &&
        validation["vfe_present"] &&
        validation["belief_entropy_ok"]

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
            "num_states" => NUM_STATES,
            "num_observations" => NUM_OBSERVATIONS,
            "num_actions" => NUM_ACTIONS,
            "inference_iterations" => INFERENCE_ITERATIONS
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
            "model_kind" => MODEL_KIND
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

    # ------------------------------------------------------------------

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_gnn_to_rxinfer(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to RxInfer.jl simulation script.

    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_path: Path for output RxInfer script
        options: Optional rendering options

    Returns:
        Tuple of (success, message, warnings: List[str])
    """
    logger = logging.getLogger(__name__)

    try:
        # Validate input
        if not isinstance(gnn_spec, dict):
            return False, "Invalid GNN specification: must be a dictionary", []

        renderer = RxInferRenderer(options)

        # Get model name safely
        model_name = gnn_spec.get("name") or gnn_spec.get("model_name", "GNN_Model")

        # Generate simulation code directly from spec (using simplified working version)
        try:
            # Use the full generator with updated syntax
            rxinfer_code = renderer._generate_rxinfer_simulation_code(
                gnn_spec, model_name
            )
        except Exception as gen_error:
            logger.error(f"Code generation failed: {gen_error}")
            return False, f"Error generating RxInfer.jl code: {gen_error}", []

        # Write output file
        try:
            metadata = build_rxinfer_execution_metadata(gnn_spec)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rxinfer_code)
            if metadata:
                _write_rxinfer_execution_metadata(output_path, gnn_spec, metadata)
        except Exception as write_error:
            logger.error(f"Failed to write output file: {write_error}")
            return False, f"Error writing RxInfer.jl script: {write_error}", []

        message = f"Generated RxInfer.jl simulation script: {output_path}"
        warnings: list[Any] = []

        # Check for potential issues
        if not (
            gnn_spec.get("initial_parameterization")
            or gnn_spec.get("initialparameterization")
        ):
            warnings.append("No initial parameterization found - using defaults")

        if not gnn_spec.get("model_parameters"):
            warnings.append("No model parameters found - using inferred dimensions")

        logger.info(f"Successfully generated RxInfer.jl script for {model_name}")
        return True, message, warnings

    except Exception as e:
        logger.error(f"Unexpected error in render_gnn_to_rxinfer: {e}", exc_info=True)
        return False, f"Error generating RxInfer.jl script: {e}", []


def build_rxinfer_execution_metadata(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Build Step 12 execution metadata for declared RxInfer agent populations."""
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        return {}

    agents = _extract_declared_rxinfer_agents(initial)
    if not agents:
        return {}

    from .toml_generator import _extract_agent_topology

    topology = _extract_agent_topology(initial, agents)
    return {
        "schema": "gnn_rxinfer_execution_metadata_v1",
        "agent_count": len(agents),
        "agents": agents,
        "topology": topology,
    }


def _extract_declared_rxinfer_agents(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract explicitly declared agents without inventing default agents."""
    from .toml_generator import (
        _coerce_positive_int,
        _extract_compact_agents,
        _extract_indexed_agents,
    )

    nr_agents = _coerce_positive_int(params.get("nr_agents"))
    if nr_agents > 0:
        compact_agents = _extract_compact_agents(params, nr_agents)
        if compact_agents is not None:
            return compact_agents

        indexed_agents = _extract_indexed_agents(params, nr_agents)
        if len(indexed_agents) == nr_agents:
            return indexed_agents

        raise ValueError(
            "nr_agents was provided but agent configuration is incomplete. "
            "Provide compact agent_ids/agent_initial_positions/agent_target_positions "
            "or complete agent{i}_id/agent{i}_initial_position/agent{i}_target_position keys."
        )

    indexed_count = _infer_indexed_agent_count(params)
    if indexed_count <= 0:
        return []
    indexed_agents = _extract_indexed_agents(params, indexed_count)
    if len(indexed_agents) != indexed_count:
        raise ValueError(
            "Indexed agent configuration is incomplete. Provide complete "
            "agent{i}_id/agent{i}_initial_position/agent{i}_target_position keys."
        )
    return indexed_agents


def _infer_indexed_agent_count(params: Dict[str, Any]) -> int:
    """Infer agent count from agent{i}_... keys when nr_agents is omitted."""
    agent_indices: set[int] = set()
    for key in params:
        match = re.match(r"agent(\d+)_", str(key))
        if match:
            agent_indices.add(int(match.group(1)))
    return max(agent_indices) if agent_indices else 0


def _write_rxinfer_execution_metadata(
    output_path: Path,
    gnn_spec: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Write a sibling execution metadata JSON artifact when metadata exists."""
    metadata = (
        metadata if metadata is not None else build_rxinfer_execution_metadata(gnn_spec)
    )
    if not metadata:
        return None
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = dict(metadata)
    metadata["script_path"] = str(output_path)
    metadata["script_sha256"] = _sha256_file(output_path)
    metadata["metadata_provenance"] = "rendered_rxinfer_sidecar"
    topology = dict(metadata.get("topology") or {})
    topology.setdefault("source", str(metadata_path))
    metadata["topology"] = topology
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest for a rendered script."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
