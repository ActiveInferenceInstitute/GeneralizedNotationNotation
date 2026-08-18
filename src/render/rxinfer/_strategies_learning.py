#!/usr/bin/env python3
"""Dirichlet likelihood-learning code-generation helpers — extracted from model_strategies.py."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict

from render.rxinfer._common import now

_REQUIRED_KEYS = ("dirichlet_A", "A", "B", "C", "D")


def _generate_learning_code(
    gnn_spec: Dict[str, Any], model_name: str, kind_value: str
) -> str:
    """Generate the native Dirichlet-likelihood-learning RxInfer.jl script."""
    initial = gnn_spec.get("initialparameterization") or {}
    missing = [key for key in _REQUIRED_KEYS if key not in initial]
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
# Generated: {now()}
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
const MODEL_KIND = "{kind_value}"
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
