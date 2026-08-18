#!/usr/bin/env python3
"""Factored two-factor code-generation helpers — extracted from model_strategies.py."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict

from render.rxinfer._common import now

_REQUIRED_MATRICES = ("A_m0", "A_m1", "B_f0", "B_f1", "D_f0", "D_f1")


def _descriptor_name(descriptors: Any, index: int, default: str) -> str:
    """Return ``descriptors[index]["name"]`` from a GNN descriptor list."""
    if isinstance(descriptors, list) and len(descriptors) > index:
        entry = descriptors[index]
        if isinstance(entry, dict) and entry.get("name"):
            return str(entry["name"])
    return default


def _generate_factored_code(
    gnn_spec: Dict[str, Any], model_name: str, kind_value: str
) -> str:
    """Generate the native two-factor mean-field RxInfer.jl script."""
    matrices = (gnn_spec.get("structured_pomdp") or {}).get("matrices") or {}
    model_params = gnn_spec.get("model_parameters", {}) or {}
    missing = [key for key in _REQUIRED_MATRICES if key not in matrices]
    num_factors = int(model_params.get("num_factors", 0))
    num_modalities = int(model_params.get("num_modalities", 0))

    problems = []
    if missing:
        problems.append(f"missing per-factor matrices {missing}")
    if num_factors != 2:
        problems.append(f"num_factors is {num_factors}, native path requires 2")
    if num_modalities != 2:
        problems.append(f"num_modalities is {num_modalities}, native path requires 2")
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
    modality0 = _descriptor_name(model_params.get("observation_modalities"), 0, "o_m0")
    modality1 = _descriptor_name(model_params.get("observation_modalities"), 1, "o_m1")
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
# Generated: {now()}
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
const MODEL_KIND = "{kind_value}"
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
