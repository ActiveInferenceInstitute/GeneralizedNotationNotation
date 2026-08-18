#!/usr/bin/env python3
"""Hierarchical two-level code-generation helpers — extracted from model_strategies.py."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict

from render.rxinfer._common import now


def _generate_two_level_code(
    gnn_spec: Dict[str, Any], model_name: str, kind_value: str
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
# Generated: {now()}
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
