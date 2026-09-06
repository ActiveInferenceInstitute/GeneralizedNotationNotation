#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation — genuine @model + infer() pipeline
# Generated from GNN Model: Simple Markov Chain
# Generated: 2026-09-05 20:25:28
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
const MODEL_NAME = "Simple Markov Chain"
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 1
const TIME_STEPS = 40
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const INFERENCE_ITERATIONS = 20
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "flat"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogInNfcHJpbWUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJzX3ByaW1lIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAiQiJ9XSwgImRlc2NyaXB0aW9uIjogIlRoaXMgbW9kZWwgZGVzY3JpYmVzIGEgbWluaW1hbCBkaXNjcmV0ZS10aW1lIE1hcmtvdiBDaGFpbjpcbi0gMyBzdGF0ZXMgcmVwcmVzZW50aW5nIHdlYXRoZXIgKHN1bm55LCBjbG91ZHksIHJhaW55KS5cbi0gTm8gYWN0aW9ucyBcdTIwMTQgdGhlIHN5c3RlbSBldm9sdmVzIHBhc3NpdmVseS5cbi0gT2JzZXJ2YXRpb25zID0gc3RhdGVzIGRpcmVjdGx5IChpZGVudGl0eSBtYXBwaW5nIGZvciBtb25pdG9yaW5nKS5cbi0gU3RhdGlvbmFyeSB0cmFuc2l0aW9uIG1hdHJpeCB3aXRoIHJlYWxpc3RpYyB3ZWF0aGVyIGR5bmFtaWNzLlxuLSBUZXN0cyB0aGUgc2ltcGxlc3QgbW9kZWwgc3RydWN0dXJlOiBwYXNzaXZlIHN0YXRlIGV2b2x1dGlvbiB3aXRoIG5vIGNvbnRyb2wuIiwgImdubl9zZWN0aW9uIjogIkhpZGRlbk1hcmtvdk1vZGVsIiwgImluaXRpYWxfcGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMS4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjBdXSwgIkIiOiBbW1swLjddLCBbMC4zXSwgWzAuMV1dLCBbWzAuMTk5OTk5OTk5OTk5OTk5OThdLCBbMC40XSwgWzAuM11dLCBbWzAuMDk5OTk5OTk5OTk5OTk5OTldLCBbMC4zXSwgWzAuNl1dXSwgIkMiOiBbMC4wLCAwLjAsIDAuMF0sICJEIjogWzAuNSwgMC4zLCAwLjJdfSwgImluaXRpYWxwYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCAiQiI6IFtbWzAuN10sIFswLjNdLCBbMC4xXV0sIFtbMC4xOTk5OTk5OTk5OTk5OTk5OF0sIFswLjRdLCBbMC4zXV0sIFtbMC4wOTk5OTk5OTk5OTk5OTk5OV0sIFswLjNdLCBbMC42XV1dLCAiQyI6IFswLjAsIDAuMCwgMC4wXSwgIkQiOiBbMC41LCAwLjMsIDAuMl19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY2xhaW1lZF9zbGljZV9jb252ZW50aW9uIjogbnVsbCwgImNvbnRyYWRpY3Rpb24iOiBmYWxzZSwgImRlY2xhcmVkX29yZGVyIjogWyJuZXh0X3N0YXRlIiwgInByZXZpb3VzX3N0YXRlIiwgImFjdGlvbiJdLCAiZGVyaXZlZCI6IGZhbHNlLCAiZGV0ZWN0ZWRfb3JkZXIiOiBudWxsLCAicmVhc29uIjogbnVsbCwgInNoYXBlIjogWzMsIDMsIDFdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IHRydWUsICJyZWFzb24iOiAiemVybyBwcmVmZXJlbmNlcyBmb3IgcGFzc2l2ZSBITU0vTWFya292IG1vZGVsIiwgInNoYXBlIjogWzNdLCAic291cmNlIjogInBhc3NpdmVfbW9kZWxfYWRhcHRlciJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIlNpbXBsZSBNYXJrb3YgQ2hhaW4iLCAibW9kZWxfcGFyYW1ldGVycyI6IHsiYl90ZW5zb3Jfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY29udHJvbF9mYWN0b3JzIjogW10sICJudW1fYWN0aW9ucyI6IDEsICJudW1faGlkZGVuX3N0YXRlcyI6IDMsICJudW1fbW9kYWxpdGllcyI6IDEsICJudW1fb2JzIjogMywgIm51bV9zdGF0ZV9mYWN0b3JzIjogMiwgIm51bV90aW1lc3RlcHMiOiA0MCwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAibyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV0sICJwYXNzaXZlX21vZGVsIjogdHJ1ZSwgInNpbXVsYXRpb25fcGFyYW1zIjoge30sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJDdXJyZW50IHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJzIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTmV4dCBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic19wcmltZSIsICJyb2xlIjogImJvb2trZWVwaW5nIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XX0sICJuYW1lIjogIlNpbXBsZSBNYXJrb3YgQ2hhaW4iLCAib250b2xvZ3lfbWFwcGluZyI6IHsiQSI6ICJFbWlzc2lvbk1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiRCI6ICJJbml0aWFsU3RhdGVEaXN0cmlidXRpb24iLCAibyI6ICJPYnNlcnZhdGlvbiIsICJzIjogIkhpZGRlblN0YXRlIiwgInNfcHJpbWUiOiAiTmV4dEhpZGRlblN0YXRlIiwgInQiOiAiVGltZSJ9LCAic3RydWN0dXJlZF9wb21kcCI6IHsiYWRhcHRlcl9ub3RlcyI6IFsicGFzc2l2ZV9tb2RlbF96ZXJvX3ByZWZlcmVuY2VzIl0sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbXSwgIm1hdHJpY2VzIjogeyJBIjogW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCAiQiI6IFtbMC43LCAwLjMsIDAuMV0sIFswLjIsIDAuNCwgMC4zXSwgWzAuMSwgMC4zLCAwLjZdXSwgIkMiOiBbMC4wLCAwLjAsIDAuMF0sICJEIjogWzAuNSwgMC4zLCAwLjJdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCAzXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNsYWltZWRfc2xpY2VfY29udmVudGlvbiI6IG51bGwsICJjb250cmFkaWN0aW9uIjogZmFsc2UsICJkZWNsYXJlZF9vcmRlciI6IFsibmV4dF9zdGF0ZSIsICJwcmV2aW91c19zdGF0ZSIsICJhY3Rpb24iXSwgImRlcml2ZWQiOiBmYWxzZSwgImRldGVjdGVkX29yZGVyIjogbnVsbCwgInJlYXNvbiI6IG51bGwsICJzaGFwZSI6IFszLCAzLCAxXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiIsICJzb3VyY2Vfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24ifSwgIkMiOiB7ImRlcml2ZWQiOiB0cnVlLCAicmVhc29uIjogInplcm8gcHJlZmVyZW5jZXMgZm9yIHBhc3NpdmUgSE1NL01hcmtvdiBtb2RlbCIsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJwYXNzaXZlX21vZGVsX2FkYXB0ZXIifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInMiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJzX3ByaW1lIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dfSwgInZhcmlhYmxlcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAicyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAibmFtZSI6ICJzX3ByaW1lIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkRpc2NyZXRlIHRpbWUgc3RlcCIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ0IiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAibyIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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

beliefs = Vector{Vector{Float64}}()
efe_per_action = Vector{Vector{Float64}}()
selected_efe = Float64[]
policy_posterior = Vector{Vector{Float64}}()

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
