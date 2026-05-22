#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Two State Bistable POMDP

using Pkg
using ActiveInference
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON
using Base64
using Dates

const SCHEMA_VERSION = "activeinference_jl_simulation_v1"
const MODEL_NAME = "Two State Bistable POMDP"
const NUM_STATES = 2
const NUM_OBSERVATIONS = 2
const NUM_ACTIONS = 2
const TIME_STEPS = 20
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogInNfcHJpbWUifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJCIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQyIsICJ0YXJnZXQiOiAiRyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkUiLCAidGFyZ2V0IjogIlx1MDNjMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkciLCAidGFyZ2V0IjogIlx1MDNjMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIlx1MDNjMCIsICJ0YXJnZXQiOiAidSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkIiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJ1IiwgInRhcmdldCI6ICJzX3ByaW1lIn1dLCAiZGVzY3JpcHRpb24iOiAiVGhpcyBtb2RlbCBkZXNjcmliZXMgYSBtaW5pbWFsIDItc3RhdGUgYmlzdGFibGUgUE9NRFA6XG4tIDIgaGlkZGVuIHN0YXRlczogXCJsZWZ0XCIgYW5kIFwicmlnaHRcIiBpbiBhIHN5bW1ldHJpYyBiaXN0YWJsZSBwb3RlbnRpYWwuXG4tIDIgbm9pc3kgb2JzZXJ2YXRpb25zOiB0aGUgYWdlbnQgZ2V0cyBhIG5vaXN5IHJlYWRvdXQgb2Ygd2hpY2ggc2lkZSBpdCBpcyBvbi5cbi0gMiBhY3Rpb25zOiBwdXNoLWxlZnQgb3IgcHVzaC1yaWdodC5cbi0gVGhlIGFnZW50IHByZWZlcnMgb2JzZXJ2YXRpb24gMSAoXCJyaWdodFwiKSBvdmVyIG9ic2VydmF0aW9uIDAgKFwibGVmdFwiKS5cbi0gVGVzdHMgdGhlIGFic29sdXRlIHNtYWxsZXN0IFBPTURQIHdpdGggZnVsbCBhY3RpdmUgaW5mZXJlbmNlIHN0cnVjdHVyZS4iLCAiaW5pdGlhbF9wYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1swLjgsIDAuMl0sIFswLjIsIDAuOF1dLCAiQiI6IFtbWzAuNzI3MjcyNzI3MjcyNzI3MywgMC4yNzI3MjcyNzI3MjcyNzI3XSwgWzAuMjIyMjIyMjIyMjIyMjIyMjcsIDAuNzc3Nzc3Nzc3Nzc3Nzc3OF1dLCBbWzAuMjcyNzI3MjcyNzI3MjcyNywgMC43MjcyNzI3MjcyNzI3MjczXSwgWzAuNzc3Nzc3Nzc3Nzc3Nzc3OCwgMC4yMjIyMjIyMjIyMjIyMjIyN11dXSwgIkMiOiBbMC4wLCAyLjBdLCAiRCI6IFswLjUsIDAuNV0sICJFIjogWzAuNSwgMC41XX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC44LCAwLjJdLCBbMC4yLCAwLjhdXSwgIkIiOiBbW1swLjcyNzI3MjcyNzI3MjcyNzMsIDAuMjcyNzI3MjcyNzI3MjcyN10sIFswLjIyMjIyMjIyMjIyMjIyMjI3LCAwLjc3Nzc3Nzc3Nzc3Nzc3NzhdXSwgW1swLjI3MjcyNzI3MjcyNzI3MjcsIDAuNzI3MjcyNzI3MjcyNzI3M10sIFswLjc3Nzc3Nzc3Nzc3Nzc3NzgsIDAuMjIyMjIyMjIyMjIyMjIyMjddXV0sICJDIjogWzAuMCwgMi4wXSwgIkQiOiBbMC41LCAwLjVdLCAiRSI6IFswLjUsIDAuNV19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzIsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkUiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAibW9kZWxfbmFtZSI6ICJUd28gU3RhdGUgQmlzdGFibGUgUE9NRFAiLCAibW9kZWxfcGFyYW1ldGVycyI6IHsiYl90ZW5zb3Jfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY29udHJvbF9mYWN0b3JzIjogW3siY29tbWVudCI6ICJQb2xpY3kgb3ZlciBhY3Rpb25zIiwgImRpbWVuc2lvbnMiOiBbMl0sICJpbmRleCI6IDIsICJuYW1lIjogIlx1MDNjMCIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkNob3NlbiBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFsxXSwgImluZGV4IjogMywgIm5hbWUiOiAidSIsICJzaXplIjogMSwgInR5cGUiOiAiZmxvYXQifV0sICJudW1fYWN0aW9ucyI6IDIsICJudW1faGlkZGVuX3N0YXRlcyI6IDIsICJudW1fbW9kYWxpdGllcyI6IDEsICJudW1fb2JzIjogMiwgIm51bV9zdGF0ZV9mYWN0b3JzIjogMiwgIm51bV90aW1lc3RlcHMiOiAyMCwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAibyIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV0sICJwYXNzaXZlX21vZGVsIjogZmFsc2UsICJzaW11bGF0aW9uX3BhcmFtcyI6IHt9LCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBzdGF0ZSBiZWxpZWYiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMiwgIm5hbWUiOiAicyIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIk5leHQgc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInNfcHJpbWUiLCAic2l6ZSI6IDIsICJ0eXBlIjogImZsb2F0In1dfSwgIm5hbWUiOiAiVHdvIFN0YXRlIEJpc3RhYmxlIFBPTURQIiwgIm9udG9sb2d5X21hcHBpbmciOiB7IkEiOiAiTGlrZWxpaG9vZE1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiQyI6ICJMb2dQcmVmZXJlbmNlVmVjdG9yIiwgIkQiOiAiUHJpb3JPdmVySGlkZGVuU3RhdGVzIiwgIkUiOiAiSGFiaXQiLCAiRyI6ICJFeHBlY3RlZEZyZWVFbmVyZ3kiLCAibyI6ICJPYnNlcnZhdGlvbiIsICJzIjogIkhpZGRlblN0YXRlIiwgInNfcHJpbWUiOiAiTmV4dEhpZGRlblN0YXRlIiwgInQiOiAiVGltZSIsICJ1IjogIkFjdGlvbiIsICJcdTAzYzAiOiAiUG9saWN5VmVjdG9yIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFsyXSwgImluZGV4IjogMiwgIm5hbWUiOiAiXHUwM2MwIiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ2hvc2VuIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAzLCAibmFtZSI6ICJ1IiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm1hdHJpY2VzIjogeyJBIjogW1swLjgsIDAuMl0sIFswLjIsIDAuOF1dLCAiQiI6IFtbWzAuOCwgMC4zXSwgWzAuMiwgMC43XV0sIFtbMC4zLCAwLjhdLCBbMC43LCAwLjJdXV0sICJDIjogWzAuMCwgMi4wXSwgIkQiOiBbMC41LCAwLjVdLCAiRSI6IFswLjUsIDAuNV19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzIsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkUiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvIiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJpbmRleCI6IDIsICJuYW1lIjogInMiLCAic2l6ZSI6IDIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IHN0YXRlIGJlbGllZiIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAiaW5kZXgiOiAzLCAibmFtZSI6ICJzX3ByaW1lIiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9XX0sICJ2YXJpYWJsZXMiOiBbeyJjb21tZW50IjogIk5vaXN5IG9ic2VydmF0aW9uIG9mIHN0YXRlIiwgImRpbWVuc2lvbnMiOiBbMiwgMl0sICJuYW1lIjogIkEiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJpb3Igb3ZlciBpbml0aWFsIHN0YXRlcyIsICJkaW1lbnNpb25zIjogWzJdLCAibmFtZSI6ICJEIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkN1cnJlbnQgc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJuYW1lIjogInMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTmV4dCBzdGF0ZSBiZWxpZWYiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgIm5hbWUiOiAic19wcmltZSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJEaXNjcmV0ZSB0aW1lIHN0ZXAiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJuYW1lIjogIm8iLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQWN0aW9uLWRlcGVuZGVudCB0cmFuc2l0aW9ucyIsICJkaW1lbnNpb25zIjogWzIsIDIsIDJdLCAibmFtZSI6ICJCIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlByaW9yIG92ZXIgYWN0aW9ucyIsICJkaW1lbnNpb25zIjogWzJdLCAibmFtZSI6ICJFIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlBvbGljeSBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFsyXSwgIm5hbWUiOiAiXHUwM2MwIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkNob3NlbiBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidSIsICJ0eXBlIjogImZsb2F0In1dfQ=="
const GNN_SPEC = JSON.parse(String(base64decode(GNN_SPEC_JSON_B64)))

function package_version(name::String)
    for (_, dep) in Pkg.dependencies()
        if dep.name == name
            return string(dep.version)
        end
    end
    return "unknown"
end

function to_float_matrix(raw)
    rows = collect(raw)
    matrix = zeros(Float64, length(rows), length(collect(rows[1])))
    for row in eachindex(rows)
        values = collect(rows[row])
        for column in eachindex(values)
            matrix[row, column] = Float64(values[column])
        end
    end
    return matrix
end

function to_float_tensor(raw)
    blocks = collect(raw)
    rows = length(blocks)
    columns = length(collect(blocks[1]))
    actions = length(collect(collect(blocks[1])[1]))
    tensor = zeros(Float64, rows, columns, actions)
    for next_state in 1:rows
        block = collect(blocks[next_state])
        for previous_state in 1:columns
            values = collect(block[previous_state])
            for action in 1:actions
                tensor[next_state, previous_state, action] = Float64(values[action])
            end
        end
    end
    return tensor
end

function normalize_vector(values)
    vector = Float64.(collect(values))
    total = sum(vector)
    if !isfinite(total) || total <= 0
        error("probability vector has invalid mass")
    end
    return vector ./ total
end

function normalize_columns!(matrix)
    for column in 1:size(matrix, 2)
        total = sum(matrix[:, column])
        if !isfinite(total) || total <= 0
            error("matrix column has invalid probability mass")
        end
        matrix[:, column] ./= total
    end
    return matrix
end

function normalize_tensor!(tensor)
    for action in 1:size(tensor, 3)
        for previous_state in 1:size(tensor, 2)
            total = sum(tensor[:, previous_state, action])
            if !isfinite(total) || total <= 0
                error("transition column has invalid probability mass")
            end
            tensor[:, previous_state, action] ./= total
        end
    end
    return tensor
end

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

function run_simulation()
    Random.seed!(RANDOM_SEED)
    initial = GNN_SPEC["initialparameterization"]
    A = normalize_columns!(to_float_matrix(initial["A"]))
    B = normalize_tensor!(to_float_tensor(initial["B"]))
    C = Float64.(collect(initial["C"]))
    D = normalize_vector(initial["D"])
    E = haskey(initial, "E") ? normalize_vector(initial["E"]) : fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
    validate_dimensions(A, B, C, D)

    C_pref = softmax(C)
    current_state = categorical_index(D)
    current_belief = copy(D)

    observations = Int[]
    true_states = Int[]
    actions = Int[]
    beliefs = Vector{Vector{Float64}}()
    efe_per_action = Vector{Vector{Float64}}()
    selected_efe = Float64[]
    policy_posterior = Vector{Vector{Float64}}()

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
        push!(beliefs, copy(current_belief))
        push!(efe_per_action, copy(efe_values))
        push!(selected_efe, efe_values[action])
        push!(policy_posterior, copy(policy))
    end

    validation = Dict(
        "all_beliefs_valid" => all(b -> all(v -> 0.0 <= v <= 1.0, b), beliefs),
        "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0; atol=1e-6), beliefs),
        "actions_in_range" => all(a -> 0 <= a < NUM_ACTIONS, actions),
        "all_valid" => true
    )
    validation["all_valid"] = validation["all_beliefs_valid"] &&
        validation["beliefs_sum_to_one"] &&
        validation["actions_in_range"]

    return Dict(
        "schema_version" => SCHEMA_VERSION,
        "success" => true,
        "framework" => "ActiveInference.jl",
        "model_name" => MODEL_NAME,
        "num_timesteps" => TIME_STEPS,
        "observations_by_modality" => Dict("joint_observation" => observations),
        "hidden_states_by_factor" => Dict("joint_state" => true_states),
        "actions_by_control_factor" => Dict("joint_action" => actions),
        "beliefs_by_factor" => Dict("joint_state" => beliefs),
        "expected_free_energy" => selected_efe,
        "efe_per_action" => efe_per_action,
        "variational_free_energy" => Float64[],
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
            "num_actions" => NUM_ACTIONS
        ),
        "matrix_provenance" => get(GNN_SPEC, "matrix_provenance", Dict()),
        "runtime_metadata" => Dict(
            "random_seed" => RANDOM_SEED,
            "schema_version" => SCHEMA_VERSION,
            "generated_at" => string(now()),
            "activeinference_jl_version" => package_version("ActiveInference"),
            "julia_version" => string(VERSION)
        ),
        "metrics" => Dict(
            "expected_free_energy" => selected_efe,
            "policy_posterior" => policy_posterior,
            "belief_confidence" => [maximum(b) for b in beliefs]
        ),
        "validation" => validation
    )
end

function main()
    results = run_simulation()
    open("simulation_results.json", "w") do file
        JSON.print(file, results, 2)
    end
    println("ActiveInference.jl simulation wrote simulation_results.json")
    return results["validation"]["all_valid"] ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
