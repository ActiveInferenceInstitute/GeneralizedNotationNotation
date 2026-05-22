#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Time-Varying Transition Dynamics Agent

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
const MODEL_NAME = "Time-Varying Transition Dynamics Agent"
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 2
const TIME_STEPS = 10
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInNfdCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIihzX3QsIHVfdCkiLCAidGFyZ2V0IjogIkJfdCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkJfdCIsICJ0YXJnZXQiOiAic190KzEifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzX3QiLCAidGFyZ2V0IjogIkEifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJBIiwgInRhcmdldCI6ICJvX3QifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJvX3QifV0sICJkZXNjcmlwdGlvbiI6ICJBIFBPTURQIGFnZW50IG9wZXJhdGluZyBpbiBhIG5vbi1zdGF0aW9uYXJ5IGVudmlyb25tZW50LiBUaGUga2V5IGZlYXR1cmVcbmlzIHRoYXQgdGhlIHRyYW5zaXRpb24gbWF0cml4IGBCYCBpcyBpbmRleGVkIGJ5IHRpbWUgKGBCX3RgKSwgY2FwdHVyaW5nXG5keW5hbWljcyB0aGF0IGV2b2x2ZSBhY3Jvc3MgdGhlIHBsYW5uaW5nIGhvcml6b24gXHUyMDE0IGUuZy4sIHNoaWZ0aW5nIHdpbmRcbnBhdHRlcm5zIGZvciBhIHNhaWxpbmcgYWdlbnQsIG9yIGNoYW5naW5nIG9wcG9uZW50IHN0cmF0ZWd5IGluIGFcbnNlcXVlbnRpYWwgZ2FtZS5cbi0gMyBoaWRkZW4gc3RhdGVzLCAzIG9ic2VydmF0aW9ucywgMiBhY3Rpb25zXG4tIEJfdDogM0QgdHJhbnNpdGlvbiB0ZW5zb3IgcGVyIHRpbWVzdGVwIChzaGFwZTogbmV4dF9zdGF0ZSBcdTAwZDcgY3VycmVudF9zdGF0ZSBcdTAwZDcgYWN0aW9uKVxuLSBBZ2VudCBtdXN0IGFkYXB0IGJlbGllZiB1cGRhdGVzIGVhY2ggc3RlcCB0byB0aGUgY3VycmVudCBCX3Rcbi0gRXhlcmNpc2VzIHRpbWUtdmFyeWluZyBtYXRyaXggaGFuZGxpbmcgaW4gcmVuZGVyZXJzXG5UaGlzIHNhbXBsZSBwdXNoZXMgdGhlIGxhbmd1YWdlIGV4dGVuc2lvbnMgYXJvdW5kIHRpbWUtaW5kZXhlZCB0ZW5zb3JzXG5hbmQgdGVzdHMgZG93bnN0cmVhbSBjb2RlIGdlbmVyYXRpb24gd2hlbiBtYXRyaXggbGl0ZXJhbHMgYXJlXG50aW1lc3RlcC1kZXBlbmRlbnQuIiwgImluaXRpYWxfcGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC44NSwgMC4xLCAwLjA1XSwgWzAuMSwgMC44LCAwLjFdLCBbMC4wNSwgMC4xLCAwLjg1XV0sICJCIjogW1tbMC42MDAwMDAwMDAwMDAwMDAxLCAwLjFdLCBbMC4zMDAwMDAwMDAwMDAwMDAwNCwgMC4xXSwgWzAuMSwgMC42NjY2NjY2NjY2NjY2NjY3XV0sIFtbMC4zMDAwMDAwMDAwMDAwMDAwNCwgMC4xXSwgWzAuNjAwMDAwMDAwMDAwMDAwMSwgMC42XSwgWzAuMSwgMC4yNDk5OTk5OTk5OTk5OTk5N11dLCBbWzAuMSwgMC44XSwgWzAuMSwgMC4zXSwgWzAuOCwgMC4wODMzMzMzMzMzMzMzMzMzNF1dXSwgIkMiOiBbMC4wLCAwLjAsIDEuMF0sICJEIjogWzAuMzMsIDAuMzMsIDAuMzRdfSwgImluaXRpYWxwYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1swLjg1LCAwLjEsIDAuMDVdLCBbMC4xLCAwLjgsIDAuMV0sIFswLjA1LCAwLjEsIDAuODVdXSwgIkIiOiBbW1swLjYwMDAwMDAwMDAwMDAwMDEsIDAuMV0sIFswLjMwMDAwMDAwMDAwMDAwMDA0LCAwLjFdLCBbMC4xLCAwLjY2NjY2NjY2NjY2NjY2NjddXSwgW1swLjMwMDAwMDAwMDAwMDAwMDA0LCAwLjFdLCBbMC42MDAwMDAwMDAwMDAwMDAxLCAwLjZdLCBbMC4xLCAwLjI0OTk5OTk5OTk5OTk5OTk3XV0sIFtbMC4xLCAwLjhdLCBbMC4xLCAwLjNdLCBbMC44LCAwLjA4MzMzMzMzMzMzMzMzMzM0XV1dLCAiQyI6IFswLjAsIDAuMCwgMS4wXSwgIkQiOiBbMC4zMywgMC4zMywgMC4zNF19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IHRydWUsICJyZWFzb24iOiAiUHlNRFAgc3RhdGljIHRyYW5zaXRpb24gY29udHJhY3QgdXNlcyB0aGUgZGVjbGFyZWQgQl90IHRlbnNvciBmb3IgZXhlY3V0aW9uIiwgInNoYXBlIjogWzMsIDMsIDJdLCAic291cmNlIjogInRpbWVfaW5kZXhlZF90cmFuc2l0aW9uX3Byb2plY3Rpb24iLCAic291cmNlX2tleSI6ICJCX3QiLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJCX3QiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDMsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIlRpbWUtVmFyeWluZyBUcmFuc2l0aW9uIER5bmFtaWNzIEFnZW50IiwgIm1vZGVsX3BhcmFtZXRlcnMiOiB7ImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFtdLCAibnVtX2FjdGlvbnMiOiAyLCAibnVtX2hpZGRlbl9zdGF0ZXMiOiAzLCAibnVtX21vZGFsaXRpZXMiOiAxLCAibnVtX29icyI6IDMsICJudW1fc3RhdGVfZmFjdG9ycyI6IDEsICJudW1fdGltZXN0ZXBzIjogMTAsICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJPYnNlcnZhdGlvbiBhdCB0aW1lIHQiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAib190IiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XSwgInBhc3NpdmVfbW9kZWwiOiBmYWxzZSwgInNpbXVsYXRpb25fcGFyYW1zIjoge30sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgYXQgdGltZSB0IiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDEsICJuYW1lIjogInNfdCIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV19LCAibmFtZSI6ICJUaW1lLVZhcnlpbmcgVHJhbnNpdGlvbiBEeW5hbWljcyBBZ2VudCIsICJvbnRvbG9neV9tYXBwaW5nIjogeyJBIjogIkxpa2VsaWhvb2RNYXRyaXgiLCAiQl90IjogIlRpbWVWYXJ5aW5nVHJhbnNpdGlvbk1hdHJpeCIsICJDIjogIlByZWZlcmVuY2VWZWN0b3IiLCAiRCI6ICJQcmlvciIsICJvX3QiOiAiT2JzZXJ2YXRpb24iLCAic190IjogIkhpZGRlblN0YXRlIiwgInVfdCI6ICJBY3Rpb24ifSwgInN0cnVjdHVyZWRfcG9tZHAiOiB7ImFkYXB0ZXJfbm90ZXMiOiBbXSwgImNhbm9uaWNhbF9iX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFtdLCAibWF0cmljZXMiOiB7IkEiOiBbWzAuODUsIDAuMSwgMC4wNV0sIFswLjEsIDAuOCwgMC4xXSwgWzAuMDUsIDAuMSwgMC44NV1dLCAiQl90IjogW1tbMC42LCAwLjFdLCBbMC4zLCAwLjFdLCBbMC4xLCAwLjhdXSwgW1swLjMsIDAuMV0sIFswLjYsIDAuNl0sIFswLjEsIDAuM11dLCBbWzAuMSwgMC44XSwgWzAuMSwgMC4zXSwgWzAuOCwgMC4xXV1dLCAiQyI6IFswLjAsIDAuMCwgMS4wXSwgIkQiOiBbMC4zMywgMC4zMywgMC4zNF19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IHRydWUsICJyZWFzb24iOiAiUHlNRFAgc3RhdGljIHRyYW5zaXRpb24gY29udHJhY3QgdXNlcyB0aGUgZGVjbGFyZWQgQl90IHRlbnNvciBmb3IgZXhlY3V0aW9uIiwgInNoYXBlIjogWzMsIDMsIDJdLCAic291cmNlIjogInRpbWVfaW5kZXhlZF90cmFuc2l0aW9uX3Byb2plY3Rpb24iLCAic291cmNlX2tleSI6ICJCX3QiLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJCX3QiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDMsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJPYnNlcnZhdGlvbiBhdCB0aW1lIHQiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAib190IiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkhpZGRlbiBzdGF0ZSBhdCB0aW1lIHQiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic190IiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XX0sICJ2YXJpYWJsZXMiOiBbeyJjb21tZW50IjogIkluaXRpYWwgc3RhdGUgcHJpb3IiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAiRCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgYXQgdGltZSB0IiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJuYW1lIjogInNfdCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJPYnNlcnZhdGlvbiBtb2RlbCAodGltZS1pbnZhcmlhbnQpIiwgImRpbWVuc2lvbnMiOiBbMywgM10sICJuYW1lIjogIkEiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiT2JzZXJ2YXRpb24gYXQgdGltZSB0IiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJuYW1lIjogIm9fdCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJBY3Rpb24gYXQgdGltZSB0IiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJuYW1lIjogInVfdCIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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
