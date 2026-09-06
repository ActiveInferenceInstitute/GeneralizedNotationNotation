#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Static Perception Model

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
const MODEL_NAME = "Static Perception Model"
const NUM_STATES = 2
const NUM_OBSERVATIONS = 2
const NUM_ACTIONS = 2
const TIME_STEPS = 5
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogIkIifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJ1In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAidSIsICJ0YXJnZXQiOiAicyJ9XSwgImRlc2NyaXB0aW9uIjogIlRoZSBzaW1wbGVzdCBBY3RpdmUgSW5mZXJlbmNlIG1vZGVsIGRlbW9uc3RyYXRpbmcgcHVyZSBwZXJjZXB0aW9uOlxuLSAyIGhpZGRlbiBzdGF0ZXMgbWFwcGVkIHRvIDIgb2JzZXJ2YXRpb25zIHZpYSBhIHJlY29nbml0aW9uIG1hdHJpeCBBXG4tIFByaW9yIEQgZW5jb2RlcyBpbml0aWFsIGJlbGllZnMgb3ZlciBoaWRkZW4gc3RhdGVzXG4tIE1pbmltYWwgMi1hY3Rpb24gdHJhbnNpdGlvbiBjb21wb25lbnQgQiBzbyB0aGUgbW9kZWwgaXMgYSBjb21wbGV0ZSBQT01EUFxuKHJlbmRlcmFibGUgYW5kIGV4ZWN1dGFibGUgYnkgcHltZHAgYW5kIHRoZSBnZW5lcmFsIHNpbXVsYXRpb24gZnJhbWV3b3Jrcylcbi0gU3VpdGFibGUgYXMgYSBtaW5pbWFsIGJhc2VsaW5lIGFuZCBmb3IgdGVzdGluZyBwZXJjZXB0aW9uLW9ubHkgaW5mZXJlbmNlIiwgImdubl9zZWN0aW9uIjogIkFjdEluZlBPTURQIiwgImluaXRpYWxfcGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC44MTgxODE4MTgxODE4MTgxLCAwLjExMTExMTExMTExMTExMTEyXSwgWzAuMTgxODE4MTgxODE4MTgxODIsIDAuODg4ODg4ODg4ODg4ODg5XV0sICJCIjogW1tbMC45NSwgMC4wNV0sIFswLjA1LCAwLjk1XV0sIFtbMC4wNSwgMC45NV0sIFswLjk1LCAwLjA1XV1dLCAiQyI6IFswLjAsIDAuMF0sICJEIjogWzAuNSwgMC41XX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC44MTgxODE4MTgxODE4MTgxLCAwLjExMTExMTExMTExMTExMTEyXSwgWzAuMTgxODE4MTgxODE4MTgxODIsIDAuODg4ODg4ODg4ODg4ODg5XV0sICJCIjogW1tbMC45NSwgMC4wNV0sIFswLjA1LCAwLjk1XV0sIFtbMC4wNSwgMC45NV0sIFswLjk1LCAwLjA1XV1dLCAiQyI6IFswLjAsIDAuMF0sICJEIjogWzAuNSwgMC41XX0sICJtYXRyaXhfcHJvdmVuYW5jZSI6IHsiQSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjbGFpbWVkX3NsaWNlX2NvbnZlbnRpb24iOiAicm93c19uZXh0X2NvbHVtbnNfcHJldmlvdXMiLCAiY29udHJhZGljdGlvbiI6IGZhbHNlLCAiZGVjbGFyZWRfb3JkZXIiOiBbIm5leHRfc3RhdGUiLCAicHJldmlvdXNfc3RhdGUiLCAiYWN0aW9uIl0sICJkZXJpdmVkIjogZmFsc2UsICJkZXRlY3RlZF9vcmRlciI6IG51bGwsICJyZWFzb24iOiBudWxsLCAic2hhcGUiOiBbMiwgMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIlN0YXRpYyBQZXJjZXB0aW9uIE1vZGVsIiwgIm1vZGVsX3BhcmFtZXRlcnMiOiB7ImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQWN0aW9uIHRha2VuIiwgImRpbWVuc2lvbnMiOiBbMV0sICJpbmRleCI6IDAsICJuYW1lIjogInUiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDEsICJ0eXBlIjogImZsb2F0In1dLCAibnVtX2FjdGlvbnMiOiAyLCAibnVtX2hpZGRlbl9zdGF0ZXMiOiAyLCAibnVtX21vZGFsaXRpZXMiOiAxLCAibnVtX29icyI6IDIsICJudW1fc3RhdGVfZmFjdG9ycyI6IDEsICJudW1fdGltZXN0ZXBzIjogNSwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIk9ic2VydmF0aW9uIChvbmUtaG90IGVuY29kZWQpIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogIm8iLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDIsICJ0eXBlIjogImZsb2F0In1dLCAicGFzc2l2ZV9tb2RlbCI6IGZhbHNlLCAic2ltdWxhdGlvbl9wYXJhbXMiOiB7fSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkhpZGRlbiBzdGF0ZSAocG9zdGVyaW9yIGJlbGllZikiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAicyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV19LCAibmFtZSI6ICJTdGF0aWMgUGVyY2VwdGlvbiBNb2RlbCIsICJvbnRvbG9neV9tYXBwaW5nIjogeyJBIjogIlJlY29nbml0aW9uTWF0cml4IiwgIkIiOiAiVHJhbnNpdGlvbk1hdHJpeCIsICJDIjogIlByZWZlcmVuY2VWZWN0b3IiLCAiRCI6ICJQcmlvciIsICJvIjogIk9ic2VydmF0aW9uIiwgInMiOiAiSGlkZGVuU3RhdGUiLCAidSI6ICJBY3Rpb24ifSwgInN0cnVjdHVyZWRfcG9tZHAiOiB7ImFkYXB0ZXJfbm90ZXMiOiBbXSwgImNhbm9uaWNhbF9iX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQWN0aW9uIHRha2VuIiwgImRpbWVuc2lvbnMiOiBbMV0sICJpbmRleCI6IDAsICJuYW1lIjogInUiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDEsICJ0eXBlIjogImZsb2F0In1dLCAibWF0cmljZXMiOiB7IkEiOiBbWzAuOSwgMC4xXSwgWzAuMiwgMC44XV0sICJCIjogW1tbMC45NSwgMC4wNV0sIFswLjA1LCAwLjk1XV0sIFtbMC4wNSwgMC45NV0sIFswLjk1LCAwLjA1XV1dLCAiQyI6IFswLjAsIDAuMF0sICJEIjogWzAuNSwgMC41XX0sICJtYXRyaXhfcHJvdmVuYW5jZSI6IHsiQSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjbGFpbWVkX3NsaWNlX2NvbnZlbnRpb24iOiAicm93c19uZXh0X2NvbHVtbnNfcHJldmlvdXMiLCAiY29udHJhZGljdGlvbiI6IGZhbHNlLCAiZGVjbGFyZWRfb3JkZXIiOiBbIm5leHRfc3RhdGUiLCAicHJldmlvdXNfc3RhdGUiLCAiYWN0aW9uIl0sICJkZXJpdmVkIjogZmFsc2UsICJkZXRlY3RlZF9vcmRlciI6IG51bGwsICJyZWFzb24iOiBudWxsLCAic2hhcGUiOiBbMiwgMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJPYnNlcnZhdGlvbiAob25lLWhvdCBlbmNvZGVkKSIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkhpZGRlbiBzdGF0ZSAocG9zdGVyaW9yIGJlbGllZikiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAicyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV19LCAidmFyaWFibGVzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgKHBvc3RlcmlvciBiZWxpZWYpIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJuYW1lIjogInMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiT2JzZXJ2YXRpb24gKG9uZS1ob3QgZW5jb2RlZCkiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgIm5hbWUiOiAibyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJBY3Rpb24gdGFrZW4iLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidSIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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
