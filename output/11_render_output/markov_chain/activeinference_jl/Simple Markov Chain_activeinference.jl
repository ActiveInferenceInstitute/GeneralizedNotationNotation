#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Simple Markov Chain

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
const MODEL_NAME = "Simple Markov Chain"
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 1
const TIME_STEPS = 40
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogInNfcHJpbWUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJzX3ByaW1lIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAiQiJ9XSwgImRlc2NyaXB0aW9uIjogIlRoaXMgbW9kZWwgZGVzY3JpYmVzIGEgbWluaW1hbCBkaXNjcmV0ZS10aW1lIE1hcmtvdiBDaGFpbjpcbi0gMyBzdGF0ZXMgcmVwcmVzZW50aW5nIHdlYXRoZXIgKHN1bm55LCBjbG91ZHksIHJhaW55KS5cbi0gTm8gYWN0aW9ucyBcdTIwMTQgdGhlIHN5c3RlbSBldm9sdmVzIHBhc3NpdmVseS5cbi0gT2JzZXJ2YXRpb25zID0gc3RhdGVzIGRpcmVjdGx5IChpZGVudGl0eSBtYXBwaW5nIGZvciBtb25pdG9yaW5nKS5cbi0gU3RhdGlvbmFyeSB0cmFuc2l0aW9uIG1hdHJpeCB3aXRoIHJlYWxpc3RpYyB3ZWF0aGVyIGR5bmFtaWNzLlxuLSBUZXN0cyB0aGUgc2ltcGxlc3QgbW9kZWwgc3RydWN0dXJlOiBwYXNzaXZlIHN0YXRlIGV2b2x1dGlvbiB3aXRoIG5vIGNvbnRyb2wuIiwgImluaXRpYWxfcGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMS4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjBdXSwgIkIiOiBbW1swLjddLCBbMC4zXSwgWzAuMV1dLCBbWzAuMTk5OTk5OTk5OTk5OTk5OThdLCBbMC40XSwgWzAuM11dLCBbWzAuMDk5OTk5OTk5OTk5OTk5OTldLCBbMC4zXSwgWzAuNl1dXSwgIkMiOiBbMC4wLCAwLjAsIDAuMF0sICJEIjogWzAuNSwgMC4zLCAwLjJdfSwgImluaXRpYWxwYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCAiQiI6IFtbWzAuN10sIFswLjNdLCBbMC4xXV0sIFtbMC4xOTk5OTk5OTk5OTk5OTk5OF0sIFswLjRdLCBbMC4zXV0sIFtbMC4wOTk5OTk5OTk5OTk5OTk5OV0sIFswLjNdLCBbMC42XV1dLCAiQyI6IFswLjAsIDAuMCwgMC4wXSwgIkQiOiBbMC41LCAwLjMsIDAuMl19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMywgMywgMV0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogdHJ1ZSwgInJlYXNvbiI6ICJ6ZXJvIHByZWZlcmVuY2VzIGZvciBwYXNzaXZlIEhNTS9NYXJrb3YgbW9kZWwiLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAicGFzc2l2ZV9tb2RlbF9hZGFwdGVyIn0sICJEIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiU2ltcGxlIE1hcmtvdiBDaGFpbiIsICJtb2RlbF9wYXJhbWV0ZXJzIjogeyJiX3RlbnNvcl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbXSwgIm51bV9hY3Rpb25zIjogMSwgIm51bV9oaWRkZW5fc3RhdGVzIjogMywgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiAzLCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDQwLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJvIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XSwgInBhc3NpdmVfbW9kZWwiOiB0cnVlLCAic2ltdWxhdGlvbl9wYXJhbXMiOiB7fSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDEsICJuYW1lIjogInMiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAyLCAibmFtZSI6ICJzX3ByaW1lIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XX0sICJuYW1lIjogIlNpbXBsZSBNYXJrb3YgQ2hhaW4iLCAib250b2xvZ3lfbWFwcGluZyI6IHsiQSI6ICJFbWlzc2lvbk1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiRCI6ICJJbml0aWFsU3RhdGVEaXN0cmlidXRpb24iLCAibyI6ICJPYnNlcnZhdGlvbiIsICJzIjogIkhpZGRlblN0YXRlIiwgInNfcHJpbWUiOiAiTmV4dEhpZGRlblN0YXRlIiwgInQiOiAiVGltZSJ9LCAic3RydWN0dXJlZF9wb21kcCI6IHsiYWRhcHRlcl9ub3RlcyI6IFsicGFzc2l2ZV9tb2RlbF96ZXJvX3ByZWZlcmVuY2VzIl0sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbXSwgIm1hdHJpY2VzIjogeyJBIjogW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCAiQiI6IFtbMC43LCAwLjMsIDAuMV0sIFswLjIsIDAuNCwgMC4zXSwgWzAuMSwgMC4zLCAwLjZdXSwgIkMiOiBbMC4wLCAwLjAsIDAuMF0sICJEIjogWzAuNSwgMC4zLCAwLjJdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCAzXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDMsIDFdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IHRydWUsICJyZWFzb24iOiAiemVybyBwcmVmZXJlbmNlcyBmb3IgcGFzc2l2ZSBITU0vTWFya292IG1vZGVsIiwgInNoYXBlIjogWzNdLCAic291cmNlIjogInBhc3NpdmVfbW9kZWxfYWRhcHRlciJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDEsICJuYW1lIjogIm8iLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dLCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAicyIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIk5leHQgc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDIsICJuYW1lIjogInNfcHJpbWUiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dfSwgInZhcmlhYmxlcyI6IFt7ImNvbW1lbnQiOiAiUHJpb3Igb3ZlciBpbml0aWFsIHN0YXRlcyIsICJkaW1lbnNpb25zIjogWzNdLCAibmFtZSI6ICJEIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkN1cnJlbnQgc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJuYW1lIjogInMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTmV4dCBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAic19wcmltZSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJEaXNjcmV0ZSB0aW1lIHN0ZXAiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJPYnNlcnZhdGlvbiBtb2RlbCAoaWRlbnRpdHkgZm9yIGRpcmVjdCBtb25pdG9yaW5nKSIsICJkaW1lbnNpb25zIjogWzMsIDNdLCAibmFtZSI6ICJBIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAibyIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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
