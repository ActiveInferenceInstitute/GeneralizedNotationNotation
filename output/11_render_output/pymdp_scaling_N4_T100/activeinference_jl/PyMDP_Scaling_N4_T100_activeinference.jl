#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: PyMDP Scaling N4 T100

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
const MODEL_NAME = "PyMDP Scaling N4 T100"
const NUM_STATES = 4
const NUM_OBSERVATIONS = 4
const NUM_ACTIONS = 4
const TIME_STEPS = 100
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogIkIifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRyIsICJ0YXJnZXQiOiAicGkifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJwaSIsICJ0YXJnZXQiOiAidSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkIiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJGIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAibyIsICJ0YXJnZXQiOiAiRiJ9XSwgImRlc2NyaXB0aW9uIjogIlB5TURQIHJ1bnRpbWUgc2NhbGluZyBzd2VlcCB3aXRoIG5vaXN5IG9ic2VydmF0aW9uIGFuZCBzdG9jaGFzdGljIHRyYW5zaXRpb25zLiIsICJnbm5fc2VjdGlvbiI6ICJBY3RJbmZQT01EUCIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuOTI1LCAwLjAyNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjkyNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC45MjUsIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuOTI1XV0sICJCIjogW1tbMC44NSwgMC4wNSwgMC4wNSwgMC4wNV0sIFswLjg1LCAwLjA1LCAwLjA1LCAwLjA1XSwgWzAuODUsIDAuMDUsIDAuMDUsIDAuMDVdLCBbMC44NSwgMC4wNSwgMC4wNSwgMC4wNV1dLCBbWzAuMDUsIDAuODUsIDAuMDUsIDAuMDVdLCBbMC4wNSwgMC44NSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjg1LCAwLjA1LCAwLjA1XSwgWzAuMDUsIDAuODUsIDAuMDUsIDAuMDVdXSwgW1swLjA1LCAwLjA1LCAwLjg1LCAwLjA1XSwgWzAuMDUsIDAuMDUsIDAuODUsIDAuMDVdLCBbMC4wNSwgMC4wNSwgMC44NSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjg1LCAwLjA1XV0sIFtbMC4wNSwgMC4wNSwgMC4wNSwgMC44NV0sIFswLjA1LCAwLjA1LCAwLjA1LCAwLjg1XSwgWzAuMDUsIDAuMDUsIDAuMDUsIDAuODVdLCBbMC4wNSwgMC4wNSwgMC4wNSwgMC44NV1dXSwgIkMiOiBbMC4wLCAwLjAsIDAuMCwgMy4wXSwgIkQiOiBbMC4yNSwgMC4yNSwgMC4yNSwgMC4yNV19LCAiaW5pdGlhbHBhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuOTI1LCAwLjAyNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjkyNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC45MjUsIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuOTI1XV0sICJCIjogW1tbMC44NSwgMC4wNSwgMC4wNSwgMC4wNV0sIFswLjg1LCAwLjA1LCAwLjA1LCAwLjA1XSwgWzAuODUsIDAuMDUsIDAuMDUsIDAuMDVdLCBbMC44NSwgMC4wNSwgMC4wNSwgMC4wNV1dLCBbWzAuMDUsIDAuODUsIDAuMDUsIDAuMDVdLCBbMC4wNSwgMC44NSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjg1LCAwLjA1LCAwLjA1XSwgWzAuMDUsIDAuODUsIDAuMDUsIDAuMDVdXSwgW1swLjA1LCAwLjA1LCAwLjg1LCAwLjA1XSwgWzAuMDUsIDAuMDUsIDAuODUsIDAuMDVdLCBbMC4wNSwgMC4wNSwgMC44NSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjg1LCAwLjA1XV0sIFtbMC4wNSwgMC4wNSwgMC4wNSwgMC44NV0sIFswLjA1LCAwLjA1LCAwLjA1LCAwLjg1XSwgWzAuMDUsIDAuMDUsIDAuMDUsIDAuODVdLCBbMC4wNSwgMC4wNSwgMC4wNSwgMC44NV1dXSwgIkMiOiBbMC4wLCAwLjAsIDAuMCwgMy4wXSwgIkQiOiBbMC4yNSwgMC4yNSwgMC4yNSwgMC4yNV19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY2xhaW1lZF9zbGljZV9jb252ZW50aW9uIjogbnVsbCwgImNvbnRyYWRpY3Rpb24iOiBmYWxzZSwgImRlY2xhcmVkX29yZGVyIjogWyJuZXh0X3N0YXRlIiwgInByZXZpb3VzX3N0YXRlIiwgImFjdGlvbiJdLCAiZGVyaXZlZCI6IGZhbHNlLCAiZGV0ZWN0ZWRfb3JkZXIiOiBbIm5leHRfc3RhdGUiLCAicHJldmlvdXNfc3RhdGUiLCAiYWN0aW9uIl0sICJyZWFzb24iOiBudWxsLCAic2hhcGUiOiBbNCwgNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIlB5TURQIFNjYWxpbmcgTjQgVDEwMCIsICJtb2RlbF9wYXJhbWV0ZXJzIjogeyJiX3RlbnNvcl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbNF0sICJpbmRleCI6IDAsICJuYW1lIjogInBpIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6IG51bGwsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJ1IiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm51bV9hY3Rpb25zIjogNCwgIm51bV9oaWRkZW5fc3RhdGVzIjogNCwgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiA0LCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAxLCAibnVtX3RpbWVzdGVwcyI6IDEwMCwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogIm8iLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dLCAicGFzc2l2ZV9tb2RlbCI6IGZhbHNlLCAic2ltdWxhdGlvbl9wYXJhbXMiOiB7fSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInMiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dfSwgIm5hbWUiOiAiUHlNRFAgU2NhbGluZyBONCBUMTAwIiwgIm9udG9sb2d5X21hcHBpbmciOiB7IkEiOiAiTGlrZWxpaG9vZE1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiQyI6ICJMb2dQcmVmZXJlbmNlVmVjdG9yIiwgIkQiOiAiUHJpb3JPdmVySGlkZGVuU3RhdGVzIiwgIkYiOiAiVmFyaWF0aW9uYWxGcmVlRW5lcmd5IiwgIkciOiAiRXhwZWN0ZWRGcmVlRW5lcmd5IiwgIm8iOiAiT2JzZXJ2YXRpb24iLCAicGkiOiAiUG9saWN5VmVjdG9yIiwgInMiOiAiSGlkZGVuU3RhdGUiLCAidCI6ICJUaW1lIiwgInUiOiAiQWN0aW9uIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbNF0sICJpbmRleCI6IDAsICJuYW1lIjogInBpIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6IG51bGwsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJ1IiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm1hdHJpY2VzIjogeyJBIjogW1swLjkyNSwgMC4wMjUsIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC45MjUsIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuOTI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuMDI1LCAwLjkyNV1dLCAiQiI6IFtbWzAuODUsIDAuMDUsIDAuMDUsIDAuMDVdLCBbMC44NSwgMC4wNSwgMC4wNSwgMC4wNV0sIFswLjg1LCAwLjA1LCAwLjA1LCAwLjA1XSwgWzAuODUsIDAuMDUsIDAuMDUsIDAuMDVdXSwgW1swLjA1LCAwLjg1LCAwLjA1LCAwLjA1XSwgWzAuMDUsIDAuODUsIDAuMDUsIDAuMDVdLCBbMC4wNSwgMC44NSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjg1LCAwLjA1LCAwLjA1XV0sIFtbMC4wNSwgMC4wNSwgMC44NSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjg1LCAwLjA1XSwgWzAuMDUsIDAuMDUsIDAuODUsIDAuMDVdLCBbMC4wNSwgMC4wNSwgMC44NSwgMC4wNV1dLCBbWzAuMDUsIDAuMDUsIDAuMDUsIDAuODVdLCBbMC4wNSwgMC4wNSwgMC4wNSwgMC44NV0sIFswLjA1LCAwLjA1LCAwLjA1LCAwLjg1XSwgWzAuMDUsIDAuMDUsIDAuMDUsIDAuODVdXV0sICJDIjogWzAuMCwgMC4wLCAwLjAsIDMuMF0sICJEIjogWzAuMjUsIDAuMjUsIDAuMjUsIDAuMjVdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNsYWltZWRfc2xpY2VfY29udmVudGlvbiI6IG51bGwsICJjb250cmFkaWN0aW9uIjogZmFsc2UsICJkZWNsYXJlZF9vcmRlciI6IFsibmV4dF9zdGF0ZSIsICJwcmV2aW91c19zdGF0ZSIsICJhY3Rpb24iXSwgImRlcml2ZWQiOiBmYWxzZSwgImRldGVjdGVkX29yZGVyIjogWyJuZXh0X3N0YXRlIiwgInByZXZpb3VzX3N0YXRlIiwgImFjdGlvbiJdLCAicmVhc29uIjogbnVsbCwgInNoYXBlIjogWzQsIDQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiBudWxsLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAibyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifV0sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6IG51bGwsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJzIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9XX0sICJ2YXJpYWJsZXMiOiBbeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJuYW1lIjogInMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiBudWxsLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6IG51bGwsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJvIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbNF0sICJuYW1lIjogInBpIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogbnVsbCwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInUiLCAidHlwZSI6ICJmbG9hdCJ9XX0="
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
