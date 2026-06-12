#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Hidden Markov Model Baseline

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
const MODEL_NAME = "Hidden Markov Model Baseline"
const NUM_STATES = 4
const NUM_OBSERVATIONS = 6
const NUM_ACTIONS = 1
const TIME_STEPS = 50
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAic19wcmltZSJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkEiLCAidGFyZ2V0IjogIm8ifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJzX3ByaW1lIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAiQiJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogIkYifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJvIiwgInRhcmdldCI6ICJGIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAiYWxwaGEifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJvIiwgInRhcmdldCI6ICJhbHBoYSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogImFscGhhIiwgInRhcmdldCI6ICJzX3ByaW1lIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAic19wcmltZSIsICJ0YXJnZXQiOiAiYmV0YSJ9XSwgImRlc2NyaXB0aW9uIjogIkEgc3RhbmRhcmQgZGlzY3JldGUgSGlkZGVuIE1hcmtvdiBNb2RlbCB3aXRoOlxuLSA0IGhpZGRlbiBzdGF0ZXMgd2l0aCBNYXJrb3ZpYW4gZHluYW1pY3Ncbi0gNiBvYnNlcnZhdGlvbiBzeW1ib2xzXG4tIEZpeGVkIHRyYW5zaXRpb24gYW5kIGVtaXNzaW9uIG1hdHJpY2VzXG4tIE5vIGFjdGlvbiBzZWxlY3Rpb24gKHBhc3NpdmUgaW5mZXJlbmNlIG9ubHkpXG4tIFN1aXRhYmxlIGZvciBzZXF1ZW5jZSBtb2RlbGluZyBhbmQgc3RhdGUgZXN0aW1hdGlvbiB0YXNrcyIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuNDY2NjY2NjY2NjY2NjY2NiwgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2N10sIFswLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjQ2NjY2NjY2NjY2NjY2NjYsIDAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuMDY2NjY2NjY2NjY2NjY2NjddLCBbMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC40NjY2NjY2NjY2NjY2NjY2LCAwLjA2NjY2NjY2NjY2NjY2NjY3XSwgWzAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuNDY2NjY2NjY2NjY2NjY2Nl0sIFswLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjI2NjY2NjY2NjY2NjY2NjY2LCAwLjI2NjY2NjY2NjY2NjY2NjY2XSwgWzAuMjY2NjY2NjY2NjY2NjY2NjYsIDAuMjY2NjY2NjY2NjY2NjY2NjYsIDAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuMDY2NjY2NjY2NjY2NjY2NjddXSwgIkIiOiBbW1swLjcwMDAwMDAwMDAwMDAwMDFdLCBbMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjFdLCBbMC4xXV0sIFtbMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjcwMDAwMDAwMDAwMDAwMDFdLCBbMC4yXSwgWzAuMV1dLCBbWzAuMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjZdLCBbMC4yXV0sIFtbMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMV0sIFswLjZdXV0sICJDIjogWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCAiRCI6IFswLjI1LCAwLjI1LCAwLjI1LCAwLjI1XX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC40NjY2NjY2NjY2NjY2NjY2LCAwLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjA2NjY2NjY2NjY2NjY2NjY3XSwgWzAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuNDY2NjY2NjY2NjY2NjY2NiwgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2N10sIFswLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjA2NjY2NjY2NjY2NjY2NjY3LCAwLjQ2NjY2NjY2NjY2NjY2NjYsIDAuMDY2NjY2NjY2NjY2NjY2NjddLCBbMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC40NjY2NjY2NjY2NjY2NjY2XSwgWzAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuMDY2NjY2NjY2NjY2NjY2NjcsIDAuMjY2NjY2NjY2NjY2NjY2NjYsIDAuMjY2NjY2NjY2NjY2NjY2NjZdLCBbMC4yNjY2NjY2NjY2NjY2NjY2NiwgMC4yNjY2NjY2NjY2NjY2NjY2NiwgMC4wNjY2NjY2NjY2NjY2NjY2NywgMC4wNjY2NjY2NjY2NjY2NjY2N11dLCAiQiI6IFtbWzAuNzAwMDAwMDAwMDAwMDAwMV0sIFswLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMV0sIFswLjFdXSwgW1swLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuNzAwMDAwMDAwMDAwMDAwMV0sIFswLjJdLCBbMC4xXV0sIFtbMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuNl0sIFswLjJdXSwgW1swLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC4xXSwgWzAuNl1dXSwgIkMiOiBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sICJEIjogWzAuMjUsIDAuMjUsIDAuMjUsIDAuMjVdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs2LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDQsIDFdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IHRydWUsICJyZWFzb24iOiAiemVybyBwcmVmZXJlbmNlcyBmb3IgcGFzc2l2ZSBITU0vTWFya292IG1vZGVsIiwgInNoYXBlIjogWzZdLCAic291cmNlIjogInBhc3NpdmVfbW9kZWxfYWRhcHRlciJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIkhpZGRlbiBNYXJrb3YgTW9kZWwgQmFzZWxpbmUiLCAibW9kZWxfcGFyYW1ldGVycyI6IHsiYl90ZW5zb3Jfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY29udHJvbF9mYWN0b3JzIjogW10sICJudW1fYWN0aW9ucyI6IDEsICJudW1faGlkZGVuX3N0YXRlcyI6IDQsICJudW1fbW9kYWxpdGllcyI6IDEsICJudW1fb2JzIjogNiwgIm51bV9vYnNlcnZhdGlvbnMiOiA2LCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDUwLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiAob25lLWhvdCkiLCAiZGltZW5zaW9ucyI6IFs2LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAibyIsICJzaXplIjogNiwgInR5cGUiOiAiZmxvYXQifV0sICJwYXNzaXZlX21vZGVsIjogdHJ1ZSwgInNpbXVsYXRpb25fcGFyYW1zIjoge30sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgYmVsaWVmIChwb3N0ZXJpb3IpIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDIsICJuYW1lIjogInMiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiAzLCAibmFtZSI6ICJzX3ByaW1lIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9XX0sICJuYW1lIjogIkhpZGRlbiBNYXJrb3YgTW9kZWwgQmFzZWxpbmUiLCAib250b2xvZ3lfbWFwcGluZyI6IHsiQSI6ICJFbWlzc2lvbk1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiRCI6ICJJbml0aWFsU3RhdGVEaXN0cmlidXRpb24iLCAiRiI6ICJWYXJpYXRpb25hbEZyZWVFbmVyZ3kiLCAiYWxwaGEiOiAiRm9yd2FyZFZhcmlhYmxlIiwgImJldGEiOiAiQmFja3dhcmRWYXJpYWJsZSIsICJvIjogIk9ic2VydmF0aW9uIiwgInMiOiAiSGlkZGVuU3RhdGUiLCAic19wcmltZSI6ICJOZXh0SGlkZGVuU3RhdGUiLCAidCI6ICJUaW1lIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogWyJwYXNzaXZlX21vZGVsX3plcm9fcHJlZmVyZW5jZXMiXSwgImNhbm9uaWNhbF9iX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFtdLCAibWF0cmljZXMiOiB7IkEiOiBbWzAuNywgMC4xLCAwLjEsIDAuMV0sIFswLjEsIDAuNywgMC4xLCAwLjFdLCBbMC4xLCAwLjEsIDAuNywgMC4xXSwgWzAuMSwgMC4xLCAwLjEsIDAuN10sIFswLjEsIDAuMSwgMC40LCAwLjRdLCBbMC40LCAwLjQsIDAuMSwgMC4xXV0sICJCIjogW1swLjcsIDAuMSwgMC4xLCAwLjFdLCBbMC4xLCAwLjcsIDAuMiwgMC4xXSwgWzAuMSwgMC4xLCAwLjYsIDAuMl0sIFswLjEsIDAuMSwgMC4xLCAwLjZdXSwgIkMiOiBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sICJEIjogWzAuMjUsIDAuMjUsIDAuMjUsIDAuMjVdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs2LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDQsIDFdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IHRydWUsICJyZWFzb24iOiAiemVybyBwcmVmZXJlbmNlcyBmb3IgcGFzc2l2ZSBITU0vTWFya292IG1vZGVsIiwgInNoYXBlIjogWzZdLCAic291cmNlIjogInBhc3NpdmVfbW9kZWxfYWRhcHRlciJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIChvbmUtaG90KSIsICJkaW1lbnNpb25zIjogWzYsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvIiwgInNpemUiOiA2LCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkhpZGRlbiBzdGF0ZSBiZWxpZWYgKHBvc3RlcmlvcikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMiwgIm5hbWUiOiAicyIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIk5leHQgaGlkZGVuIHN0YXRlIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInNfcHJpbWUiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dfSwgInZhcmlhYmxlcyI6IFt7ImNvbW1lbnQiOiAiRW1pc3Npb24gbWF0cml4OiBvYnNlcnZhdGlvbnMgeCBoaWRkZW4gc3RhdGVzIiwgImRpbWVuc2lvbnMiOiBbNiwgNF0sICJuYW1lIjogIkEiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiSW5pdGlhbCBzdGF0ZSBkaXN0cmlidXRpb24gKHByaW9yKSIsICJkaW1lbnNpb25zIjogWzRdLCAibmFtZSI6ICJEIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkhpZGRlbiBzdGF0ZSBiZWxpZWYgKHBvc3RlcmlvcikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAicyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJzX3ByaW1lIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkZvcndhcmQgdmFyaWFibGUgKGJlbGllZiBwcm9wYWdhdGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAiYWxwaGEiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQmFja3dhcmQgdmFyaWFibGUiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAiYmV0YSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJEaXNjcmV0ZSB0aW1lIHN0ZXAiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIChvbmUtaG90KSIsICJkaW1lbnNpb25zIjogWzYsIDFdLCAibmFtZSI6ICJvIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlRyYW5zaXRpb24gbWF0cml4IChubyBhY3Rpb24gZGVwZW5kZW5jZSkiLCAiZGltZW5zaW9ucyI6IFs0LCA0XSwgIm5hbWUiOiAiQiIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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
