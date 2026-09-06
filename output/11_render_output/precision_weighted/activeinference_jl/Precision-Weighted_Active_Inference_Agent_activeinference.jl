#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Precision-Weighted Active Inference Agent

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
const MODEL_NAME = "Precision-Weighted Active Inference Agent"
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 3
const TIME_STEPS = 30
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIlx1MDNjOSIsICJ0YXJnZXQiOiAiQSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogInNfcHJpbWUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRyIsICJ0YXJnZXQiOiAiXHUwM2MwIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiXHUwM2IzIiwgInRhcmdldCI6ICJcdTAzYzAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJcdTAzYjIiLCAidGFyZ2V0IjogIlx1MDNjMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkUiLCAidGFyZ2V0IjogIlx1MDNjMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIlx1MDNjMCIsICJ0YXJnZXQiOiAidSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkIiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJ1IiwgInRhcmdldCI6ICJzX3ByaW1lIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAiRiJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIm8iLCAidGFyZ2V0IjogIkYifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJcdTAzYzkiLCAidGFyZ2V0IjogIkYifV0sICJkZXNjcmlwdGlvbiI6ICJBbiBBY3RpdmUgSW5mZXJlbmNlIGFnZW50IHdpdGggZXhwbGljaXQgcHJlY2lzaW9uIHBhcmFtZXRlcnM6XG4tIFx1MDNjOSAob21lZ2EpOiBzZW5zb3J5IHByZWNpc2lvbiB3ZWlnaHRpbmcgbGlrZWxpaG9vZCBjb25maWRlbmNlXG4tIFx1MDNiMyAoZ2FtbWEpOiBwb2xpY3kgcHJlY2lzaW9uIGNvbnRyb2xsaW5nIGFjdGlvbiByYW5kb21uZXNzXG4tIFx1MDNiMiAoYmV0YSk6IGludmVyc2UgdGVtcGVyYXR1cmUgZm9yIHBvbGljeSBzZWxlY3Rpb24gKHNvZnRtYXgpXG4tIDMgaGlkZGVuIHN0YXRlcywgMyBvYnNlcnZhdGlvbnMsIDMgYWN0aW9ucyAoc2FtZSB0b3BvbG9neSBhcyBiYXNlIFBPTURQKVxuLSBQcmVjaXNpb24gcGFyYW1ldGVycyBlbmFibGUgbW9kZWxpbmcgb2YgYXR0ZW50aW9uIGFuZCBjb25maWRlbmNlIiwgImdubl9zZWN0aW9uIjogIkFjdEluZlBPTURQIiwgImluaXRpYWxfcGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC45LCAwLjA1LCAwLjA1XSwgWzAuMDUsIDAuOSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjldXSwgIkIiOiBbW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCBbWzAuMCwgMS4wLCAwLjBdLCBbMS4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzEuMCwgMS4wLCAwLjBdXV0sICJDIjogWzAuMSwgMC4xLCAxLjBdLCAiRCI6IFswLjMzMzMzMzMzMzMzMzMzMzMsIDAuMzMzMzMzMzMzMzMzMzMzMywgMC4zMzMzMzMzMzMzMzMzMzMzXSwgIkUiOiBbMC4zMzMzMzMzMzMzMzMzMzMzLCAwLjMzMzMzMzMzMzMzMzMzMzMsIDAuMzMzMzMzMzMzMzMzMzMzM10sICJcdTAzYjIiOiBbMC41XSwgIlx1MDNiMyI6IFsyLjBdLCAiXHUwM2M5IjogWzQuMF19LCAiaW5pdGlhbHBhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuOSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjksIDAuMDVdLCBbMC4wNSwgMC4wNSwgMC45XV0sICJCIjogW1tbMS4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjBdXSwgW1swLjAsIDEuMCwgMC4wXSwgWzEuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDEuMCwgMC4wXV1dLCAiQyI6IFswLjEsIDAuMSwgMS4wXSwgIkQiOiBbMC4zMzMzMzMzMzMzMzMzMzMzLCAwLjMzMzMzMzMzMzMzMzMzMzMsIDAuMzMzMzMzMzMzMzMzMzMzM10sICJFIjogWzAuMzMzMzMzMzMzMzMzMzMzMywgMC4zMzMzMzMzMzMzMzMzMzMzLCAwLjMzMzMzMzMzMzMzMzMzMzNdLCAiXHUwM2IyIjogWzAuNV0sICJcdTAzYjMiOiBbMi4wXSwgIlx1MDNjOSI6IFs0LjBdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCAzXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNsYWltZWRfc2xpY2VfY29udmVudGlvbiI6IG51bGwsICJjb250cmFkaWN0aW9uIjogZmFsc2UsICJkZWNsYXJlZF9vcmRlciI6IFsibmV4dF9zdGF0ZSIsICJwcmV2aW91c19zdGF0ZSIsICJhY3Rpb24iXSwgImRlcml2ZWQiOiBmYWxzZSwgImRldGVjdGVkX29yZGVyIjogbnVsbCwgInJlYXNvbiI6IG51bGwsICJzaGFwZSI6IFszLCAzLCAzXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiIsICJzb3VyY2Vfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24ifSwgIkMiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIlByZWNpc2lvbi1XZWlnaHRlZCBBY3RpdmUgSW5mZXJlbmNlIEFnZW50IiwgIm1vZGVsX3BhcmFtZXRlcnMiOiB7ImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzNdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJcdTAzYzAiLCAicm9sZSI6ICJib29ra2VlcGluZyIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlNlbGVjdGVkIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJ1IiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm51bV9hY3Rpb25zIjogMywgIm51bV9oaWRkZW5fc3RhdGVzIjogMywgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiAzLCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDMwLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XSwgInBhc3NpdmVfbW9kZWwiOiBmYWxzZSwgInBvbGljeV9wcmVjaXNpb24iOiAyLjAsICJzZW5zb3J5X3ByZWNpc2lvbiI6IDQuMCwgInNpbXVsYXRpb25fcGFyYW1zIjoge30sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInMiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJzX3ByaW1lIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dfSwgIm5hbWUiOiAiUHJlY2lzaW9uLVdlaWdodGVkIEFjdGl2ZSBJbmZlcmVuY2UgQWdlbnQiLCAib250b2xvZ3lfbWFwcGluZyI6IHsiQSI6ICJMaWtlbGlob29kTWF0cml4IiwgIkIiOiAiVHJhbnNpdGlvbk1hdHJpeCIsICJDIjogIkxvZ1ByZWZlcmVuY2VWZWN0b3IiLCAiRCI6ICJQcmlvck92ZXJIaWRkZW5TdGF0ZXMiLCAiRSI6ICJIYWJpdCIsICJGIjogIlZhcmlhdGlvbmFsRnJlZUVuZXJneSIsICJHIjogIkV4cGVjdGVkRnJlZUVuZXJneSIsICJvIjogIk9ic2VydmF0aW9uIiwgInMiOiAiSGlkZGVuU3RhdGUiLCAic19wcmltZSI6ICJOZXh0SGlkZGVuU3RhdGUiLCAidCI6ICJUaW1lIiwgInUiOiAiQWN0aW9uIiwgIlx1MDNiMiI6ICJJbnZlcnNlVGVtcGVyYXR1cmUiLCAiXHUwM2IzIjogIlBvbGljeVByZWNpc2lvbiIsICJcdTAzYzAiOiAiUG9saWN5VmVjdG9yIiwgIlx1MDNjOSI6ICJTZW5zb3J5UHJlY2lzaW9uIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszXSwgImluZGV4IjogMCwgIm5hbWUiOiAiXHUwM2MwIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFsxXSwgImluZGV4IjogMSwgIm5hbWUiOiAidSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMSwgInR5cGUiOiAiZmxvYXQifV0sICJtYXRyaWNlcyI6IHsiQSI6IFtbMC45LCAwLjA1LCAwLjA1XSwgWzAuMDUsIDAuOSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjldXSwgIkIiOiBbW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCBbWzAuMCwgMS4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMS4wXV0sIFtbMC4wLCAwLjAsIDEuMF0sIFswLjAsIDEuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjBdXV0sICJDIjogWzAuMSwgMC4xLCAxLjBdLCAiRCI6IFswLjMzMywgMC4zMzMsIDAuMzMzXSwgIkUiOiBbMC4zMzMsIDAuMzMzLCAwLjMzM119LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY2xhaW1lZF9zbGljZV9jb252ZW50aW9uIjogbnVsbCwgImNvbnRyYWRpY3Rpb24iOiBmYWxzZSwgImRlY2xhcmVkX29yZGVyIjogWyJuZXh0X3N0YXRlIiwgInByZXZpb3VzX3N0YXRlIiwgImFjdGlvbiJdLCAiZGVyaXZlZCI6IGZhbHNlLCAiZGV0ZWN0ZWRfb3JkZXIiOiBudWxsLCAicmVhc29uIjogbnVsbCwgInNoYXBlIjogWzMsIDMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJFIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAibyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV0sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInMiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJzX3ByaW1lIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dfSwgInZhcmlhYmxlcyI6IFt7ImNvbW1lbnQiOiAiSGlkZGVuIHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAibmFtZSI6ICJzIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIk5leHQgaGlkZGVuIHN0YXRlIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJuYW1lIjogInNfcHJpbWUiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAibmFtZSI6ICJvIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlBvbGljeSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszXSwgIm5hbWUiOiAiXHUwM2MwIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlNlbGVjdGVkIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ1IiwgInR5cGUiOiAiZmxvYXQifV19"
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
