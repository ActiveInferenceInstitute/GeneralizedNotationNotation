#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation
# Generated from GNN Model: Active Inference POMDP Agent
# Generated: 2026-05-22 06:18:15

using Pkg
using RxInfer
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON
using Base64
using Dates

const SCHEMA_VERSION = "rxinfer_simulation_v1"
const MODEL_NAME = "Active Inference POMDP Agent"
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 3
const TIME_STEPS = 30
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAic19wcmltZSJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkEiLCAidGFyZ2V0IjogIm8ifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJCIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQyIsICJ0YXJnZXQiOiAiRyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkUiLCAidGFyZ2V0IjogIlx1MDNjMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkciLCAidGFyZ2V0IjogIlx1MDNjMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIlx1MDNjMCIsICJ0YXJnZXQiOiAidSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkIiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJ1IiwgInRhcmdldCI6ICJzX3ByaW1lIn1dLCAiZGVzY3JpcHRpb24iOiAiVGhpcyBtb2RlbCBkZXNjcmliZXMgYSBjbGFzc2ljIEFjdGl2ZSBJbmZlcmVuY2UgYWdlbnQgZm9yIGEgZGlzY3JldGUgUE9NRFA6XG4tIE9uZSBvYnNlcnZhdGlvbiBtb2RhbGl0eSAoXCJzdGF0ZV9vYnNlcnZhdGlvblwiKSB3aXRoIDMgcG9zc2libGUgb3V0Y29tZXMuXG4tIE9uZSBoaWRkZW4gc3RhdGUgZmFjdG9yIChcImxvY2F0aW9uXCIpIHdpdGggMyBwb3NzaWJsZSBzdGF0ZXMuXG4tIFRoZSBoaWRkZW4gc3RhdGUgaXMgZnVsbHkgY29udHJvbGxhYmxlIHZpYSAzIGRpc2NyZXRlIGFjdGlvbnMuXG4tIFRoZSBhZ2VudCdzIHByZWZlcmVuY2VzIGFyZSBlbmNvZGVkIGFzIGxvZy1wcm9iYWJpbGl0aWVzIG92ZXIgb2JzZXJ2YXRpb25zLlxuLSBUaGUgYWdlbnQgaGFzIGFuIGluaXRpYWwgcG9saWN5IHByaW9yIChoYWJpdCkgZW5jb2RlZCBhcyBsb2ctcHJvYmFiaWxpdGllcyBvdmVyIGFjdGlvbnMuIiwgImluaXRpYWxfcGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC45LCAwLjA1LCAwLjA1XSwgWzAuMDUsIDAuOSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjldXSwgIkIiOiBbW1sxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCBbWzAuMCwgMS4wLCAwLjBdLCBbMS4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzEuMCwgMS4wLCAwLjBdXV0sICJDIjogWzAuMSwgMC4xLCAxLjBdLCAiRCI6IFswLjMzMzMzMzMzMzMzMzMzMzMsIDAuMzMzMzMzMzMzMzMzMzMzMywgMC4zMzMzMzMzMzMzMzMzMzMzXSwgIkUiOiBbMC4zMzMzMzMzMzMzMzMzMzMzLCAwLjMzMzMzMzMzMzMzMzMzMzMsIDAuMzMzMzMzMzMzMzMzMzMzM119LCAiaW5pdGlhbHBhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuOSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjksIDAuMDVdLCBbMC4wNSwgMC4wNSwgMC45XV0sICJCIjogW1tbMS4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjBdXSwgW1swLjAsIDEuMCwgMC4wXSwgWzEuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDEuMCwgMC4wXV1dLCAiQyI6IFswLjEsIDAuMSwgMS4wXSwgIkQiOiBbMC4zMzMzMzMzMzMzMzMzMzMzLCAwLjMzMzMzMzMzMzMzMzMzMzMsIDAuMzMzMzMzMzMzMzMzMzMzM10sICJFIjogWzAuMzMzMzMzMzMzMzMzMzMzMywgMC4zMzMzMzMzMzMzMzMzMzMzLCAwLjMzMzMzMzMzMzMzMzMzMzNdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCAzXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJFIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiQWN0aXZlIEluZmVyZW5jZSBQT01EUCBBZ2VudCIsICJtb2RlbF9wYXJhbWV0ZXJzIjogeyJiX3RlbnNvcl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSAoZGlzdHJpYnV0aW9uIG92ZXIgYWN0aW9ucyksIG5vIHBsYW5uaW5nIiwgImRpbWVuc2lvbnMiOiBbM10sICJpbmRleCI6IDEsICJuYW1lIjogIlx1MDNjMCIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkFjdGlvbiB0YWtlbiIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAyLCAibmFtZSI6ICJ1IiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm51bV9hY3Rpb25zIjogMywgIm51bV9oaWRkZW5fc3RhdGVzIjogMywgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiAzLCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDMwLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiAoaW50ZWdlciBpbmRleCkiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMiwgIm5hbWUiOiAibyIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV0sICJwYXNzaXZlX21vZGVsIjogZmFsc2UsICJzaW11bGF0aW9uX3BhcmFtcyI6IHt9LCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInMiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogNCwgIm5hbWUiOiAic19wcmltZSIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV19LCAibmFtZSI6ICJBY3RpdmUgSW5mZXJlbmNlIFBPTURQIEFnZW50IiwgIm9udG9sb2d5X21hcHBpbmciOiB7IkEiOiAiTGlrZWxpaG9vZE1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiQyI6ICJMb2dQcmVmZXJlbmNlVmVjdG9yIiwgIkQiOiAiUHJpb3JPdmVySGlkZGVuU3RhdGVzIiwgIkUiOiAiSGFiaXQiLCAiRiI6ICJWYXJpYXRpb25hbEZyZWVFbmVyZ3kiLCAiRyI6ICJFeHBlY3RlZEZyZWVFbmVyZ3kiLCAibyI6ICJPYnNlcnZhdGlvbiIsICJzIjogIkhpZGRlblN0YXRlIiwgInNfcHJpbWUiOiAiTmV4dEhpZGRlblN0YXRlIiwgInQiOiAiVGltZSIsICJ1IjogIkFjdGlvbiAgICAgICAjIENob3NlbiBhY3Rpb24iLCAiXHUwM2MwIjogIlBvbGljeVZlY3RvciAjIERpc3RyaWJ1dGlvbiBvdmVyIGFjdGlvbnMifSwgInN0cnVjdHVyZWRfcG9tZHAiOiB7ImFkYXB0ZXJfbm90ZXMiOiBbXSwgImNhbm9uaWNhbF9iX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiUG9saWN5IChkaXN0cmlidXRpb24gb3ZlciBhY3Rpb25zKSwgbm8gcGxhbm5pbmciLCAiZGltZW5zaW9ucyI6IFszXSwgImluZGV4IjogMSwgIm5hbWUiOiAiXHUwM2MwIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQWN0aW9uIHRha2VuIiwgImRpbWVuc2lvbnMiOiBbMV0sICJpbmRleCI6IDIsICJuYW1lIjogInUiLCAic2l6ZSI6IDEsICJ0eXBlIjogImZsb2F0In1dLCAibWF0cmljZXMiOiB7IkEiOiBbWzAuOSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjksIDAuMDVdLCBbMC4wNSwgMC4wNSwgMC45XV0sICJCIjogW1tbMS4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjBdXSwgW1swLjAsIDEuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMF1dLCBbWzAuMCwgMC4wLCAxLjBdLCBbMC4wLCAxLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wXV1dLCAiQyI6IFswLjEsIDAuMSwgMS4wXSwgIkQiOiBbMC4zMzMzMywgMC4zMzMzMywgMC4zMzMzM10sICJFIjogWzAuMzMzMzMsIDAuMzMzMzMsIDAuMzMzMzNdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCAzXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDMsIDNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJFIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24gKGludGVnZXIgaW5kZXgpIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDIsICJuYW1lIjogIm8iLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dLCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInMiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogNCwgIm5hbWUiOiAic19wcmltZSIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV19LCAidmFyaWFibGVzIjogW3siY29tbWVudCI6ICJMaWtlbGlob29kIG1hcHBpbmcgaGlkZGVuIHN0YXRlcyB0byBvYnNlcnZhdGlvbnMiLCAiZGltZW5zaW9ucyI6IFszLCAzXSwgIm5hbWUiOiAiQSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTdGF0ZSB0cmFuc2l0aW9ucyBnaXZlbiBwcmV2aW91cyBzdGF0ZSBhbmQgYWN0aW9uIiwgImRpbWVuc2lvbnMiOiBbMywgMywgM10sICJuYW1lIjogIkIiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJpb3Igb3ZlciBpbml0aWFsIGhpZGRlbiBzdGF0ZXMiLCAiZGltZW5zaW9ucyI6IFszXSwgIm5hbWUiOiAiRCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJDdXJyZW50IGhpZGRlbiBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAicyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAic19wcmltZSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJEaXNjcmV0ZSB0aW1lIHN0ZXAiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMb2ctcHJlZmVyZW5jZXMgb3ZlciBvYnNlcnZhdGlvbnMiLCAiZGltZW5zaW9ucyI6IFszXSwgIm5hbWUiOiAiQyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJWYXJpYXRpb25hbCBGcmVlIEVuZXJneSBmb3IgYmVsaWVmIHVwZGF0aW5nIGZyb20gb2JzZXJ2YXRpb25zIiwgImRpbWVuc2lvbnMiOiBbIlx1MDNjMCJdLCAibmFtZSI6ICJGIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24gKGludGVnZXIgaW5kZXgpIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJuYW1lIjogIm8iLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiSW5pdGlhbCBwb2xpY3kgcHJpb3IgKGhhYml0KSBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFszXSwgIm5hbWUiOiAiRSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQb2xpY3kgKGRpc3RyaWJ1dGlvbiBvdmVyIGFjdGlvbnMpLCBubyBwbGFubmluZyIsICJkaW1lbnNpb25zIjogWzNdLCAibmFtZSI6ICJcdTAzYzAiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQWN0aW9uIHRha2VuIiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInUiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiRXhwZWN0ZWQgRnJlZSBFbmVyZ3kgKHBlciBwb2xpY3kpIiwgImRpbWVuc2lvbnMiOiBbIlx1MDNjMCJdLCAibmFtZSI6ICJHIiwgInR5cGUiOiAiZmxvYXQifV19"
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
        "framework" => "RxInfer.jl",
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
            "rxinfer_version" => package_version("RxInfer"),
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
    println("RxInfer.jl simulation wrote simulation_results.json")
    return results["validation"]["all_valid"] ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
