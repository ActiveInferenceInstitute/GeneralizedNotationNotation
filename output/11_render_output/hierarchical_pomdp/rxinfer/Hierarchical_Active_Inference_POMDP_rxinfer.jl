#!/usr/bin/env julia
# RxInfer.jl two-level hierarchical POMDP simulation — genuine @model + infer()
# Generated from GNN Model: Hierarchical Active Inference POMDP
# Generated: 2026-09-05 20:32:38
#
# Structure (matches the GNN file's declared semantics):
#   z (slow context, 2 states) --A_level2--> fast-state prior
#   s[t] (fast, 4 states) driven by actions over B_level1
#   y[t] (4 obs) emitted through A_level1
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
const MODEL_NAME = "Hierarchical Active Inference POMDP"
const NUM_FAST = 4
const NUM_SLOW = 2
const NUM_OBSERVATIONS = 4
const NUM_ACTIONS = 3
const TIME_STEPS = 20
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const INFERENCE_ITERATIONS = 20
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "hierarchical"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkRfbGV2ZWwxIiwgInRhcmdldCI6ICJzX2xldmVsMSJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbGV2ZWwxIiwgInRhcmdldCI6ICJBX2xldmVsMSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInNfbGV2ZWwxIiwgInRhcmdldCI6ICJ4X25leHQxIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQV9sZXZlbDEiLCAidGFyZ2V0IjogIm9fbGV2ZWwxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQ19sZXZlbDEiLCAidGFyZ2V0IjogIkcxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRzEiLCAidGFyZ2V0IjogIlx1MDNjMDEifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJcdTAzYzAxIiwgInRhcmdldCI6ICJ1X2xldmVsMSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkJfbGV2ZWwxIiwgInRhcmdldCI6ICJ1X2xldmVsMSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInVfbGV2ZWwxIiwgInRhcmdldCI6ICJ4X25leHQxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAic19sZXZlbDEiLCAidGFyZ2V0IjogIm9fbGV2ZWwyIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRF9sZXZlbDIiLCAidGFyZ2V0IjogInNfbGV2ZWwyIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAic19sZXZlbDIiLCAidGFyZ2V0IjogIkFfbGV2ZWwyIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQV9sZXZlbDIiLCAidGFyZ2V0IjogIkRfbGV2ZWwxIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAic19sZXZlbDIiLCAidGFyZ2V0IjogIkJfbGV2ZWwyIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQ19sZXZlbDIiLCAidGFyZ2V0IjogIkcyIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRzIiLCAidGFyZ2V0IjogInNfbGV2ZWwyIn1dLCAiZGVzY3JpcHRpb24iOiAiQSB0d28tbGV2ZWwgaGllcmFyY2hpY2FsIFBPTURQIHdoZXJlOlxuLSBMZXZlbCAxIChmYXN0KTogNCBvYnNlcnZhdGlvbnMsIDQgaGlkZGVuIHN0YXRlcywgMyBhY3Rpb25zXG4tIExldmVsIDIgKHNsb3cpOiAyIGNvbnRleHR1YWwgc3RhdGVzIHRoYXQgbW9kdWxhdGUgTGV2ZWwgMSBsaWtlbGlob29kXG4tIEhpZ2hlci1sZXZlbCBiZWxpZWZzIGFyZSB1cGRhdGVkIGF0IGEgc2xvd2VyIHRpbWVzY2FsZVxuLSBUb3AtZG93biBwcmVkaWN0aW9ucyBjb25zdHJhaW4gYm90dG9tLXVwIGluZmVyZW5jZSBhdCBMZXZlbCAxIiwgImdubl9zZWN0aW9uIjogIkFjdEluZlBPTURQX0hpZXJhcmNoaWNhbCIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuMzgyNSwgMC4wNDI1LCAwLjAyMjUwMDAwMDAwMDAwMDAwMywgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwNiwgMC4wMDI1MDAwMDAwMDAwMDAwMDEsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjAwMjUwMDAwMDAwMDAwMDAwMDVdLCBbMC4wNDI1LCAwLjM4MjUsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDMsIDAuMDAyNTAwMDAwMDAwMDAwMDAxLCAwLjAyMjUwMDAwMDAwMDAwMDAwNiwgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwM10sIFswLjIxMjUsIDAuMjEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNV0sIFswLjIxMjUsIDAuMjEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNV0sIFswLjAyMjUwMDAwMDAwMDAwMDAwNiwgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjM4MjUsIDAuMDQyNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAxLCAwLjAyMjUwMDAwMDAwMDAwMDAwMywgMC4wMDI1MDAwMDAwMDAwMDAwMDA1XSwgWzAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDQyNSwgMC4zODI1LCAwLjAwMjUwMDAwMDAwMDAwMDAwMSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDNdLCBbMC4wMTI1LCAwLjAxMjUsIDAuMjEyNTAwMDAwMDAwMDAwMDIsIDAuMjEyNTAwMDAwMDAwMDAwMDIsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNV0sIFswLjAxMjUsIDAuMDEyNSwgMC4yMTI1MDAwMDAwMDAwMDAwMiwgMC4yMTI1MDAwMDAwMDAwMDAwMiwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1XSwgWzAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMzgyNSwgMC4wNDI1MDAwMDAwMDAwMDAwMSwgMC4wMjI1MDAwMDAwMDAwMDAwMDMsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNV0sIFswLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjA0MjUwMDAwMDAwMDAwMDAxLCAwLjM4MjUsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDNdLCBbMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjIxMjUsIDAuMjEyNSwgMC4wMTI1LCAwLjAxMjVdLCBbMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjIxMjUsIDAuMjEyNSwgMC4wMTI1LCAwLjAxMjVdLCBbMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDMsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAxLCAwLjM4MjUsIDAuMDQyNV0sIFswLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjAwMjUwMDAwMDAwMDAwMDAwMSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDQyNSwgMC4zODI1XSwgWzAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMjEyNSwgMC4yMTI1XSwgWzAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMjEyNSwgMC4yMTI1XV0sICJCIjogW1tbMC45LCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4wXSwgWzAuMCwgMC45LCAwLjBdLCBbMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuMCwgMC45XSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4xLCAwLjAsIDAuMF0sIFswLjksIDAuMCwgMC4wXSwgWzAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjldLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMSwgMC4wXSwgWzAuOSwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuOV0sIFswLjAsIDAuMCwgMC4xXV0sIFtbMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuOSwgMC4wXSwgWzAuMSwgMC4wLCAwLjBdLCBbMC45LCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC45XV0sIFtbMC4wLCAwLjAsIDAuOV0sIFswLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjksIDAuMCwgMC4wXSwgWzAuMSwgMC4wLCAwLjBdLCBbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMSwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC45XSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4wXSwgWzAuOSwgMC4wLCAwLjBdLCBbMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuOSwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjldLCBbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuOSwgMC4wXSwgWzAuMCwgMC4xLCAwLjBdLCBbMC45LCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuOV0sIFswLjAsIDAuMSwgMC4wXSwgWzAuMCwgMC45LCAwLjBdLCBbMC4xLCAwLjAsIDAuMF0sIFswLjksIDAuMCwgMC4wXV1dLCAiQyI6IFswLjEsIDAuNiwgMC4xLCAwLjYsIDAuMSwgMC42LCAwLjEsIDAuNiwgMC4xLCAwLjYsIDAuMSwgMC42LCAxLjAsIDEuNSwgMS4wLCAxLjVdLCAiRCI6IFswLjEyNSwgMC4xMjUsIDAuMTI1LCAwLjEyNSwgMC4xMjUsIDAuMTI1LCAwLjEyNSwgMC4xMjVdfSwgImluaXRpYWxwYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1swLjM4MjUsIDAuMDQyNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDMsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAxLCAwLjAyMjUwMDAwMDAwMDAwMDAwMywgMC4wMDI1MDAwMDAwMDAwMDAwMDA1XSwgWzAuMDQyNSwgMC4zODI1LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjAwMjUwMDAwMDAwMDAwMDAwMSwgMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4wMjI1MDAwMDAwMDAwMDAwMDNdLCBbMC4yMTI1LCAwLjIxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjVdLCBbMC4yMTI1LCAwLjIxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjVdLCBbMC4wMjI1MDAwMDAwMDAwMDAwMDYsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNSwgMC4zODI1LCAwLjA0MjUsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMSwgMC4wMjI1MDAwMDAwMDAwMDAwMDMsIDAuMDAyNTAwMDAwMDAwMDAwMDAwNV0sIFswLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjA0MjUsIDAuMzgyNSwgMC4wMDI1MDAwMDAwMDAwMDAwMDEsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzXSwgWzAuMDEyNSwgMC4wMTI1LCAwLjIxMjUwMDAwMDAwMDAwMDAyLCAwLjIxMjUwMDAwMDAwMDAwMDAyLCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjVdLCBbMC4wMTI1LCAwLjAxMjUsIDAuMjEyNTAwMDAwMDAwMDAwMDIsIDAuMjEyNTAwMDAwMDAwMDAwMDIsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNV0sIFswLjAyMjUwMDAwMDAwMDAwMDAwNiwgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwMywgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjM4MjUsIDAuMDQyNTAwMDAwMDAwMDAwMDEsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjAwMjUwMDAwMDAwMDAwMDAwMDVdLCBbMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwNiwgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwMywgMC4wNDI1MDAwMDAwMDAwMDAwMSwgMC4zODI1LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzXSwgWzAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4yMTI1LCAwLjIxMjUsIDAuMDEyNSwgMC4wMTI1XSwgWzAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4yMTI1LCAwLjIxMjUsIDAuMDEyNSwgMC4wMTI1XSwgWzAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDAzLCAwLjAwMjUwMDAwMDAwMDAwMDAwMDUsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjAwMjUwMDAwMDAwMDAwMDAwMSwgMC4zODI1LCAwLjA0MjVdLCBbMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwNiwgMC4wMDI1MDAwMDAwMDAwMDAwMDA1LCAwLjAyMjUwMDAwMDAwMDAwMDAwMywgMC4wMDI1MDAwMDAwMDAwMDAwMDEsIDAuMDIyNTAwMDAwMDAwMDAwMDA2LCAwLjA0MjUsIDAuMzgyNV0sIFswLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjIxMjUsIDAuMjEyNV0sIFswLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjAxMjUsIDAuMDEyNSwgMC4wMTI1LCAwLjIxMjUsIDAuMjEyNV1dLCAiQiI6IFtbWzAuOSwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMF0sIFswLjAsIDAuOSwgMC4wXSwgWzAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjAsIDAuOV0sIFswLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF1dLCBbWzAuMSwgMC4wLCAwLjBdLCBbMC45LCAwLjAsIDAuMF0sIFswLjAsIDAuMSwgMC4wXSwgWzAuMCwgMC45LCAwLjBdLCBbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC45XSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC45LCAwLjBdLCBbMC4wLCAwLjEsIDAuMF0sIFswLjksIDAuMCwgMC4wXSwgWzAuMSwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjldLCBbMC4wLCAwLjAsIDAuMV1dLCBbWzAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjksIDAuMF0sIFswLjEsIDAuMCwgMC4wXSwgWzAuOSwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuOV1dLCBbWzAuMCwgMC4wLCAwLjldLCBbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC45LCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4wXSwgWzAuMCwgMC45LCAwLjBdLCBbMC4wLCAwLjEsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuOV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMF0sIFswLjksIDAuMCwgMC4wXSwgWzAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjksIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC45XSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMSwgMC4wXSwgWzAuOSwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjldLCBbMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuOSwgMC4wXSwgWzAuMSwgMC4wLCAwLjBdLCBbMC45LCAwLjAsIDAuMF1dXSwgIkMiOiBbMC4xLCAwLjYsIDAuMSwgMC42LCAwLjEsIDAuNiwgMC4xLCAwLjYsIDAuMSwgMC42LCAwLjEsIDAuNiwgMS4wLCAxLjUsIDEuMCwgMS41XSwgIkQiOiBbMC4xMjUsIDAuMTI1LCAwLjEyNSwgMC4xMjUsIDAuMTI1LCAwLjEyNSwgMC4xMjUsIDAuMTI1XX0sICJtYXRyaXhfcHJvdmVuYW5jZSI6IHsiQSI6IHsiZGVyaXZlZCI6IHRydWUsICJzaGFwZSI6IFsxNiwgOF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkFfbGV2ZWwxIiwgIkFfbGV2ZWwyIl19LCAiQV9sZXZlbDEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJBX2xldmVsMiI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNCwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJkZXJpdmVkIjogdHJ1ZSwgImZhY3Rvcl9hY3Rpb25fY291bnRzIjogWzMsIDFdLCAia3JvbmVja2VyX2ZhY3Rvcml6ZWQiOiBmYWxzZSwgInNoYXBlIjogWzgsIDgsIDNdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJCX2xldmVsMSIsICJCX2xldmVsMiJdLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJCX2xldmVsMSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMywgNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkJfbGV2ZWwyIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyLCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IHRydWUsICJzaGFwZSI6IFsxNl0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkNfbGV2ZWwxIiwgIkNfbGV2ZWwyIl19LCAiQ19sZXZlbDEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDX2xldmVsMiI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbOF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkRfbGV2ZWwxIiwgIkRfbGV2ZWwyIl19LCAiRF9sZXZlbDEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEX2xldmVsMiI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJtb2RlbF9uYW1lIjogIkhpZXJhcmNoaWNhbCBBY3RpdmUgSW5mZXJlbmNlIFBPTURQIiwgIm1vZGVsX3BhcmFtZXRlcnMiOiB7ImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFtdLCAibnVtX2FjdGlvbnMiOiAzLCAibnVtX2FjdGlvbnNfbDEiOiAzLCAibnVtX2NvbnRleHRfc3RhdGVzX2wyIjogMiwgIm51bV9oaWRkZW5fc3RhdGVzIjogOCwgIm51bV9oaWRkZW5fc3RhdGVzX2wxIjogNCwgIm51bV9tb2RhbGl0aWVzIjogMiwgIm51bV9vYnMiOiAxNiwgIm51bV9vYnNfbDEiOiA0LCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDIwLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiTGV2ZWwgMSBvYnNlcnZhdGlvbnMiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAib19sZXZlbDEiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMZXZlbCAyIG9ic2VydmF0aW9uICg9IExldmVsIDEgaGlkZGVuIHN0YXRlIGRpc3RyaWJ1dGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAib19sZXZlbDIiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dLCAicGFzc2l2ZV9tb2RlbCI6IGZhbHNlLCAic2ltdWxhdGlvbl9wYXJhbXMiOiB7fSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkxldmVsIDEgaGlkZGVuIHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJzX2xldmVsMSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkxldmVsIDIgY29udGV4dHVhbCBoaWRkZW4gc3RhdGUiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMywgIm5hbWUiOiAic19sZXZlbDIiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDIsICJ0eXBlIjogImZsb2F0In1dLCAidGltZXNjYWxlX3JhdGlvIjogNX0sICJuYW1lIjogIkhpZXJhcmNoaWNhbCBBY3RpdmUgSW5mZXJlbmNlIFBPTURQIiwgIm9udG9sb2d5X21hcHBpbmciOiB7IkFfbGV2ZWwxIjogIkxpa2VsaWhvb2RNYXRyaXgiLCAiQV9sZXZlbDIiOiAiSGlnaGVyTGV2ZWxMaWtlbGlob29kTWF0cml4IiwgIkJfbGV2ZWwxIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiQl9sZXZlbDIiOiAiQ29udGV4dFRyYW5zaXRpb25NYXRyaXgiLCAiQ19sZXZlbDEiOiAiTG9nUHJlZmVyZW5jZVZlY3RvciIsICJEX2xldmVsMSI6ICJQcmlvck92ZXJIaWRkZW5TdGF0ZXMiLCAiRzEiOiAiRXhwZWN0ZWRGcmVlRW5lcmd5IiwgIkcyIjogIkhpZ2hlckxldmVsRXhwZWN0ZWRGcmVlRW5lcmd5IiwgIm9fbGV2ZWwxIjogIk9ic2VydmF0aW9uIiwgIm9fbGV2ZWwyIjogIkhpZ2hlckxldmVsT2JzZXJ2YXRpb24iLCAic19sZXZlbDEiOiAiSGlkZGVuU3RhdGUiLCAic19sZXZlbDIiOiAiQ29udGV4dHVhbEhpZGRlblN0YXRlIiwgInVfbGV2ZWwxIjogIkFjdGlvbiIsICJcdTAzYzAxIjogIlBvbGljeVZlY3RvciJ9LCAic3RydWN0dXJlZF9wb21kcCI6IHsiYWRhcHRlcl9ub3RlcyI6IFtdLCAiY2Fub25pY2FsX2Jfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY29udHJvbF9mYWN0b3JzIjogW10sICJtYXRyaWNlcyI6IHsiQV9sZXZlbDEiOiBbWzAuODUsIDAuMDUsIDAuMDUsIDAuMDVdLCBbMC4wNSwgMC44NSwgMC4wNSwgMC4wNV0sIFswLjA1LCAwLjA1LCAwLjg1LCAwLjA1XSwgWzAuMDUsIDAuMDUsIDAuMDUsIDAuODVdXSwgIkFfbGV2ZWwyIjogW1swLjksIDAuMV0sIFswLjEsIDAuOV0sIFswLjUsIDAuNV0sIFswLjUsIDAuNV1dLCAiQl9sZXZlbDEiOiBbW1sxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjBdXSwgW1swLjAsIDEuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjBdXV0sICJCX2xldmVsMiI6IFtbMC45LCAwLjFdLCBbMC4xLCAwLjldXSwgIkNfbGV2ZWwxIjogWzAuMSwgMC4xLCAwLjEsIDEuMF0sICJDX2xldmVsMiI6IFswLjAsIDAuNSwgMC4wLCAwLjVdLCAiRF9sZXZlbDEiOiBbMC4yNSwgMC4yNSwgMC4yNSwgMC4yNV0sICJEX2xldmVsMiI6IFswLjUsIDAuNV19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbMTYsIDhdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJBX2xldmVsMSIsICJBX2xldmVsMiJdfSwgIkFfbGV2ZWwxIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQV9sZXZlbDIiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IHRydWUsICJmYWN0b3JfYWN0aW9uX2NvdW50cyI6IFszLCAxXSwgImtyb25lY2tlcl9mYWN0b3JpemVkIjogZmFsc2UsICJzaGFwZSI6IFs4LCA4LCAzXSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiQl9sZXZlbDEiLCAiQl9sZXZlbDIiXSwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQl9sZXZlbDEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCX2xldmVsMiI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkMiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbMTZdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJDX2xldmVsMSIsICJDX2xldmVsMiJdfSwgIkNfbGV2ZWwxIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQ19sZXZlbDIiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzhdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJEX2xldmVsMSIsICJEX2xldmVsMiJdfSwgIkRfbGV2ZWwxIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRF9sZXZlbDIiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiTGV2ZWwgMSBvYnNlcnZhdGlvbnMiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAib19sZXZlbDEiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMZXZlbCAyIG9ic2VydmF0aW9uICg9IExldmVsIDEgaGlkZGVuIHN0YXRlIGRpc3RyaWJ1dGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAib19sZXZlbDIiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dLCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiTGV2ZWwgMSBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInNfbGV2ZWwxIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTGV2ZWwgMiBjb250ZXh0dWFsIGhpZGRlbiBzdGF0ZSIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAiaW5kZXgiOiAzLCAibmFtZSI6ICJzX2xldmVsMiIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV19LCAidmFyaWFibGVzIjogW3siY29tbWVudCI6ICJMZXZlbCAxIGhpZGRlbiBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAic19sZXZlbDEiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTGV2ZWwgMSBuZXh0IGhpZGRlbiBzdGF0ZSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJ4X25leHQxIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkxldmVsIDEgRXhwZWN0ZWQgRnJlZSBFbmVyZ3kiLCAiZGltZW5zaW9ucyI6IFsiXHUwM2MwMSJdLCAibmFtZSI6ICJHMSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMZXZlbCAyIGNvbnRleHR1YWwgaGlkZGVuIHN0YXRlIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJuYW1lIjogInNfbGV2ZWwyIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkxldmVsIDIgRXhwZWN0ZWQgRnJlZSBFbmVyZ3kiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAiRzIiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiRmFzdCB0aW1lc2NhbGUgY291bnRlciIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ0MSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTbG93IHRpbWVzY2FsZSBjb3VudGVyIiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInQyIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkxldmVsIDEgb2JzZXJ2YXRpb25zIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJuYW1lIjogIm9fbGV2ZWwxIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkxldmVsIDIgb2JzZXJ2YXRpb24gKD0gTGV2ZWwgMSBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uKSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJvX2xldmVsMiIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMZXZlbCAxIHBvbGljeSAoYWN0aW9ucykiLCAiZGltZW5zaW9ucyI6IFszXSwgIm5hbWUiOiAiXHUwM2MwMSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMZXZlbCAxIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ1X2xldmVsMSIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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
B_slow = Matrix{Float64}(I, NUM_SLOW, NUM_SLOW)
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
context_beliefs = Vector{Vector{Float64}}()
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
