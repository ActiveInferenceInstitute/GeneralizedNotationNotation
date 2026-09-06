#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Hierarchical Active Inference POMDP

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
const MODEL_NAME = "Hierarchical Active Inference POMDP"
const NUM_STATES = 8
const NUM_OBSERVATIONS = 16
const NUM_ACTIONS = 3
const TIME_STEPS = 20
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
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
