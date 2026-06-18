#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: POMDP GridWorld 3x3

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
const MODEL_NAME = "POMDP GridWorld 3x3"
const NUM_STATES = 9
const NUM_OBSERVATIONS = 9
const NUM_ACTIONS = 5
const TIME_STEPS = 15
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogIkIifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJ1In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAidSIsICJ0YXJnZXQiOiAic19wcmltZSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkMiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJFIiwgInRhcmdldCI6ICJcdTAzYzAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHIiwgInRhcmdldCI6ICJcdTAzYzAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJcdTAzYzAiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJzX3ByaW1lIn1dLCAiZGVzY3JpcHRpb24iOiAiRGlzY3JldGUgM3gzIEdyaWRXb3JsZCBQT01EUCBmb3Igc3RyaWN0IGNyb3NzLWZyYW1ld29yayB2YWxpZGF0aW9uLiBUaGUgbW9kZWwgaGFzIG9uZSBoaWRkZW4gc3RhdGUgZmFjdG9yIHdpdGggOSBncmlkIGNlbGxzLCBvbmUgb2JzZXJ2YXRpb24gbW9kYWxpdHkgd2l0aCBub2lzeSBjZWxsIG9ic2VydmF0aW9ucywgYW5kIG9uZSBjb250cm9sIGZhY3RvciB3aXRoIDUgYm91bmRhcnktY2xhbXBlZCBhY3Rpb25zOiB1cCwgZG93biwgbGVmdCwgcmlnaHQsIGFuZCBzdGF5LiIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuODUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC44NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjg1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuODUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC44NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjg1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuODUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC44NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjg1XV0sICJCIjogW1tbMS4wLCAwLjAsIDEuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAxLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDEuMCwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDEuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDEuMCwgMS4wXV1dLCAiQyI6IFswLjAsIDAuMSwgMC4zLCAwLjEsIDAuNCwgMC44LCAwLjMsIDAuOCwgMy4wXSwgIkQiOiBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sICJFIjogWzAuMiwgMC4yLCAwLjIsIDAuMiwgMC4yXX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC44NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjg1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuODUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC44NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjg1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuODUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC44NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjg1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuODVdXSwgIkIiOiBbW1sxLjAsIDAuMCwgMS4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMS4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMS4wLCAwLjAsIDEuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMS4wLCAxLjBdXV0sICJDIjogWzAuMCwgMC4xLCAwLjMsIDAuMSwgMC40LCAwLjgsIDAuMywgMC44LCAzLjBdLCAiRCI6IFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgIkUiOiBbMC4yLCAwLjIsIDAuMiwgMC4yLCAwLjJdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs5LCA5XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzksIDksIDVdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbOV0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzldLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJFIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs1XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiUE9NRFAgR3JpZFdvcmxkIDN4MyIsICJtb2RlbF9wYXJhbWV0ZXJzIjogeyJhY3Rpb25fbGFiZWxzIjogInVwLGRvd24sbGVmdCxyaWdodCxzdGF5IiwgImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFs1XSwgImluZGV4IjogMSwgIm5hbWUiOiAiXHUwM2MwIiwgInNpemUiOiA1LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQWN0aW9uIGluZGV4IiwgImRpbWVuc2lvbnMiOiBbMV0sICJpbmRleCI6IDIsICJuYW1lIjogInUiLCAic2l6ZSI6IDEsICJ0eXBlIjogImZsb2F0In1dLCAiZ29hbF9zdGF0ZSI6IDgsICJncmlkX2NvbHMiOiAzLCAiZ3JpZF9yb3dzIjogMywgIm51bV9hY3Rpb25zIjogNSwgIm51bV9oaWRkZW5fc3RhdGVzIjogOSwgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiA5LCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDE1LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzksIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJvIiwgInNpemUiOiA5LCAidHlwZSI6ICJmbG9hdCJ9XSwgInBhc3NpdmVfbW9kZWwiOiBmYWxzZSwgInJhbmRvbV9zZWVkIjogNDIsICJzaW11bGF0aW9uX3BhcmFtcyI6IHt9LCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbOSwgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInMiLCAic2l6ZSI6IDksICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBkaXN0cmlidXRpb24iLCAiZGltZW5zaW9ucyI6IFs5LCAxXSwgImluZGV4IjogNCwgIm5hbWUiOiAic19wcmltZSIsICJzaXplIjogOSwgInR5cGUiOiAiZmxvYXQifV19LCAibmFtZSI6ICJQT01EUCBHcmlkV29ybGQgM3gzIiwgIm9udG9sb2d5X21hcHBpbmciOiB7IkEiOiAiTGlrZWxpaG9vZE1hdHJpeCIsICJCIjogIlRyYW5zaXRpb25NYXRyaXgiLCAiQyI6ICJMb2dQcmVmZXJlbmNlVmVjdG9yIiwgIkQiOiAiUHJpb3JPdmVySGlkZGVuU3RhdGVzIiwgIkUiOiAiSGFiaXQiLCAiRyI6ICJFeHBlY3RlZEZyZWVFbmVyZ3kiLCAibyI6ICJPYnNlcnZhdGlvbiIsICJzIjogIkhpZGRlblN0YXRlIiwgInNfcHJpbWUiOiAiTmV4dEhpZGRlblN0YXRlIiwgInQiOiAiVGltZSIsICJ1IjogIkFjdGlvbiIsICJcdTAzYzAiOiAiUG9saWN5VmVjdG9yIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBkaXN0cmlidXRpb24gb3ZlciBhY3Rpb25zIiwgImRpbWVuc2lvbnMiOiBbNV0sICJpbmRleCI6IDEsICJuYW1lIjogIlx1MDNjMCIsICJzaXplIjogNSwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkFjdGlvbiBpbmRleCIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAyLCAibmFtZSI6ICJ1IiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm1hdHJpY2VzIjogeyJBIjogW1swLjg1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuODUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC44NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjg1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuODUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC44NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NV0sIFswLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjAxODc1LCAwLjg1LCAwLjAxODc1LCAwLjAxODc1XSwgWzAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuMDE4NzUsIDAuODUsIDAuMDE4NzVdLCBbMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC4wMTg3NSwgMC44NV1dLCAiQiI6IFtbWzEuMCwgMC4wLCAxLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAxLjAsIDEuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjAsIDEuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAxLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAxLjAsIDAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAxLjAsIDEuMF1dXSwgIkMiOiBbMC4wLCAwLjEsIDAuMywgMC4xLCAwLjQsIDAuOCwgMC4zLCAwLjgsIDMuMF0sICJEIjogWzEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCAiRSI6IFswLjIsIDAuMiwgMC4yLCAwLjIsIDAuMl19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzksIDldLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbOSwgOSwgNV0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs5XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbOV0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkUiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzVdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzksIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJvIiwgInNpemUiOiA5LCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgaGlkZGVuIHN0YXRlIGRpc3RyaWJ1dGlvbiIsICJkaW1lbnNpb25zIjogWzksIDFdLCAiaW5kZXgiOiAzLCAibmFtZSI6ICJzIiwgInNpemUiOiA5LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTmV4dCBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbOSwgMV0sICJpbmRleCI6IDQsICJuYW1lIjogInNfcHJpbWUiLCAic2l6ZSI6IDksICJ0eXBlIjogImZsb2F0In1dfSwgInZhcmlhYmxlcyI6IFt7ImNvbW1lbnQiOiAiTGlrZWxpaG9vZCBtYXRyaXg6IG9ic2VydmF0aW9ucyBieSBoaWRkZW4gc3RhdGVzIiwgImRpbWVuc2lvbnMiOiBbOSwgOV0sICJuYW1lIjogIkEiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiVHJhbnNpdGlvbiB0ZW5zb3I6IG5leHRfc3RhdGUsIHByZXZpb3VzX3N0YXRlLCBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFs5LCA5LCA1XSwgIm5hbWUiOiAiQiIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmlvciBvdmVyIGluaXRpYWwgaGlkZGVuIHN0YXRlIiwgImRpbWVuc2lvbnMiOiBbOV0sICJuYW1lIjogIkQiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbOSwgMV0sICJuYW1lIjogInMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTmV4dCBoaWRkZW4gc3RhdGUgZGlzdHJpYnV0aW9uIiwgImRpbWVuc2lvbnMiOiBbOSwgMV0sICJuYW1lIjogInNfcHJpbWUiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiRGlzY3JldGUgdGltZSBzdGVwIiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInQiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTG9nLXByZWZlcmVuY2VzIG92ZXIgb2JzZXJ2YXRpb25zIiwgImRpbWVuc2lvbnMiOiBbOV0sICJuYW1lIjogIkMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzksIDFdLCAibmFtZSI6ICJvIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlBvbGljeSBwcmlvciBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFs1XSwgIm5hbWUiOiAiRSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQb2xpY3kgZGlzdHJpYnV0aW9uIG92ZXIgYWN0aW9ucyIsICJkaW1lbnNpb25zIjogWzVdLCAibmFtZSI6ICJcdTAzYzAiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQWN0aW9uIGluZGV4IiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInUiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiRXhwZWN0ZWQgRnJlZSBFbmVyZ3kgcGVyIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWyJcdTAzYzAiXSwgIm5hbWUiOiAiRyIsICJ0eXBlIjogImZsb2F0In1dfQ=="
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
