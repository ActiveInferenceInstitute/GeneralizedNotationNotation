#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation — genuine @model + infer() pipeline
# Generated from GNN Model: T-Maze Epistemic Foraging Agent
# Generated: 2026-09-08 06:58:13
#
# This script uses real RxInfer.jl variational message-passing inference:
#   - @model defines the generative POMDP with Categorical / DiscreteTransition nodes
#   - infer() with free_energy=true returns posteriors over hidden states
#     and real variational free energy traces
#   - EFE and policy selection remain custom (not RxInfer's domain)

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

# --- Optional Julia-native plotting via Plots.jl (matplotlib-free PNGs).
# Guarded so a missing Plots installation/backend degrades gracefully and the
# script NEVER fails to run because of plotting.
const PLOTS_READY = try
@eval using Plots
true
catch e
println("⚠️ Plots unavailable; PNG plotting disabled: $e")
false
end

const SCHEMA_VERSION = "rxinfer_simulation_v1"
const MODEL_NAME = "T-Maze Epistemic Foraging Agent"
const NUM_STATES = 8
const NUM_OBSERVATIONS = 12
const NUM_ACTIONS = 4
const TIME_STEPS = 3
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const INFERENCE_ITERATIONS = 20
const B_TENSOR_ORDER = "next_state_previous_state_action"
const MODEL_KIND = "flat"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkRfbG9jIiwgInRhcmdldCI6ICJzX2xvYyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkRfY3R4IiwgInRhcmdldCI6ICJzX2N0eCJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbG9jIiwgInRhcmdldCI6ICJBX2xvYyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkFfbG9jIiwgInRhcmdldCI6ICJvX2xvYyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbG9jIiwgInRhcmdldCI6ICJBX3JldyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfY3R4IiwgInRhcmdldCI6ICJBX3JldyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkFfcmV3IiwgInRhcmdldCI6ICJvX3JldyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbG9jIiwgInRhcmdldCI6ICJCX2xvYyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfY3R4IiwgInRhcmdldCI6ICJCX2N0eCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkNfcmV3IiwgInRhcmdldCI6ICJHX2lucyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkdfZXBpIiwgInRhcmdldCI6ICJHIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiR19pbnMiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHIiwgInRhcmdldCI6ICJwaSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInBpIiwgInRhcmdldCI6ICJ1In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQl9sb2MiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzX2xvYyIsICJ0YXJnZXQiOiAiRiJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfY3R4IiwgInRhcmdldCI6ICJGIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAib19sb2MiLCAidGFyZ2V0IjogIkYifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJvX3JldyIsICJ0YXJnZXQiOiAiRiJ9XSwgImRlc2NyaXB0aW9uIjogIlRoZSBjbGFzc2ljIFQtbWF6ZSB0YXNrIGZyb20gQWN0aXZlIEluZmVyZW5jZSBsaXRlcmF0dXJlIChGcmlzdG9uIGV0IGFsLik6XG4tIEFnZW50IG5hdmlnYXRlcyBhIFQtc2hhcGVkIG1hemUgd2l0aCA0IGxvY2F0aW9uczogY2VudGVyLCBsZWZ0IGFybSwgcmlnaHQgYXJtLCBjdWUgbG9jYXRpb25cbi0gVHdvIG9ic2VydmF0aW9uIG1vZGFsaXRpZXM6IGxvY2F0aW9uICh3aGVyZSBhbSBJPykgYW5kIHJld2FyZC9jdWUgKHdoYXQgZG8gSSBzZWU/KVxuLSBSZXdhcmQgaXMgaGlkZGVuIGJlaGluZCBvbmUgb2YgdGhlIHR3byBhcm1zIChsZWZ0IG9yIHJpZ2h0KSwgZGV0ZXJtaW5lZCBieSBjb250ZXh0XG4tIEN1ZSBsb2NhdGlvbiBwcm92aWRlcyBwYXJ0aWFsIGluZm9ybWF0aW9uIGFib3V0IHdoaWNoIGFybSBob2xkcyB0aGUgcmV3YXJkXG4tIEFnZW50IG11c3QgZGVjaWRlOiBnbyBkaXJlY3RseSB0byBhbiBhcm0gKGV4cGxvaXQpIG9yIHZpc2l0IGN1ZSBsb2NhdGlvbiBmaXJzdCAoZXhwbG9yZSlcbi0gRGVtb25zdHJhdGVzIGVwaXN0ZW1pYyBmb3JhZ2luZzogQWN0aXZlIEluZmVyZW5jZSBuYXR1cmFsbHkgYmFsYW5jZXMgZXhwbG9yYXRpb24gdnMgZXhwbG9pdGF0aW9uXG4tIFRoZSBFeHBlY3RlZCBGcmVlIEVuZXJneSBkZWNvbXBvc2VzIGludG8gZXBpc3RlbWljIChpbmZvcm1hdGlvbiBnYWluKSArIGluc3RydW1lbnRhbCAocmV3YXJkKSB2YWx1ZSIsICJnbm5fc2VjdGlvbiI6ICJBY3RJbmZQT01EUCIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzEuMCwgMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXV0sICJCIjogW1tbMC4yNSwgMC4yNSwgMC4yNSwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC41LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDEuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC41LCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuNSwgMC4wXV0sIFtbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC41LCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC41XV0sIFtbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC41XV0sIFtbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDEuMCwgMC41LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAxLjAsIDAuNSwgMC4wXV1dLCAiQyI6IFstMS4wLCAzLjAsIDAuMCwgLTEuMCwgMy4wLCAwLjAsIC0xLjAsIDMuMCwgMC4wLCAtMS4wLCAzLjAsIDAuMF0sICJEIjogWzAuNSwgMC41LCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMS4wLCAxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdXSwgIkIiOiBbW1swLjI1LCAwLjI1LCAwLjI1LCAxLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuNSwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC41LCAwLjBdXSwgW1swLjI1LCAwLjI1LCAwLjI1LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuNSwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdXSwgW1swLjI1LCAwLjI1LCAwLjI1LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdXSwgW1swLjI1LCAwLjI1LCAwLjI1LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMS4wLCAwLjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDEuMCwgMC41LCAwLjBdXV0sICJDIjogWy0xLjAsIDMuMCwgMC4wLCAtMS4wLCAzLjAsIDAuMCwgLTEuMCwgMy4wLCAwLjAsIC0xLjAsIDMuMCwgMC4wXSwgIkQiOiBbMC41LCAwLjUsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzEyLCA4XSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiQV9sb2MiLCAiQV9yZXciXX0sICJBX2xvYyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkFfcmV3IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCA0LCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiB0cnVlLCAiZmFjdG9yX2FjdGlvbl9jb3VudHMiOiBbMSwgNF0sICJrcm9uZWNrZXJfZmFjdG9yaXplZCI6IGZhbHNlLCAic2hhcGUiOiBbOCwgOCwgNF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkJfY3R4IiwgIkJfbG9jIl0sICJzb3VyY2Vfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24ifSwgIkJfY3R4IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyLCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQl9sb2MiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzEyXSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiQ19sb2MiLCAiQ19yZXciXX0sICJDX2xvYyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkNfcmV3IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IHRydWUsICJzaGFwZSI6IFs4XSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiRF9jdHgiLCAiRF9sb2MiXX0sICJEX2N0eCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkRfbG9jIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiVC1NYXplIEVwaXN0ZW1pYyBGb3JhZ2luZyBBZ2VudCIsICJtb2RlbF9wYXJhbWV0ZXJzIjogeyJiX3RlbnNvcl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBvdmVyIDQgYWN0aW9uczogKGdvX2xlZnQsIGdvX3JpZ2h0LCBnb19jdWUsIHN0YXkpIiwgImRpbWVuc2lvbnMiOiBbNF0sICJpbmRleCI6IDAsICJuYW1lIjogInBpIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFsxXSwgImluZGV4IjogMSwgIm5hbWUiOiAidSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMSwgInR5cGUiOiAiZmxvYXQifV0sICJudW1fYWN0aW9ucyI6IDQsICJudW1fY29udGV4dHMiOiAyLCAibnVtX2hpZGRlbl9zdGF0ZXMiOiA4LCAibnVtX2xvY2F0aW9uX29icyI6IDQsICJudW1fbG9jYXRpb25zIjogNCwgIm51bV9tb2RhbGl0aWVzIjogMiwgIm51bV9vYnMiOiAxMiwgIm51bV9yZXdhcmRfb2JzIjogMywgIm51bV9zdGF0ZV9mYWN0b3JzIjogMiwgIm51bV90aW1lc3RlcHMiOiAzLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiTG9jYXRpb24gb2JzZXJ2YXRpb246ICgwOmNlbnRlciwgMTpsZWZ0LCAyOnJpZ2h0LCAzOmN1ZSkiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAib19sb2MiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJSZXdhcmQvY3VlIG9ic2VydmF0aW9uOiAoMDpub19yZXdhcmQsIDE6cmV3YXJkLCAyOmN1ZV9sZWZ0KSIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJvX3JldyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV0sICJwYXNzaXZlX21vZGVsIjogZmFsc2UsICJzaW11bGF0aW9uX3BhcmFtcyI6IHt9LCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiTG9jYXRpb24gc3RhdGU6ICgwOmNlbnRlciwgMTpsZWZ0X2FybSwgMjpyaWdodF9hcm0sIDM6Y3VlX2xvY2F0aW9uKSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJzX2xvYyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkNvbnRleHQgc3RhdGU6ICgwOnJld2FyZF9sZWZ0LCAxOnJld2FyZF9yaWdodCkiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic19jdHgiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDIsICJ0eXBlIjogImZsb2F0In1dfSwgIm5hbWUiOiAiVC1NYXplIEVwaXN0ZW1pYyBGb3JhZ2luZyBBZ2VudCIsICJvbnRvbG9neV9tYXBwaW5nIjogeyJBX2xvYyI6ICJMb2NhdGlvbkxpa2VsaWhvb2RNYXRyaXgiLCAiQV9yZXciOiAiUmV3YXJkTGlrZWxpaG9vZE1hdHJpeCIsICJCX2N0eCI6ICJDb250ZXh0VHJhbnNpdGlvbk1hdHJpeCIsICJCX2xvYyI6ICJMb2NhdGlvblRyYW5zaXRpb25NYXRyaXgiLCAiQ19sb2MiOiAiTG9jYXRpb25QcmVmZXJlbmNlVmVjdG9yIiwgIkNfcmV3IjogIlJld2FyZFByZWZlcmVuY2VWZWN0b3IiLCAiRF9jdHgiOiAiQ29udGV4dFByaW9yIiwgIkRfbG9jIjogIkxvY2F0aW9uUHJpb3IiLCAiRiI6ICJWYXJpYXRpb25hbEZyZWVFbmVyZ3kiLCAiRyI6ICJFeHBlY3RlZEZyZWVFbmVyZ3kiLCAiR19lcGkiOiAiRXBpc3RlbWljVmFsdWUiLCAiR19pbnMiOiAiSW5zdHJ1bWVudGFsVmFsdWUiLCAib19sb2MiOiAiTG9jYXRpb25PYnNlcnZhdGlvbiIsICJvX3JldyI6ICJSZXdhcmRPYnNlcnZhdGlvbiIsICJwaSI6ICJQb2xpY3lWZWN0b3IiLCAic19jdHgiOiAiQ29udGV4dEhpZGRlblN0YXRlIiwgInNfbG9jIjogIkxvY2F0aW9uSGlkZGVuU3RhdGUiLCAidCI6ICJUaW1lIiwgInUiOiAiQWN0aW9uIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBvdmVyIDQgYWN0aW9uczogKGdvX2xlZnQsIGdvX3JpZ2h0LCBnb19jdWUsIHN0YXkpIiwgImRpbWVuc2lvbnMiOiBbNF0sICJpbmRleCI6IDAsICJuYW1lIjogInBpIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFsxXSwgImluZGV4IjogMSwgIm5hbWUiOiAidSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMSwgInR5cGUiOiAiZmxvYXQifV0sICJtYXRyaWNlcyI6IHsiQV9sb2MiOiBbWzEuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMF1dLCAiQV9yZXciOiBbW1sxLjAsIDEuMF0sIFswLjAsIDEuMF0sIFsxLjAsIDAuMF0sIFswLjAsIDEuMF1dLCBbWzAuMCwgMC4wXSwgWzEuMCwgMC4wXSwgWzAuMCwgMS4wXSwgWzAuMCwgMC4wXV0sIFtbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMC4wLCAwLjBdLCBbMS4wLCAwLjBdXV0sICJCX2N0eCI6IFtbMS4wLCAwLjBdLCBbMC4wLCAxLjBdXSwgIkJfbG9jIjogW1tbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjAsIDAuMCwgMS4wXV0sIFtbMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAxLjAsIDEuMCwgMC4wXV1dLCAiQ19sb2MiOiBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgIkNfcmV3IjogWy0xLjAsIDMuMCwgMC4wXSwgIkRfY3R4IjogWzAuNSwgMC41XSwgIkRfbG9jIjogWzEuMCwgMC4wLCAwLjAsIDAuMF19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbMTIsIDhdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJBX2xvYyIsICJBX3JldyJdfSwgIkFfbG9jIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQV9yZXciOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzMsIDQsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IHRydWUsICJmYWN0b3JfYWN0aW9uX2NvdW50cyI6IFsxLCA0XSwgImtyb25lY2tlcl9mYWN0b3JpemVkIjogZmFsc2UsICJzaGFwZSI6IFs4LCA4LCA0XSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiQl9jdHgiLCAiQl9sb2MiXSwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQl9jdHgiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzIsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCX2xvYyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNCwgNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkMiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbMTJdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJDX2xvYyIsICJDX3JldyJdfSwgIkNfbG9jIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQ19yZXciOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzNdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzhdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJEX2N0eCIsICJEX2xvYyJdfSwgIkRfY3R4IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRF9sb2MiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn19LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiTG9jYXRpb24gb2JzZXJ2YXRpb246ICgwOmNlbnRlciwgMTpsZWZ0LCAyOnJpZ2h0LCAzOmN1ZSkiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAib19sb2MiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJSZXdhcmQvY3VlIG9ic2VydmF0aW9uOiAoMDpub19yZXdhcmQsIDE6cmV3YXJkLCAyOmN1ZV9sZWZ0KSIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJvX3JldyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV0sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJMb2NhdGlvbiBzdGF0ZTogKDA6Y2VudGVyLCAxOmxlZnRfYXJtLCAyOnJpZ2h0X2FybSwgMzpjdWVfbG9jYXRpb24pIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInNfbG9jIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ29udGV4dCBzdGF0ZTogKDA6cmV3YXJkX2xlZnQsIDE6cmV3YXJkX3JpZ2h0KSIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJzX2N0eCIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV19LCAidmFyaWFibGVzIjogW3siY29tbWVudCI6ICJMb2NhdGlvbiBzdGF0ZTogKDA6Y2VudGVyLCAxOmxlZnRfYXJtLCAyOnJpZ2h0X2FybSwgMzpjdWVfbG9jYXRpb24pIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJuYW1lIjogInNfbG9jIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkNvbnRleHQgc3RhdGU6ICgwOnJld2FyZF9sZWZ0LCAxOnJld2FyZF9yaWdodCkiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgIm5hbWUiOiAic19jdHgiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiRGlzY3JldGUgdGltZSBzdGVwIiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInQiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTG9jYXRpb24gb2JzZXJ2YXRpb246ICgwOmNlbnRlciwgMTpsZWZ0LCAyOnJpZ2h0LCAzOmN1ZSkiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAib19sb2MiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUmV3YXJkL2N1ZSBvYnNlcnZhdGlvbjogKDA6bm9fcmV3YXJkLCAxOnJld2FyZCwgMjpjdWVfbGVmdCkiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAib19yZXciLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUG9saWN5IG92ZXIgNCBhY3Rpb25zOiAoZ29fbGVmdCwgZ29fcmlnaHQsIGdvX2N1ZSwgc3RheSkiLCAiZGltZW5zaW9ucyI6IFs0XSwgIm5hbWUiOiAicGkiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiU2VsZWN0ZWQgYWN0aW9uIiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInUiLCAidHlwZSI6ICJmbG9hdCJ9XX0="
const GNN_SPEC = JSON.parse(String(base64decode(GNN_SPEC_JSON_B64)))

function package_version(name::String)
for (_, dep) in Pkg.dependencies()
    if dep.name == name
        return string(dep.version)
    end
end
return "unknown"
end

# --- Real RxInfer.jl generative model ---
# The @model definition is precompiled in the GnnRxInferModels package module.
# Using `using` loads the precompiled cache (built once via PrecompileTools.jl),
# eliminating ~85s of JIT compilation on every run.
#
# The model is a generative POMDP: hidden states evolve via
# DiscreteTransition conditioned on the previous state and selected action;
# observations are emitted via DiscreteTransition through the likelihood
# matrix A.

using GnnRxInferModels: pomdp_model

# --- Custom EFE computation (Active Inference domain, not RxInfer's) ---

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

# Policy: softmax(log E - gamma * EFE). The habit prior E enters via
# log-add (Active Inference habit term); with the uniform default E the
# log-term is constant and cancels inside softmax, preserving the
# E-less behavior exactly.
function select_action(belief, A, B, C_pref, E_prior)
efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
action = categorical_index(policy)
return action, efe_values, policy
end

function compute_efe_and_policy(belief, A, B, C_pref, E_prior)
efe_values = [compute_efe(belief, action, A, B, C_pref) for action in 1:size(B, 3)]
policy = softmax(log.(max.(E_prior, 1e-16)) .- ACTION_PRECISION .* efe_values)
return efe_values, policy
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

# --- Offline batch inference (Bayesian smoothing) with post-hoc EFE policy
# evaluation.
#
# This is NOT online active inference. The pipeline is:
#   Phase 1 — Forward simulation for data collection: run the environment
#     forward using the hand-rolled EFE to collect observations, actions,
#     and true states. (The hand-rolled forward filter here is a data
#     collection mechanism, not a substitute for RxInfer inference.)
#   Phase 2 — Real RxInfer batch inference: run infer() with
#     free_energy=true on the collected data. If infer() fails, the script
#     crashes (exit non-zero). There is NO fallback.
#   Phase 3 — Posterior extraction: extract per-timestep smoothed posteriors
#     from result.posteriors[:s].
#   Phase 4 — Post-hoc EFE/policy from posteriors: compute EFE and policy
#     from the smoothed posteriors. These are post-hoc policy evaluations,
#     not online control.

function belief_entropy(belief)
# Shannon entropy in nats. Returns 0 for a degenerate point-mass.
safe = max.(belief, 1e-16)
return -sum(safe .* log.(safe))
end

function run_simulation()
Random.seed!(RANDOM_SEED)
initial = GNN_SPEC["initialparameterization"]
A = zeros(Float64, NUM_OBSERVATIONS, NUM_STATES)
raw_A = initial["A"]
for obs in 1:NUM_OBSERVATIONS
    row = collect(raw_A[obs])
    for state in 1:NUM_STATES
        A[obs, state] = Float64(row[state])
    end
end
# B is stored as (next_state, previous_state, action)
raw_B = initial["B"]
B = zeros(Float64, NUM_STATES, NUM_STATES, NUM_ACTIONS)
for ns in 1:NUM_STATES
    for ps in 1:NUM_STATES
        for a in 1:NUM_ACTIONS
            B[ns, ps, a] = Float64(raw_B[ns][ps][a])
        end
    end
end
C = Float64.(collect(initial["C"]))
D = Float64.(collect(initial["D"]))
E = haskey(initial, "E") ? Float64.(collect(initial["E"])) : fill(1.0 / NUM_ACTIONS, NUM_ACTIONS)
if length(E) != NUM_ACTIONS
    error("E length $(length(E)) does not match expected $NUM_ACTIONS")
end
E = E ./ sum(E)  # normalize the habit prior
validate_dimensions(A, B, C, D)

C_pref = softmax(C)

# --- Phase 1: Forward simulation for data collection ---
# Uses a hand-rolled EFE-based forward filter to collect the observation
# and action sequence. This is NOT the inference step — it is data
# collection for the subsequent RxInfer batch inference.
current_state = categorical_index(D)
current_belief = copy(D)

observations = Int[]
true_states = Int[]
actions = Int[]
action_seq_full = Int[]  # 1-indexed actions for the model

for step in 1:TIME_STEPS
    observation = categorical_index(A[:, current_state])
    emitting_state = current_state  # the state that generated this observation

    # Simple Bayesian update for the forward-pass belief
    obs_onehot = [i == observation ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS]
    likelihood = A[observation, :]
    updated = current_belief .* likelihood
    if sum(updated) <= 0
        error("belief update produced zero mass at step $step")
    end
    current_belief = updated ./ sum(updated)

    # Action selection via EFE + habit prior E (forward-pass policy)
    action, efe_values, policy = select_action(current_belief, A, B, C_pref, E)

    # Environment transition
    next_probs = B[:, current_state, action]
    current_state = categorical_index(next_probs)

    # Predict next belief
    predicted = B[:, :, action] * current_belief
    current_belief = predicted ./ sum(predicted)

    push!(observations, observation - 1)  # 0-indexed for JSON
    push!(true_states, emitting_state - 1)  # state that emitted observation t (matches beliefs[t])
    push!(actions, action - 1)  # 0-indexed for JSON
    push!(action_seq_full, action)  # 1-indexed for model
end

# --- Phase 2: Real RxInfer batch inference (no fallback) ---
# Build one-hot observation sequence for the model
obs_seq = [[i == (obs + 1) ? 1.0 : 0.0 for i in 1:NUM_OBSERVATIONS] for obs in observations]

    # The model needs u[1:T-1] for transitions, plus a padding u[T]
model_actions = copy(action_seq_full)
while length(model_actions) < TIME_STEPS
    push!(model_actions, 1)
end

# NO try/catch — if infer() fails, the script crashes with a clear error.
# This is deliberate: real RxInfer inference or nothing.
result = infer(
    model = pomdp_model(A=A, B=B, D=D, u=model_actions, T=TIME_STEPS),
    data = (y = obs_seq,),
    iterations = INFERENCE_ITERATIONS,
    free_energy = true
)

uses_real_rxinfer = true  # only reached if infer() succeeded

# --- Phase 3: Posterior extraction (smoothed posteriors) ---
# RxInfer returns posteriors[:s] as Vector of Vector of Categorical.
# Outer index = iteration, inner index = time step.
# We take the final iteration's posteriors — these are smoothed
# (joint) posteriors from batch inference, not filtered (online) beliefs.
posteriors_s = result.posteriors[:s]
final_iter = posteriors_s[end]
if isa(final_iter, Vector)
    posterior_per_step = final_iter
else
    # Single Categorical (T=1 case)
    posterior_per_step = [final_iter]
end

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

    # Phase 4: Post-hoc EFE and policy from the smoothed posterior
    efe_vals, pol = compute_efe_and_policy(belief, A, B, C_pref, E)
    push!(efe_per_action, efe_vals)
    push!(selected_efe, efe_vals[action_seq_full[t]])
    push!(policy_posterior, pol)
end

# --- VFE recording: per-iteration trace (the real convergence diagnostic) ---
# RxInfer returns one VFE scalar per inference iteration (for the whole
# model), NOT per timestep. We record the full per-iteration vector.
vfe_per_iteration = Float64.(result.free_energy)  # length = INFERENCE_ITERATIONS

# variational_free_energy (consumed by the analyzer): report the
# per-iteration trace directly. This is per-iteration, not per-step.
# Documented clearly in the results dict and the analyzer.
variational_free_energy = copy(vfe_per_iteration)

# Convergence check using the real per-iteration trace
if length(vfe_per_iteration) >= 5
    last_5 = vfe_per_iteration[end-4:end]
    inference_converged = (maximum(last_5) - minimum(last_5)) < 1e-4
elseif length(vfe_per_iteration) >= 2
    inference_converged = abs(vfe_per_iteration[end] - vfe_per_iteration[end-1]) < 1e-4
else
    inference_converged = false  # too few iterations to assess
end

# --- Strengthened validation ---
vfe_present = !isempty(vfe_per_iteration) && all(v -> v > 0, vfe_per_iteration)

# Belief-entropy diagnostics. Exact Bayesian smoothing legitimately
# produces near-zero-entropy marginals in high-signal regimes (each
# marginal conditions on the WHOLE observation sequence), so low entropy
# is not a failure by itself — systematic collapse only signals failure
# when the beliefs also point at the WRONG states, which the
# chance-relative accuracy gate below catches. belief_entropy_ok
# therefore flags only the pathological combination: every timestep
# degenerate AND accuracy below the gate. Raw entropy stats are
# reported alongside for diagnosis.
is_identity_A = all(abs(A[i,j] - (i == j ? 1.0 : 0.0)) < 0.01
                    for i in 1:size(A,1), j in 1:size(A,2))
min_entropy = is_identity_A ? 0.0 : 0.1  # collapse threshold (nats)
belief_entropies = [belief_entropy(b) for b in beliefs]
all_beliefs_degenerate = !isempty(belief_entropies) &&
    maximum(belief_entropies) < min_entropy

# Belief accuracy: check that argmax(belief) matches the true state
# for a majority of timesteps. This catches systematic inference failures
# where beliefs are valid distributions but point at the wrong state.
belief_accuracy = 0.0
if length(beliefs) == length(true_states) && length(beliefs) > 0
    correct = 0
    for t in 1:length(beliefs)
        if argmax(beliefs[t]) == (true_states[t] + 1)  # true_states are 0-indexed
            correct += 1
        end
    end
    belief_accuracy = Float64(correct) / length(beliefs)
end
# Identity A (fully observable): expect high accuracy. Non-identity A:
# require accuracy meaningfully above chance (the old 0.0 threshold was
# vacuously true) — twice chance, capped at 0.5.
min_accuracy = is_identity_A ? 0.5 : min(0.5, 2.0 / NUM_STATES)
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
    "belief_accuracy_ok" => belief_accuracy_ok
)
validation["all_valid"] = validation["all_beliefs_valid"] &&
    validation["beliefs_sum_to_one"] &&
    validation["actions_in_range"] &&
    validation["inference_converged"] &&
    validation["vfe_present"] &&
    validation["belief_entropy_ok"] &&
    validation["belief_accuracy_ok"]

# Compute script SHA256 for reproducibility tracking
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
    "observations_by_modality" => Dict("joint_observation" => observations),
    "hidden_states_by_factor" => Dict("joint_state" => true_states),
    "actions_by_control_factor" => Dict("joint_action" => actions),
    "beliefs_by_factor" => Dict("joint_state" => beliefs),
    "expected_free_energy" => selected_efe,
    "efe_per_action" => efe_per_action,
    "variational_free_energy" => variational_free_energy,
    "vfe_per_iteration" => vfe_per_iteration,
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
        "E" => E,
        "num_states" => NUM_STATES,
        "num_observations" => NUM_OBSERVATIONS,
        "num_actions" => NUM_ACTIONS,
        "inference_iterations" => INFERENCE_ITERATIONS,
        # Per-factor structure echoed from the GNN spec so downstream
        # analysis can un-flatten joint posteriors into per-factor
        # (per-agent) marginals without re-parsing the GNN file.
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

# --- Structured per-step execution log (JSON Lines: one record per step).
# Captures per-step beliefs / action / EFE / policy posterior / validation,
# written alongside simulation_results.json. Pure JSON + Base stdlib, and
# guarded so logging can never crash the simulation.
function write_execution_log(results)
log_path = "simulation.log"
beliefs = get(get(results, "beliefs_by_factor", Dict()), "joint_state", results["beliefs"])
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

# Complete structured JSON sidecar for downstream tooling that prefers a
# single document over JSONL.
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

# --- Julia-native visualization via Plots.jl (matplotlib-free PNGs).
# Everything is wrapped in try/catch so a missing Plots backend degrades to a
# warning and NEVER prevents the simulation from running to completion.
function write_plots(results)
if !PLOTS_READY
    println("⚠️ Skipping PNG plots (Plots backend not available)")
    return
end
try
    beliefs = get(get(results, "beliefs_by_factor", Dict()), "joint_state", results["beliefs"])
    efe = results["expected_free_energy"]
    policy = results["policy_posterior"]

    if !isempty(beliefs)
        belief_mat = hcat(beliefs...)
        steps = 1:size(belief_mat, 2)
        p1 = plot(
            title = "Belief Evolution over Time",
            xlabel = "Time step",
            ylabel = "Belief mass",
            legend = :outertopright,
            size = (900, 450),
            titlefontsize = 12,
            guidefontsize = 10,
            legendfontsize = 8,
            tickfontsize = 8,
            linewidth = 2
        )
        for state in 1:size(belief_mat, 1)
            plot!(p1, steps, belief_mat[state, :], label = "State $state")
        end
        savefig(p1, "belief_evolution.png")
    end

    if !isempty(efe)
        p2 = plot(
            1:length(efe), efe,
            title = "Expected Free Energy over Time",
            xlabel = "Time step",
            ylabel = "Action EFE",
            label = "selected EFE",
            legend = :topright,
            size = (900, 400),
            titlefontsize = 12,
            guidefontsize = 10,
            legendfontsize = 8,
            tickfontsize = 8,
            linewidth = 2
        )
        savefig(p2, "efe_over_time.png")
    end

    if !isempty(policy)
        policy_mat = hcat(policy...)
        p3 = heatmap(policy_mat,
            title = "Policy Posterior over Time",
            xlabel = "Time step",
            ylabel = "Action",
            color = :viridis,
            colorbar = :right,
            size = (900, 400),
            titlefontsize = 12,
            guidefontsize = 10,
            tickfontsize = 8
        )
        savefig(p3, "policy_posterior.png")
    end

    println("RxInfer.jl simulation wrote PNG plots (belief_evolution.png, efe_over_time.png, policy_posterior.png)")
catch e
    println("⚠️ Plotting skipped (Plots backend unavailable): $e")
end
end

function main()
results = run_simulation()
# Sanitize NaN/Inf values before JSON serialization (JSON.jl rejects them by default)
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
