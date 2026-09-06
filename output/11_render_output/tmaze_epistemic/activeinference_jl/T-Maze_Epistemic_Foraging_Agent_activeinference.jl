#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: T-Maze Epistemic Foraging Agent

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
const MODEL_NAME = "T-Maze Epistemic Foraging Agent"
const NUM_STATES = 8
const NUM_OBSERVATIONS = 12
const NUM_ACTIONS = 4
const TIME_STEPS = 3
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
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
