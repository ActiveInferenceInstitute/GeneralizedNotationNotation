#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation
# Generated from GNN Model: T-Maze Epistemic Foraging Agent
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
const MODEL_NAME = "T-Maze Epistemic Foraging Agent"
const NUM_STATES = 8
const NUM_OBSERVATIONS = 12
const NUM_ACTIONS = 4
const TIME_STEPS = 3
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkRfbG9jIiwgInRhcmdldCI6ICJzX2xvYyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkRfY3R4IiwgInRhcmdldCI6ICJzX2N0eCJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbG9jIiwgInRhcmdldCI6ICJBX2xvYyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkFfbG9jIiwgInRhcmdldCI6ICJvX2xvYyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbG9jIiwgInRhcmdldCI6ICJBX3JldyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfY3R4IiwgInRhcmdldCI6ICJBX3JldyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkFfcmV3IiwgInRhcmdldCI6ICJvX3JldyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfbG9jIiwgInRhcmdldCI6ICJCX2xvYyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfY3R4IiwgInRhcmdldCI6ICJCX2N0eCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkNfcmV3IiwgInRhcmdldCI6ICJHX2lucyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkdfZXBpIiwgInRhcmdldCI6ICJHIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiR19pbnMiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHIiwgInRhcmdldCI6ICJwaSJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInBpIiwgInRhcmdldCI6ICJ1In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQl9sb2MiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzX2xvYyIsICJ0YXJnZXQiOiAiRiJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInNfY3R4IiwgInRhcmdldCI6ICJGIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAib19sb2MiLCAidGFyZ2V0IjogIkYifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJvX3JldyIsICJ0YXJnZXQiOiAiRiJ9XSwgImRlc2NyaXB0aW9uIjogIlRoZSBjbGFzc2ljIFQtbWF6ZSB0YXNrIGZyb20gQWN0aXZlIEluZmVyZW5jZSBsaXRlcmF0dXJlIChGcmlzdG9uIGV0IGFsLik6XG4tIEFnZW50IG5hdmlnYXRlcyBhIFQtc2hhcGVkIG1hemUgd2l0aCA0IGxvY2F0aW9uczogY2VudGVyLCBsZWZ0IGFybSwgcmlnaHQgYXJtLCBjdWUgbG9jYXRpb25cbi0gVHdvIG9ic2VydmF0aW9uIG1vZGFsaXRpZXM6IGxvY2F0aW9uICh3aGVyZSBhbSBJPykgYW5kIHJld2FyZC9jdWUgKHdoYXQgZG8gSSBzZWU/KVxuLSBSZXdhcmQgaXMgaGlkZGVuIGJlaGluZCBvbmUgb2YgdGhlIHR3byBhcm1zIChsZWZ0IG9yIHJpZ2h0KSwgZGV0ZXJtaW5lZCBieSBjb250ZXh0XG4tIEN1ZSBsb2NhdGlvbiBwcm92aWRlcyBwYXJ0aWFsIGluZm9ybWF0aW9uIGFib3V0IHdoaWNoIGFybSBob2xkcyB0aGUgcmV3YXJkXG4tIEFnZW50IG11c3QgZGVjaWRlOiBnbyBkaXJlY3RseSB0byBhbiBhcm0gKGV4cGxvaXQpIG9yIHZpc2l0IGN1ZSBsb2NhdGlvbiBmaXJzdCAoZXhwbG9yZSlcbi0gRGVtb25zdHJhdGVzIGVwaXN0ZW1pYyBmb3JhZ2luZzogQWN0aXZlIEluZmVyZW5jZSBuYXR1cmFsbHkgYmFsYW5jZXMgZXhwbG9yYXRpb24gdnMgZXhwbG9pdGF0aW9uXG4tIFRoZSBFeHBlY3RlZCBGcmVlIEVuZXJneSBkZWNvbXBvc2VzIGludG8gZXBpc3RlbWljIChpbmZvcm1hdGlvbiBnYWluKSArIGluc3RydW1lbnRhbCAocmV3YXJkKSB2YWx1ZSIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzEuMCwgMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wXV0sICJCIjogW1tbMC4yNSwgMC4yNSwgMC4yNSwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC41LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDEuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC41LCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuNSwgMC4wXV0sIFtbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC41LCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC41XV0sIFtbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC41XV0sIFtbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDEuMCwgMC41LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMjUsIDAuMjUsIDAuMjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjI1LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAxLjAsIDAuNSwgMC4wXV1dLCAiQyI6IFstMS4wLCAzLjAsIDAuMCwgLTEuMCwgMy4wLCAwLjAsIC0xLjAsIDMuMCwgMC4wLCAtMS4wLCAzLjAsIDAuMF0sICJEIjogWzAuNSwgMC41LCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wXX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMS4wLCAxLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDEuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAxLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMS4wLCAwLjBdXSwgIkIiOiBbW1swLjI1LCAwLjI1LCAwLjI1LCAxLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuNSwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMS4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC41LCAwLjBdXSwgW1swLjI1LCAwLjI1LCAwLjI1LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuNSwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjUsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdXSwgW1swLjI1LCAwLjI1LCAwLjI1LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAwLjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC41LCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjVdXSwgW1swLjI1LCAwLjI1LCAwLjI1LCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4yNSwgMC4yNSwgMC4yNV0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMS4wLCAwLjUsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4yNSwgMC4yNSwgMC4yNSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMjUsIDAuMjUsIDAuMjVdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjI1LCAwLjI1XSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDEuMCwgMC41LCAwLjBdXV0sICJDIjogWy0xLjAsIDMuMCwgMC4wLCAtMS4wLCAzLjAsIDAuMCwgLTEuMCwgMy4wLCAwLjAsIC0xLjAsIDMuMCwgMC4wXSwgIkQiOiBbMC41LCAwLjUsIDAuMCwgMC4wLCAwLjAsIDAuMCwgMC4wLCAwLjBdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzEyLCA4XSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiQV9sb2MiLCAiQV9yZXciXX0sICJBX2xvYyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkFfcmV3IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCA0LCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbOCwgOCwgNF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkJfY3R4IiwgIkJfbG9jIl0sICJzb3VyY2Vfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24ifSwgIkJfY3R4IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyLCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQl9sb2MiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzEyXSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiQ19sb2MiLCAiQ19yZXciXX0sICJDX2xvYyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkNfcmV3IjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IHRydWUsICJzaGFwZSI6IFs4XSwgInNvdXJjZSI6ICJmYWN0b3JlZF9qb2ludF9jb21wb3NpdGlvbiIsICJzb3VyY2Vfa2V5cyI6IFsiRF9jdHgiLCAiRF9sb2MiXX0sICJEX2N0eCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkRfbG9jIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiVC1NYXplIEVwaXN0ZW1pYyBGb3JhZ2luZyBBZ2VudCIsICJtb2RlbF9wYXJhbWV0ZXJzIjogeyJiX3RlbnNvcl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBvdmVyIDQgYWN0aW9uczogKGdvX2xlZnQsIGdvX3JpZ2h0LCBnb19jdWUsIHN0YXkpIiwgImRpbWVuc2lvbnMiOiBbNF0sICJpbmRleCI6IDEsICJuYW1lIjogInBpIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiU2VsZWN0ZWQgYWN0aW9uIiwgImRpbWVuc2lvbnMiOiBbMV0sICJpbmRleCI6IDIsICJuYW1lIjogInUiLCAic2l6ZSI6IDEsICJ0eXBlIjogImZsb2F0In1dLCAibnVtX2FjdGlvbnMiOiA0LCAibnVtX2NvbnRleHRzIjogMiwgIm51bV9oaWRkZW5fc3RhdGVzIjogOCwgIm51bV9sb2NhdGlvbl9vYnMiOiA0LCAibnVtX2xvY2F0aW9ucyI6IDQsICJudW1fbW9kYWxpdGllcyI6IDIsICJudW1fb2JzIjogMTIsICJudW1fcmV3YXJkX29icyI6IDMsICJudW1fc3RhdGVfZmFjdG9ycyI6IDIsICJudW1fdGltZXN0ZXBzIjogMywgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIkxvY2F0aW9uIG9ic2VydmF0aW9uOiAoMDpjZW50ZXIsIDE6bGVmdCwgMjpyaWdodCwgMzpjdWUpIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogIm9fbG9jIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUmV3YXJkL2N1ZSBvYnNlcnZhdGlvbjogKDA6bm9fcmV3YXJkLCAxOnJld2FyZCwgMjpjdWVfbGVmdCkiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAib19yZXciLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dLCAicGFzc2l2ZV9tb2RlbCI6IGZhbHNlLCAic2ltdWxhdGlvbl9wYXJhbXMiOiB7fSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkxvY2F0aW9uIHN0YXRlOiAoMDpjZW50ZXIsIDE6bGVmdF9hcm0sIDI6cmlnaHRfYXJtLCAzOmN1ZV9sb2NhdGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAic19sb2MiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJDb250ZXh0IHN0YXRlOiAoMDpyZXdhcmRfbGVmdCwgMTpyZXdhcmRfcmlnaHQpIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogInNfY3R4IiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9XX0sICJuYW1lIjogIlQtTWF6ZSBFcGlzdGVtaWMgRm9yYWdpbmcgQWdlbnQiLCAib250b2xvZ3lfbWFwcGluZyI6IHsiQV9sb2MiOiAiTG9jYXRpb25MaWtlbGlob29kTWF0cml4IiwgIkFfcmV3IjogIlJld2FyZExpa2VsaWhvb2RNYXRyaXgiLCAiQl9jdHgiOiAiQ29udGV4dFRyYW5zaXRpb25NYXRyaXgiLCAiQl9sb2MiOiAiTG9jYXRpb25UcmFuc2l0aW9uTWF0cml4IiwgIkNfbG9jIjogIkxvY2F0aW9uUHJlZmVyZW5jZVZlY3RvciIsICJDX3JldyI6ICJSZXdhcmRQcmVmZXJlbmNlVmVjdG9yIiwgIkRfY3R4IjogIkNvbnRleHRQcmlvciIsICJEX2xvYyI6ICJMb2NhdGlvblByaW9yIiwgIkYiOiAiVmFyaWF0aW9uYWxGcmVlRW5lcmd5IiwgIkciOiAiRXhwZWN0ZWRGcmVlRW5lcmd5IiwgIkdfZXBpIjogIkVwaXN0ZW1pY1ZhbHVlIiwgIkdfaW5zIjogIkluc3RydW1lbnRhbFZhbHVlIiwgIm9fbG9jIjogIkxvY2F0aW9uT2JzZXJ2YXRpb24iLCAib19yZXciOiAiUmV3YXJkT2JzZXJ2YXRpb24iLCAicGkiOiAiUG9saWN5VmVjdG9yIiwgInNfY3R4IjogIkNvbnRleHRIaWRkZW5TdGF0ZSIsICJzX2xvYyI6ICJMb2NhdGlvbkhpZGRlblN0YXRlIiwgInQiOiAiVGltZSIsICJ1IjogIkFjdGlvbiJ9LCAic3RydWN0dXJlZF9wb21kcCI6IHsiYWRhcHRlcl9ub3RlcyI6IFtdLCAiY2Fub25pY2FsX2Jfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY29udHJvbF9mYWN0b3JzIjogW3siY29tbWVudCI6ICJQb2xpY3kgb3ZlciA0IGFjdGlvbnM6IChnb19sZWZ0LCBnb19yaWdodCwgZ29fY3VlLCBzdGF5KSIsICJkaW1lbnNpb25zIjogWzRdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJwaSIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlNlbGVjdGVkIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAyLCAibmFtZSI6ICJ1IiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm1hdHJpY2VzIjogeyJBX2xvYyI6IFtbMS4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMS4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMS4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMS4wXV0sICJBX3JldyI6IFtbWzEuMCwgMS4wXSwgWzAuMCwgMS4wXSwgWzEuMCwgMC4wXSwgWzAuMCwgMS4wXV0sIFtbMC4wLCAwLjBdLCBbMS4wLCAwLjBdLCBbMC4wLCAxLjBdLCBbMC4wLCAwLjBdXSwgW1swLjAsIDAuMF0sIFswLjAsIDAuMF0sIFswLjAsIDAuMF0sIFsxLjAsIDAuMF1dXSwgIkJfY3R4IjogW1sxLjAsIDAuMF0sIFswLjAsIDEuMF1dLCAiQl9sb2MiOiBbW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMS4wLCAxLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzEuMCwgMC4wLCAxLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAxLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFsxLjAsIDAuMCwgMC4wLCAxLjBdXSwgW1sxLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDEuMCwgMS4wLCAwLjBdXV0sICJDX2xvYyI6IFswLjAsIDAuMCwgMC4wLCAwLjBdLCAiQ19yZXciOiBbLTEuMCwgMy4wLCAwLjBdLCAiRF9jdHgiOiBbMC41LCAwLjVdLCAiRF9sb2MiOiBbMS4wLCAwLjAsIDAuMCwgMC4wXX0sICJtYXRyaXhfcHJvdmVuYW5jZSI6IHsiQSI6IHsiZGVyaXZlZCI6IHRydWUsICJzaGFwZSI6IFsxMiwgOF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkFfbG9jIiwgIkFfcmV3Il19LCAiQV9sb2MiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJBX3JldyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMywgNCwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzgsIDgsIDRdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJCX2N0eCIsICJCX2xvYyJdLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJCX2N0eCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkJfbG9jIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0LCA0LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IHRydWUsICJzaGFwZSI6IFsxMl0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkNfbG9jIiwgIkNfcmV3Il19LCAiQ19sb2MiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDX3JldyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbM10sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbOF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkRfY3R4IiwgIkRfbG9jIl19LCAiRF9jdHgiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEX2xvYyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJMb2NhdGlvbiBvYnNlcnZhdGlvbjogKDA6Y2VudGVyLCAxOmxlZnQsIDI6cmlnaHQsIDM6Y3VlKSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvX2xvYyIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlJld2FyZC9jdWUgb2JzZXJ2YXRpb246ICgwOm5vX3Jld2FyZCwgMTpyZXdhcmQsIDI6Y3VlX2xlZnQpIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDEsICJuYW1lIjogIm9fcmV3IiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkxvY2F0aW9uIHN0YXRlOiAoMDpjZW50ZXIsIDE6bGVmdF9hcm0sIDI6cmlnaHRfYXJtLCAzOmN1ZV9sb2NhdGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAic19sb2MiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJDb250ZXh0IHN0YXRlOiAoMDpyZXdhcmRfbGVmdCwgMTpyZXdhcmRfcmlnaHQpIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogInNfY3R4IiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9XX0sICJ2YXJpYWJsZXMiOiBbeyJjb21tZW50IjogIkxvY2F0aW9uIHN0YXRlOiAoMDpjZW50ZXIsIDE6bGVmdF9hcm0sIDI6cmlnaHRfYXJtLCAzOmN1ZV9sb2NhdGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAic19sb2MiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ29udGV4dCBzdGF0ZTogKDA6cmV3YXJkX2xlZnQsIDE6cmV3YXJkX3JpZ2h0KSIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAibmFtZSI6ICJzX2N0eCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJEaXNjcmV0ZSB0aW1lIHN0ZXAiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJMb2NhdGlvbiBvYnNlcnZhdGlvbjogKDA6Y2VudGVyLCAxOmxlZnQsIDI6cmlnaHQsIDM6Y3VlKSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJvX2xvYyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJSZXdhcmQvY3VlIG9ic2VydmF0aW9uOiAoMDpub19yZXdhcmQsIDE6cmV3YXJkLCAyOmN1ZV9sZWZ0KSIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAibmFtZSI6ICJvX3JldyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJSZXdhcmQgcHJlZmVyZW5jZTogc3Ryb25nbHkgcHJlZmVycyByZXdhcmQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFszXSwgIm5hbWUiOiAiQ19yZXciLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTG9jYXRpb24gdHJhbnNpdGlvbnM6IFAoc19sb2MnIHwgc19sb2MsIGFjdGlvbikiLCAiZGltZW5zaW9ucyI6IFs0LCA0LCA0XSwgIm5hbWUiOiAiQl9sb2MiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUG9saWN5IG92ZXIgNCBhY3Rpb25zOiAoZ29fbGVmdCwgZ29fcmlnaHQsIGdvX2N1ZSwgc3RheSkiLCAiZGltZW5zaW9ucyI6IFs0XSwgIm5hbWUiOiAicGkiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiU2VsZWN0ZWQgYWN0aW9uIiwgImRpbWVuc2lvbnMiOiBbMV0sICJuYW1lIjogInUiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiRXhwZWN0ZWQgRnJlZSBFbmVyZ3kgcGVyIHBvbGljeSIsICJkaW1lbnNpb25zIjogWyJwaSJdLCAibmFtZSI6ICJHIiwgInR5cGUiOiAiZmxvYXQifV19"
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
