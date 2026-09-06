#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Factorized Posterior Agent

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
const MODEL_NAME = "Factorized Posterior Agent"
const NUM_STATES = 8
const NUM_OBSERVATIONS = 6
const NUM_ACTIONS = 3
const TIME_STEPS = 15
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkRfZjAiLCAidGFyZ2V0IjogInNfZjAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJEX2YxIiwgInRhcmdldCI6ICJzX2YxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiKHNfZjAsIHUpIiwgInRhcmdldCI6ICJCX2YwIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQl9mMCIsICJ0YXJnZXQiOiAic19mMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInNfZjEiLCAidGFyZ2V0IjogIkJfZjEifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCX2YxIiwgInRhcmdldCI6ICJzX2YxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiKHNfZjAsIHNfZjEpIiwgInRhcmdldCI6ICJBX20wIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQV9tMCIsICJ0YXJnZXQiOiAib19tMCJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInNfZjAiLCAidGFyZ2V0IjogIkFfbTEifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJBX20xIiwgInRhcmdldCI6ICJvX20xIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQ19tMCIsICJ0YXJnZXQiOiAib19tMCJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkNfbTEiLCAidGFyZ2V0IjogIm9fbTEifV0sICJkZXNjcmlwdGlvbiI6ICJBIG1lYW4tZmllbGQgZmFjdG9yaXplZCBQT01EUCBhZ2VudC4gVGhlIGpvaW50IHBvc3RlcmlvciBvdmVyIHR3b1xuaW5kZXBlbmRlbnQgc3RhdGUgZmFjdG9ycyBgc18xYCAobG9jYXRpb24pIGFuZCBgc18yYCAoZ29hbCBpZGVudGl0eSkgaXNcbmFwcHJveGltYXRlZCBhcyB0aGUgcHJvZHVjdCBvZiBtYXJnaW5hbHMgUShzXzEsIHNfMikgPSBRKHNfMSkgKiBRKHNfMikuXG5UaGlzIGlzIHRoZSBjYW5vbmljYWwgc2ltcGxpZmljYXRpb24gdXNlZCBpbiB2YXJpYXRpb25hbCBpbmZlcmVuY2Ugd2hlblxuZXhhY3Qgam9pbnQgcG9zdGVyaW9ycyBhcmUgY29tcHV0YXRpb25hbGx5IGludHJhY3RhYmxlLlxuLSBUd28gc3RhdGUgZmFjdG9yczogbG9jYXRpb24gKDQgc3RhdGVzKSwgZ29hbCAoMiBzdGF0ZXMpXG4tIFR3byBvYnNlcnZhdGlvbiBtb2RhbGl0aWVzOiB2aXN1YWwgKDMgb2JzKSwgcHJvcHJpb2NlcHRpdmUgKDIgb2JzKVxuLSBTZXBhcmF0ZSB0cmFuc2l0aW9uIG1hdHJpY2VzIEJfMSAobG9jYXRpb24gXHUwMGQ3IGFjdGlvbikgYW5kIEJfMiAoZ29hbCBpcyBzdGF0aWMpXG4tIEV4cGxpY2l0IGZhY3Rvcml6YXRpb24gZGVjbGFyZWQgaW4gIyMgRXF1YXRpb25zXG4tIFRlc3RzIG11bHRpLWZhY3RvciAvIG11bHRpLW1vZGFsaXR5IGhhbmRsaW5nIGluIHRoZSBwYXJzZXIiLCAiZ25uX3NlY3Rpb24iOiAiQWN0SW5mRmFjdG9yaXplZCIsICJpbml0aWFsX3BhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuNjMsIDAuMDkwMDAwMDAwMDAwMDAwMDEsIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjA2OTk5OTk5OTk5OTk5OTk5LCAwLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wMTAwMDAwMDAwMDAwMDAwMDIsIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjAxMDAwMDAwMDAwMDAwMDAwMl0sIFswLjA2OTk5OTk5OTk5OTk5OTk5LCAwLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC42MywgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMV0sIFswLjA5MDAwMDAwMDAwMDAwMDAxLCAwLjYzLCAwLjA2OTk5OTk5OTk5OTk5OTk5LCAwLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wMTAwMDAwMDAwMDAwMDAwMDIsIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC4wMTAwMDAwMDAwMDAwMDAwMDIsIDAuMDY5OTk5OTk5OTk5OTk5OTksIDAuNjMsIDAuMDkwMDAwMDAwMDAwMDAwMDEsIDAuMDkwMDAwMDAwMDAwMDAwMDEsIDAuMDkwMDAwMDAwMDAwMDAwMDEsIDAuMDkwMDAwMDAwMDAwMDAwMDEsIDAuMDkwMDAwMDAwMDAwMDAwMDFdLCBbMC4xODAwMDAwMDAwMDAwMDAwMiwgMC4xODAwMDAwMDAwMDAwMDAwMiwgMC4wMjAwMDAwMDAwMDAwMDAwMDQsIDAuMDIwMDAwMDAwMDAwMDAwMDA0LCAwLjA4MDAwMDAwMDAwMDAwMDAyLCAwLjA4MDAwMDAwMDAwMDAwMDAyLCAwLjA4MDAwMDAwMDAwMDAwMDAyLCAwLjA4MDAwMDAwMDAwMDAwMDAyXSwgWzAuMDIwMDAwMDAwMDAwMDAwMDA0LCAwLjAyMDAwMDAwMDAwMDAwMDAwNCwgMC4xODAwMDAwMDAwMDAwMDAwMiwgMC4xODAwMDAwMDAwMDAwMDAwMiwgMC43MjAwMDAwMDAwMDAwMDAxLCAwLjcyMDAwMDAwMDAwMDAwMDEsIDAuNzIwMDAwMDAwMDAwMDAwMSwgMC43MjAwMDAwMDAwMDAwMDAxXV0sICJCIjogW1tbMC45LCAwLjEsIDAuOV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjksIDAuMSwgMC45XSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuOSwgMC4wXV0sIFtbMC4xLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuOSwgMC4xLCAwLjldLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC45LCAwLjEsIDAuOV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC45LCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjksIDAuMSwgMC45XSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuOSwgMC4xLCAwLjldLCBbMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4xXV0sIFtbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC45LCAwLjEsIDAuOV0sIFswLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC45LCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjksIDAuMSwgMC45XV1dLCAiQyI6IFswLjUsIDAuNSwgMC41LCAwLjUsIDEuNSwgMS41XSwgIkQiOiBbMC4xNSwgMC4xLCAwLjE1LCAwLjEsIDAuMTUsIDAuMSwgMC4xNSwgMC4xXX0sICJpbml0aWFscGFyYW1ldGVyaXphdGlvbiI6IHsiQSI6IFtbMC42MywgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wMTAwMDAwMDAwMDAwMDAwMDIsIDAuMDY5OTk5OTk5OTk5OTk5OTksIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wMTAwMDAwMDAwMDAwMDAwMDIsIDAuMDEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMDY5OTk5OTk5OTk5OTk5OTksIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjA5MDAwMDAwMDAwMDAwMDAxLCAwLjYzLCAwLjA5MDAwMDAwMDAwMDAwMDAxLCAwLjA5MDAwMDAwMDAwMDAwMDAxLCAwLjA5MDAwMDAwMDAwMDAwMDAxLCAwLjA5MDAwMDAwMDAwMDAwMDAxXSwgWzAuMDkwMDAwMDAwMDAwMDAwMDEsIDAuNjMsIDAuMDY5OTk5OTk5OTk5OTk5OTksIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wMTAwMDAwMDAwMDAwMDAwMDIsIDAuMDEwMDAwMDAwMDAwMDAwMDAyLCAwLjAxMDAwMDAwMDAwMDAwMDAwMl0sIFswLjAxMDAwMDAwMDAwMDAwMDAwMiwgMC4wNjk5OTk5OTk5OTk5OTk5OSwgMC42MywgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMSwgMC4wOTAwMDAwMDAwMDAwMDAwMV0sIFswLjE4MDAwMDAwMDAwMDAwMDAyLCAwLjE4MDAwMDAwMDAwMDAwMDAyLCAwLjAyMDAwMDAwMDAwMDAwMDAwNCwgMC4wMjAwMDAwMDAwMDAwMDAwMDQsIDAuMDgwMDAwMDAwMDAwMDAwMDIsIDAuMDgwMDAwMDAwMDAwMDAwMDIsIDAuMDgwMDAwMDAwMDAwMDAwMDIsIDAuMDgwMDAwMDAwMDAwMDAwMDJdLCBbMC4wMjAwMDAwMDAwMDAwMDAwMDQsIDAuMDIwMDAwMDAwMDAwMDAwMDA0LCAwLjE4MDAwMDAwMDAwMDAwMDAyLCAwLjE4MDAwMDAwMDAwMDAwMDAyLCAwLjcyMDAwMDAwMDAwMDAwMDEsIDAuNzIwMDAwMDAwMDAwMDAwMSwgMC43MjAwMDAwMDAwMDAwMDAxLCAwLjcyMDAwMDAwMDAwMDAwMDFdXSwgIkIiOiBbW1swLjksIDAuMSwgMC45XSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4xLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wXSwgWzAuOSwgMC4xLCAwLjldLCBbMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC45LCAwLjBdXSwgW1swLjEsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC45LCAwLjEsIDAuOV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC45LCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjksIDAuMSwgMC45XSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuOSwgMC4xLCAwLjldLCBbMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC45LCAwLjEsIDAuOV0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC4wLCAwLjFdXSwgW1swLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC45LCAwLjBdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjksIDAuMSwgMC45XSwgWzAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjBdLCBbMC4xLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4wXSwgWzAuOSwgMC4xLCAwLjldXV0sICJDIjogWzAuNSwgMC41LCAwLjUsIDAuNSwgMS41LCAxLjVdLCAiRCI6IFswLjE1LCAwLjEsIDAuMTUsIDAuMSwgMC4xNSwgMC4xLCAwLjE1LCAwLjFdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzYsIDhdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJBX20wIiwgIkFfbTEiXX0sICJBX20wIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCA0LCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQV9tMSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJkZXJpdmVkIjogdHJ1ZSwgImZhY3Rvcl9hY3Rpb25fY291bnRzIjogWzMsIDFdLCAia3JvbmVja2VyX2ZhY3Rvcml6ZWQiOiBmYWxzZSwgInNoYXBlIjogWzgsIDgsIDNdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJCX2YwIiwgIkJfZjEiXSwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQl9mMCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMywgNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkJfZjEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzIsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzZdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJDX20wIiwgIkNfbTEiXX0sICJDX20wIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQ19tMSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbOF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkRfZjAiLCAiRF9mMSJdfSwgIkRfZjAiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEX2YxIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiRmFjdG9yaXplZCBQb3N0ZXJpb3IgQWdlbnQiLCAibW9kZWxfcGFyYW1ldGVycyI6IHsiYl90ZW5zb3Jfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiY29udHJvbF9mYWN0b3JzIjogW3siY29tbWVudCI6ICIzIHBvc3NpYmxlIGFjdGlvbnM6IHN0YXksIGZvcndhcmQsIGJhY2t3YXJkIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInUiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDMsICJ0eXBlIjogImZsb2F0In1dLCAibnVtX2FjdGlvbnMiOiAzLCAibnVtX2ZhY3RvcnMiOiAyLCAibnVtX2hpZGRlbl9zdGF0ZXMiOiA4LCAibnVtX2hpZGRlbl9zdGF0ZXNfZmFjdG9yMCI6IDQsICJudW1faGlkZGVuX3N0YXRlc19mYWN0b3IxIjogMiwgIm51bV9tb2RhbGl0aWVzIjogMiwgIm51bV9vYnMiOiA2LCAibnVtX29ic19tb2RhbGl0eTAiOiAzLCAibnVtX29ic19tb2RhbGl0eTEiOiAyLCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDE1LCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiTW9kYWxpdHkgMDogdmlzdWFsIG9ic2VydmF0aW9uICgzIHZpc3VhbCBjdWVzKSIsICJkaW1lbnNpb25zIjogWzMsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvX20wIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAzLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiTW9kYWxpdHkgMTogcHJvcHJpb2NlcHRpdmUgb2JzZXJ2YXRpb24gKDIgYm9keSBzdGF0ZXMpIiwgImRpbWVuc2lvbnMiOiBbMiwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogIm9fbTEiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDIsICJ0eXBlIjogImZsb2F0In1dLCAicGFzc2l2ZV9tb2RlbCI6IGZhbHNlLCAic2ltdWxhdGlvbl9wYXJhbXMiOiB7fSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkZhY3RvciAwOiBhZ2VudCBsb2NhdGlvbiAoNCBwb3NzaWJsZSBwb3NpdGlvbnMpIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInNfZjAiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJGYWN0b3IgMTogZ29hbCBpZGVudGl0eSAoMiBwb3NzaWJsZSBnb2FscykiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic19mMSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV19LCAibmFtZSI6ICJGYWN0b3JpemVkIFBvc3RlcmlvciBBZ2VudCIsICJvbnRvbG9neV9tYXBwaW5nIjogeyJBX20wIjogIkxpa2VsaWhvb2RNYXRyaXhNb2RhbGl0eTAiLCAiQV9tMSI6ICJMaWtlbGlob29kTWF0cml4TW9kYWxpdHkxIiwgIkNfbTAiOiAiUHJlZmVyZW5jZU1vZGFsaXR5MCIsICJDX20xIjogIlByZWZlcmVuY2VNb2RhbGl0eTEiLCAiRF9mMCI6ICJQcmlvckZhY3RvcjAiLCAiRF9mMSI6ICJQcmlvckZhY3RvcjEiLCAib19tMCI6ICJPYnNlcnZhdGlvbk1vZGFsaXR5MCIsICJvX20xIjogIk9ic2VydmF0aW9uTW9kYWxpdHkxIiwgInNfZjAiOiAiSGlkZGVuU3RhdGVGYWN0b3IwIiwgInNfZjEiOiAiSGlkZGVuU3RhdGVGYWN0b3IxIiwgInUiOiAiQWN0aW9uIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIjMgcG9zc2libGUgYWN0aW9uczogc3RheSwgZm9yd2FyZCwgYmFja3dhcmQiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAidSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifV0sICJtYXRyaWNlcyI6IHsiQV9tMCI6IFtbWzAuNywgMC4xXSwgWzAuMSwgMC43XSwgWzAuMSwgMC4xXSwgWzAuMSwgMC4xXV0sIFtbMC4xLCAwLjddLCBbMC43LCAwLjFdLCBbMC4xLCAwLjFdLCBbMC4xLCAwLjFdXSwgW1swLjIsIDAuMl0sIFswLjIsIDAuMl0sIFswLjgsIDAuOF0sIFswLjgsIDAuOF1dXSwgIkFfbTEiOiBbWzAuOSwgMC4xLCAwLjEsIDAuMV0sIFswLjEsIDAuOSwgMC45LCAwLjldXSwgIkJfZjAiOiBbW1swLjksIDAuMSwgMC4wLCAwLjBdLCBbMC4xLCAwLjksIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjksIDAuMV0sIFswLjAsIDAuMCwgMC4xLCAwLjldXSwgW1swLjEsIDAuOSwgMC4wLCAwLjBdLCBbMC4wLCAwLjEsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjEsIDAuOV0sIFswLjksIDAuMCwgMC4wLCAwLjFdXSwgW1swLjksIDAuMCwgMC4wLCAwLjFdLCBbMC4xLCAwLjksIDAuMCwgMC4wXSwgWzAuMCwgMC4xLCAwLjksIDAuMF0sIFswLjAsIDAuMCwgMC4xLCAwLjldXV0sICJCX2YxIjogW1sxLjAsIDAuMF0sIFswLjAsIDEuMF1dLCAiQ19tMCI6IFswLjAsIDAuMCwgMS4wXSwgIkNfbTEiOiBbMC41LCAwLjVdLCAiRF9mMCI6IFswLjI1LCAwLjI1LCAwLjI1LCAwLjI1XSwgIkRfZjEiOiBbMC42LCAwLjRdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzYsIDhdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJBX20wIiwgIkFfbTEiXX0sICJBX20wIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszLCA0LCAyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQV9tMSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMiwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJkZXJpdmVkIjogdHJ1ZSwgImZhY3Rvcl9hY3Rpb25fY291bnRzIjogWzMsIDFdLCAia3JvbmVja2VyX2ZhY3Rvcml6ZWQiOiBmYWxzZSwgInNoYXBlIjogWzgsIDgsIDNdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJCX2YwIiwgIkJfZjEiXSwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQl9mMCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMywgNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkJfZjEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzIsIDJdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogdHJ1ZSwgInNoYXBlIjogWzZdLCAic291cmNlIjogImZhY3RvcmVkX2pvaW50X2NvbXBvc2l0aW9uIiwgInNvdXJjZV9rZXlzIjogWyJDX20wIiwgIkNfbTEiXX0sICJDX20wIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFszXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQ19tMSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbMl0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiB0cnVlLCAic2hhcGUiOiBbOF0sICJzb3VyY2UiOiAiZmFjdG9yZWRfam9pbnRfY29tcG9zaXRpb24iLCAic291cmNlX2tleXMiOiBbIkRfZjAiLCAiRF9mMSJdfSwgIkRfZjAiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEX2YxIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFsyXSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIk1vZGFsaXR5IDA6IHZpc3VhbCBvYnNlcnZhdGlvbiAoMyB2aXN1YWwgY3VlcykiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAib19tMCIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMywgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIk1vZGFsaXR5IDE6IHByb3ByaW9jZXB0aXZlIG9ic2VydmF0aW9uICgyIGJvZHkgc3RhdGVzKSIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAiaW5kZXgiOiAxLCAibmFtZSI6ICJvX20xIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiAyLCAidHlwZSI6ICJmbG9hdCJ9XSwgInN0YXRlX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIkZhY3RvciAwOiBhZ2VudCBsb2NhdGlvbiAoNCBwb3NzaWJsZSBwb3NpdGlvbnMpIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInNfZjAiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJGYWN0b3IgMTogZ29hbCBpZGVudGl0eSAoMiBwb3NzaWJsZSBnb2FscykiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic19mMSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMiwgInR5cGUiOiAiZmxvYXQifV19LCAidmFyaWFibGVzIjogW3siY29tbWVudCI6ICJGYWN0b3IgMDogYWdlbnQgbG9jYXRpb24gKDQgcG9zc2libGUgcG9zaXRpb25zKSIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJzX2YwIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkZhY3RvciAxOiBnb2FsIGlkZW50aXR5ICgyIHBvc3NpYmxlIGdvYWxzKSIsICJkaW1lbnNpb25zIjogWzIsIDFdLCAibmFtZSI6ICJzX2YxIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIk1vZGFsaXR5IDA6IHZpc3VhbCBvYnNlcnZhdGlvbiAoMyB2aXN1YWwgY3VlcykiLCAiZGltZW5zaW9ucyI6IFszLCAxXSwgIm5hbWUiOiAib19tMCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJNb2RhbGl0eSAxOiBwcm9wcmlvY2VwdGl2ZSBvYnNlcnZhdGlvbiAoMiBib2R5IHN0YXRlcykiLCAiZGltZW5zaW9ucyI6IFsyLCAxXSwgIm5hbWUiOiAib19tMSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICIzIHBvc3NpYmxlIGFjdGlvbnM6IHN0YXksIGZvcndhcmQsIGJhY2t3YXJkIiwgImRpbWVuc2lvbnMiOiBbMywgMV0sICJuYW1lIjogInUiLCAidHlwZSI6ICJmbG9hdCJ9XX0="
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
