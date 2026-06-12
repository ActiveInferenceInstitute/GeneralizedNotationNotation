#!/usr/bin/env julia
# RxInfer.jl discrete POMDP simulation
# Generated from GNN Model: Deep Planning Horizon POMDP
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
const MODEL_NAME = "Deep Planning Horizon POMDP"
const NUM_STATES = 4
const NUM_OBSERVATIONS = 4
const NUM_ACTIONS = 4
const TIME_STEPS = 30
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiLSIsICJzb3VyY2UiOiAiQSIsICJ0YXJnZXQiOiAibyJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogIkYifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJvIiwgInRhcmdldCI6ICJGIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRSIsICJ0YXJnZXQiOiAiXHUwM2MwIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiRyIsICJ0YXJnZXQiOiAiXHUwM2MwIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAic190YXUxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQiIsICJ0YXJnZXQiOiAic190YXUxIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAic190YXUxIiwgInRhcmdldCI6ICJzX3RhdTIifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJzX3RhdTIifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJzX3RhdTIiLCAidGFyZ2V0IjogInNfdGF1MyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkIiLCAidGFyZ2V0IjogInNfdGF1MyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogInNfdGF1MyIsICJ0YXJnZXQiOiAic190YXU0In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiQiIsICJ0YXJnZXQiOiAic190YXU0In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAic190YXU0IiwgInRhcmdldCI6ICJzX3RhdTUifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJBIiwgInRhcmdldCI6ICJzX3RhdTEifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJBIiwgInRhcmdldCI6ICJzX3RhdTIifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJBIiwgInRhcmdldCI6ICJzX3RhdTMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJBIiwgInRhcmdldCI6ICJzX3RhdTQifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJBIiwgInRhcmdldCI6ICJzX3RhdTUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHX3RhdTEifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHX3RhdTIifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHX3RhdTMifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHX3RhdTQifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHX3RhdTUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHX3RhdTEiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHX3RhdTIiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHX3RhdTMiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHX3RhdTQiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHX3RhdTUiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHIiwgInRhcmdldCI6ICJcdTAzYzAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJcdTAzYzAiLCAidGFyZ2V0IjogInUifV0sICJkZXNjcmlwdGlvbiI6ICJBbiBBY3RpdmUgSW5mZXJlbmNlIFBPTURQIHdpdGggZGVlcCAoVD01KSBwbGFubmluZyBob3Jpem9uOlxuLSBFdmFsdWF0ZXMgcG9saWNpZXMgb3ZlciA1IGZ1dHVyZSB0aW1lc3RlcHMgYmVmb3JlIGFjdGluZ1xuLSBVc2VzIHJvbGxvdXQgRXhwZWN0ZWQgRnJlZSBFbmVyZ3kgYWNjdW11bGF0aW9uXG4tIDQgaGlkZGVuIHN0YXRlcywgNCBvYnNlcnZhdGlvbnMsIDQgYWN0aW9uc1xuLSBFYWNoIGFjdGlvbiBwb2xpY3kgaXMgYSBzZXF1ZW5jZSBvZiBUIGFjdGlvbnM6IFx1MDNjMCA9IFthXzEsIGFfMiwgLi4uLCBhX1RdXG4tIEVuYWJsZXMgc29waGlzdGljYXRlZCBtdWx0aS1zdGVwIHJlYXNvbmluZyBhbmQgZGVsYXllZCByZXdhcmQgYXR0cmlidXRpb24iLCAiaW5pdGlhbF9wYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1swLjksIDAuMDUsIDAuMDI1LCAwLjAyNV0sIFswLjA1LCAwLjksIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuOSwgMC4wNV0sIFswLjAyNSwgMC4wMjUsIDAuMDUsIDAuOV1dLCAiQiI6IFtbWzAuOSwgMC45LCAwLjgsIDAuNzAwMDAwMDAwMDAwMDAwMV0sIFswLjAsIDAuMSwgMC4wLCAwLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMCwgMC4wLCAwLjEsIDAuMDk5OTk5OTk5OTk5OTk5OTldLCBbMC4xLCAwLjAsIDAuMSwgMC4xXV0sIFtbMC4xLCAwLjAsIDAuMSwgMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjksIDAuOSwgMC44LCAwLjcwMDAwMDAwMDAwMDAwMDFdLCBbMC4wLCAwLjEsIDAuMCwgMC4wOTk5OTk5OTk5OTk5OTk5OV0sIFswLjAsIDAuMCwgMC4xLCAwLjFdXSwgW1swLjAsIDAuMCwgMC4xLCAwLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMSwgMC4wLCAwLjEsIDAuMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC45LCAwLjksIDAuOCwgMC43XSwgWzAuMCwgMC4xLCAwLjAsIDAuMV1dLCBbWzAuMCwgMC4xLCAwLjAsIDAuMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC4wLCAwLjAsIDAuMSwgMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjEsIDAuMCwgMC4xLCAwLjA5OTk5OTk5OTk5OTk5OTk5XSwgWzAuOSwgMC45LCAwLjgsIDAuN11dXSwgIkMiOiBbLTEuMCwgLTAuNSwgLTAuNSwgMi4wXSwgIkQiOiBbMC4yNSwgMC4yNSwgMC4yNSwgMC4yNV19LCAiaW5pdGlhbHBhcmFtZXRlcml6YXRpb24iOiB7IkEiOiBbWzAuOSwgMC4wNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDUsIDAuOSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC45LCAwLjA1XSwgWzAuMDI1LCAwLjAyNSwgMC4wNSwgMC45XV0sICJCIjogW1tbMC45LCAwLjksIDAuOCwgMC43MDAwMDAwMDAwMDAwMDAxXSwgWzAuMCwgMC4xLCAwLjAsIDAuMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC4wLCAwLjAsIDAuMSwgMC4wOTk5OTk5OTk5OTk5OTk5OV0sIFswLjEsIDAuMCwgMC4xLCAwLjFdXSwgW1swLjEsIDAuMCwgMC4xLCAwLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuOSwgMC45LCAwLjgsIDAuNzAwMDAwMDAwMDAwMDAwMV0sIFswLjAsIDAuMSwgMC4wLCAwLjA5OTk5OTk5OTk5OTk5OTk5XSwgWzAuMCwgMC4wLCAwLjEsIDAuMV1dLCBbWzAuMCwgMC4wLCAwLjEsIDAuMTAwMDAwMDAwMDAwMDAwMDJdLCBbMC4xLCAwLjAsIDAuMSwgMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjksIDAuOSwgMC44LCAwLjddLCBbMC4wLCAwLjEsIDAuMCwgMC4xXV0sIFtbMC4wLCAwLjEsIDAuMCwgMC4xMDAwMDAwMDAwMDAwMDAwMl0sIFswLjAsIDAuMCwgMC4xLCAwLjEwMDAwMDAwMDAwMDAwMDAyXSwgWzAuMSwgMC4wLCAwLjEsIDAuMDk5OTk5OTk5OTk5OTk5OTldLCBbMC45LCAwLjksIDAuOCwgMC43XV1dLCAiQyI6IFstMS4wLCAtMC41LCAtMC41LCAyLjBdLCAiRCI6IFswLjI1LCAwLjI1LCAwLjI1LCAwLjI1XX0sICJtYXRyaXhfcHJvdmVuYW5jZSI6IHsiQSI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkIiOiB7ImNhbm9uaWNhbF9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0LCA0LCA0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiIsICJzb3VyY2Vfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24ifSwgIkMiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJEIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiRGVlcCBQbGFubmluZyBIb3Jpem9uIFBPTURQIiwgIm1vZGVsX3BhcmFtZXRlcnMiOiB7ImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiAob3ZlciBULXN0ZXAgYWN0aW9uIHNlcXVlbmNlcykiLCAiZGltZW5zaW9ucyI6IFs2NF0sICJpbmRleCI6IDEsICJuYW1lIjogIlx1MDNjMCIsICJzaXplIjogNjQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBmaXJzdCBhY3Rpb24gZnJvbSBiZXN0IHBvbGljeSIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAyLCAibmFtZSI6ICJ1IiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm51bV9hY3Rpb25zIjogNCwgIm51bV9oaWRkZW5fc3RhdGVzIjogNCwgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiA0LCAibnVtX3BvbGljaWVzIjogNjQsICJudW1fc3RhdGVfZmFjdG9ycyI6IDYsICJudW1fdGltZXN0ZXBzIjogMzAsICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogIm8iLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dLCAicGFzc2l2ZV9tb2RlbCI6IGZhbHNlLCAicGxhbm5pbmdfaG9yaXpvbiI6IDUsICJzaW11bGF0aW9uX3BhcmFtcyI6IHt9LCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogInMiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmVkaWN0ZWQgc3RhdGUgYXQgdGF1PTEiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMiwgIm5hbWUiOiAic190YXUxIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJlZGljdGVkIHN0YXRlIGF0IHRhdT0yIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInNfdGF1MiIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlByZWRpY3RlZCBzdGF0ZSBhdCB0YXU9MyIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiA0LCAibmFtZSI6ICJzX3RhdTMiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmVkaWN0ZWQgc3RhdGUgYXQgdGF1PTQiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogNSwgIm5hbWUiOiAic190YXU0IiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJlZGljdGVkIHN0YXRlIGF0IHRhdT01IiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDYsICJuYW1lIjogInNfdGF1NSIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifV19LCAibmFtZSI6ICJEZWVwIFBsYW5uaW5nIEhvcml6b24gUE9NRFAiLCAib250b2xvZ3lfbWFwcGluZyI6IHsiQSI6ICJMaWtlbGlob29kTWF0cml4IiwgIkIiOiAiVHJhbnNpdGlvbk1hdHJpeCIsICJDIjogIkxvZ1ByZWZlcmVuY2VWZWN0b3IiLCAiRCI6ICJQcmlvck92ZXJIaWRkZW5TdGF0ZXMiLCAiRSI6ICJQb2xpY3lQcmlvciIsICJGIjogIlZhcmlhdGlvbmFsRnJlZUVuZXJneSIsICJHIjogIkN1bXVsYXRpdmVFeHBlY3RlZEZyZWVFbmVyZ3kiLCAibyI6ICJPYnNlcnZhdGlvbiIsICJzIjogIkhpZGRlblN0YXRlIiwgInQiOiAiVGltZSIsICJ1IjogIkFjdGlvbiIsICJcdTAzYzAiOiAiUG9saWN5U2VxdWVuY2VEaXN0cmlidXRpb24ifSwgInN0cnVjdHVyZWRfcG9tZHAiOiB7ImFkYXB0ZXJfbm90ZXMiOiBbXSwgImNhbm9uaWNhbF9iX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiAob3ZlciBULXN0ZXAgYWN0aW9uIHNlcXVlbmNlcykiLCAiZGltZW5zaW9ucyI6IFs2NF0sICJpbmRleCI6IDEsICJuYW1lIjogIlx1MDNjMCIsICJzaXplIjogNjQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBmaXJzdCBhY3Rpb24gZnJvbSBiZXN0IHBvbGljeSIsICJkaW1lbnNpb25zIjogWzFdLCAiaW5kZXgiOiAyLCAibmFtZSI6ICJ1IiwgInNpemUiOiAxLCAidHlwZSI6ICJmbG9hdCJ9XSwgIm1hdHJpY2VzIjogeyJBIjogW1swLjksIDAuMDUsIDAuMDI1LCAwLjAyNV0sIFswLjA1LCAwLjksIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuOSwgMC4wNV0sIFswLjAyNSwgMC4wMjUsIDAuMDUsIDAuOV1dLCAiQiI6IFtbWzAuOSwgMC4xLCAwLjAsIDAuMF0sIFswLjAsIDAuOSwgMC4xLCAwLjBdLCBbMC4wLCAwLjAsIDAuOSwgMC4xXSwgWzAuMSwgMC4wLCAwLjAsIDAuOV1dLCBbWzAuOSwgMC4wLCAwLjAsIDAuMV0sIFswLjEsIDAuOSwgMC4wLCAwLjBdLCBbMC4wLCAwLjEsIDAuOSwgMC4wXSwgWzAuMCwgMC4wLCAwLjEsIDAuOV1dLCBbWzAuOCwgMC4xLCAwLjEsIDAuMF0sIFswLjAsIDAuOCwgMC4xLCAwLjFdLCBbMC4xLCAwLjAsIDAuOCwgMC4xXSwgWzAuMSwgMC4xLCAwLjAsIDAuOF1dLCBbWzAuNywgMC4xLCAwLjEsIDAuMV0sIFswLjEsIDAuNywgMC4xLCAwLjFdLCBbMC4xLCAwLjEsIDAuNywgMC4xXSwgWzAuMSwgMC4xLCAwLjEsIDAuN11dXSwgIkMiOiBbLTEuMCwgLTAuNSwgLTAuNSwgMi4wXSwgIkQiOiBbMC4yNSwgMC4yNSwgMC4yNSwgMC4yNV19LCAibWF0cml4X3Byb3ZlbmFuY2UiOiB7IkEiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzQsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJCIjogeyJjYW5vbmljYWxfb3JkZXIiOiAibmV4dF9zdGF0ZV9wcmV2aW91c19zdGF0ZV9hY3Rpb24iLCAiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNCwgNCwgNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24iLCAic291cmNlX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIn0sICJDIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiRCI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNF0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifX0sICJvYnNlcnZhdGlvbl9tb2RhbGl0aWVzIjogW3siY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogIm8iLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In1dLCAic3RhdGVfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDEsICJuYW1lIjogInMiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmVkaWN0ZWQgc3RhdGUgYXQgdGF1PTEiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogMiwgIm5hbWUiOiAic190YXUxIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJlZGljdGVkIHN0YXRlIGF0IHRhdT0yIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDMsICJuYW1lIjogInNfdGF1MiIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlByZWRpY3RlZCBzdGF0ZSBhdCB0YXU9MyIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAiaW5kZXgiOiA0LCAibmFtZSI6ICJzX3RhdTMiLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmVkaWN0ZWQgc3RhdGUgYXQgdGF1PTQiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgImluZGV4IjogNSwgIm5hbWUiOiAic190YXU0IiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJlZGljdGVkIHN0YXRlIGF0IHRhdT01IiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJpbmRleCI6IDYsICJuYW1lIjogInNfdGF1NSIsICJzaXplIjogNCwgInR5cGUiOiAiZmxvYXQifV19LCAidmFyaWFibGVzIjogW3siY29tbWVudCI6ICJQcmlvciBvdmVyIGluaXRpYWwgc3RhdGVzIiwgImRpbWVuc2lvbnMiOiBbNF0sICJuYW1lIjogIkQiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ3VycmVudCBoaWRkZW4gc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJuYW1lIjogInMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJlZGljdGVkIHN0YXRlIGF0IHRhdT0xIiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJuYW1lIjogInNfdGF1MSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmVkaWN0ZWQgc3RhdGUgYXQgdGF1PTIiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAic190YXUyIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlByZWRpY3RlZCBzdGF0ZSBhdCB0YXU9MyIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJzX3RhdTMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUHJlZGljdGVkIHN0YXRlIGF0IHRhdT00IiwgImRpbWVuc2lvbnMiOiBbNCwgMV0sICJuYW1lIjogInNfdGF1NCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJQcmVkaWN0ZWQgc3RhdGUgYXQgdGF1PTUiLCAiZGltZW5zaW9ucyI6IFs0LCAxXSwgIm5hbWUiOiAic190YXU1IiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlZhcmlhdGlvbmFsIEZyZWUgRW5lcmd5IGZvciBjdXJyZW50IHN0YXRlIiwgImRpbWVuc2lvbnMiOiBbIlx1MDNjMCJdLCAibmFtZSI6ICJGIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlByZWZlcmVuY2VzIChwZXIgb2JzZXJ2YXRpb24pIiwgImRpbWVuc2lvbnMiOiBbNF0sICJuYW1lIjogIkMiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzQsIDFdLCAibmFtZSI6ICJvIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlRyYW5zaXRpb24gbWF0cml4ICg0IGFjdGlvbnMpIiwgImRpbWVuc2lvbnMiOiBbNCwgNCwgNF0sICJuYW1lIjogIkIiLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiAob3ZlciBULXN0ZXAgYWN0aW9uIHNlcXVlbmNlcykiLCAiZGltZW5zaW9ucyI6IFs2NF0sICJuYW1lIjogIlx1MDNjMCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBmaXJzdCBhY3Rpb24gZnJvbSBiZXN0IHBvbGljeSIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ1IiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIkRpc2NyZXRlIHRpbWUgc3RlcCAoYWN0aW9uIHRpbWVzdGVwKSIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ0IiwgInR5cGUiOiAiZmxvYXQifV19"
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
