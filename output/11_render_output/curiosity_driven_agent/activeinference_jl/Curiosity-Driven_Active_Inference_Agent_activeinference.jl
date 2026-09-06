#!/usr/bin/env julia
# ActiveInference.jl discrete POMDP simulation
# Generated from GNN Model: Curiosity-Driven Active Inference Agent

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
const MODEL_NAME = "Curiosity-Driven Active Inference Agent"
const NUM_STATES = 5
const NUM_OBSERVATIONS = 5
const NUM_ACTIONS = 4
const TIME_STEPS = 30
const RANDOM_SEED = 42
const ACTION_PRECISION = 4.0
const B_TENSOR_ORDER = "next_state_previous_state_action"
const GNN_SPEC_JSON_B64 = "eyJjYW5vbmljYWxfcG9tZHBfc2NoZW1hIjogImNhbm9uaWNhbF9wb21kcF92MSIsICJjb25uZWN0aW9ucyI6IFt7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkQiLCAidGFyZ2V0IjogInMifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJzIiwgInRhcmdldCI6ICJBIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAicyIsICJ0YXJnZXQiOiAic19wcmltZSJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogIkEiLCAidGFyZ2V0IjogIm8ifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJDIiwgInRhcmdldCI6ICJHX2lucyJ9LCB7InJlbGF0aW9uIjogIj4iLCAic291cmNlIjogIkdfZXBpIiwgInRhcmdldCI6ICJHIn0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAiR19pbnMiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJcdTAzYjMiLCAidGFyZ2V0IjogIkcifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJFIiwgInRhcmdldCI6ICJcdTAzYzAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJHIiwgInRhcmdldCI6ICJcdTAzYzAifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJcdTAzYzAiLCAidGFyZ2V0IjogInUifSwgeyJyZWxhdGlvbiI6ICI+IiwgInNvdXJjZSI6ICJCIiwgInRhcmdldCI6ICJ1In0sIHsicmVsYXRpb24iOiAiPiIsICJzb3VyY2UiOiAidSIsICJ0YXJnZXQiOiAic19wcmltZSJ9LCB7InJlbGF0aW9uIjogIi0iLCAic291cmNlIjogInMiLCAidGFyZ2V0IjogIkYifSwgeyJyZWxhdGlvbiI6ICItIiwgInNvdXJjZSI6ICJvIiwgInRhcmdldCI6ICJGIn1dLCAiZGVzY3JpcHRpb24iOiAiQW4gQWN0aXZlIEluZmVyZW5jZSBhZ2VudCB3aXRoOlxuLSBFeHBsaWNpdCBlcGlzdGVtaWMgdmFsdWUgKGluZm9ybWF0aW9uIGdhaW4gLyBCYXllc2lhbiBzdXJwcmlzZSkgY29tcG9uZW50IGluIEdcbi0gU2VwYXJhdGUgaW5zdHJ1bWVudGFsIHZhbHVlIChwcmVmZXJlbmNlIHNhdGlzZmFjdGlvbikgY29tcG9uZW50XG4tIFByZWNpc2lvbiBwYXJhbWV0ZXIgXHUwM2IzIHdlaWdodGluZyBlcGlzdGVtaWMgdnMgaW5zdHJ1bWVudGFsIGNvbnRyaWJ1dGlvbnNcbi0gNSBoaWRkZW4gc3RhdGVzLCA1IG9ic2VydmF0aW9ucywgNCBhY3Rpb25zIGluIGEgbmF2aWdhdGlvbiBjb250ZXh0XG4tIEFnZW50IGlzIHJld2FyZGVkIGZvciByZWR1Y2luZyBwb3N0ZXJpb3IgdW5jZXJ0YWludHkiLCAiZ25uX3NlY3Rpb24iOiAiQWN0SW5mUE9NRFAiLCAiaW5pdGlhbF9wYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1swLjksIDAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjksIDAuMDI1LCAwLjAyNSwgMC4wMjVdLCBbMC4wMjUsIDAuMDI1LCAwLjksIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuMDI1LCAwLjksIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuMDI1LCAwLjldXSwgIkIiOiBbW1swLjksIDAuOSwgMS4wLCAwLjldLCBbMC4xLCAwLjAsIDAuMSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4xLCAwLjEsIDAuMCwgMC4wXSwgWzAuOCwgMC45LCAwLjksIDAuOV0sIFswLjEsIDAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuMSwgMC4wLCAwLjBdLCBbMC44LCAwLjksIDAuOSwgMC45XSwgWzAuMSwgMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC4xLCAwLjAsIDAuMF0sIFswLjgsIDAuOSwgMC45LCAwLjldLCBbMC4xLCAwLjAsIDAuMSwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wLCAwLjFdLCBbMC4xLCAwLjEsIDAuMCwgMC4xXSwgWzAuOSwgMS4wLCAwLjksIDEuMF1dXSwgIkMiOiBbLTIuMCwgLTIuMCwgLTIuMCwgLTIuMCwgMi4wXSwgIkQiOiBbMC4yLCAwLjIsIDAuMiwgMC4yLCAwLjJdLCAiRSI6IFswLjI1LCAwLjI1LCAwLjI1LCAwLjI1XSwgIlx1MDNiMyI6IFsxLjBdfSwgImluaXRpYWxwYXJhbWV0ZXJpemF0aW9uIjogeyJBIjogW1swLjksIDAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjksIDAuMDI1LCAwLjAyNSwgMC4wMjVdLCBbMC4wMjUsIDAuMDI1LCAwLjksIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuMDI1LCAwLjksIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuMDI1LCAwLjldXSwgIkIiOiBbW1swLjksIDAuOSwgMS4wLCAwLjldLCBbMC4xLCAwLjAsIDAuMSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4xLCAwLjEsIDAuMCwgMC4wXSwgWzAuOCwgMC45LCAwLjksIDAuOV0sIFswLjEsIDAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjEsIDAuMSwgMC4wLCAwLjBdLCBbMC44LCAwLjksIDAuOSwgMC45XSwgWzAuMSwgMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC4xLCAwLjAsIDAuMF0sIFswLjgsIDAuOSwgMC45LCAwLjldLCBbMC4xLCAwLjAsIDAuMSwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjAsIDAuMV0sIFswLjAsIDAuMCwgMC4wLCAwLjFdLCBbMC4xLCAwLjEsIDAuMCwgMC4xXSwgWzAuOSwgMS4wLCAwLjksIDEuMF1dXSwgIkMiOiBbLTIuMCwgLTIuMCwgLTIuMCwgLTIuMCwgMi4wXSwgIkQiOiBbMC4yLCAwLjIsIDAuMiwgMC4yLCAwLjJdLCAiRSI6IFswLjI1LCAwLjI1LCAwLjI1LCAwLjI1XSwgIlx1MDNiMyI6IFsxLjBdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs1LCA1XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNsYWltZWRfc2xpY2VfY29udmVudGlvbiI6IG51bGwsICJjb250cmFkaWN0aW9uIjogZmFsc2UsICJkZWNsYXJlZF9vcmRlciI6IFsibmV4dF9zdGF0ZSIsICJwcmV2aW91c19zdGF0ZSIsICJhY3Rpb24iXSwgImRlcml2ZWQiOiBmYWxzZSwgImRldGVjdGVkX29yZGVyIjogWyJuZXh0X3N0YXRlIiwgInByZXZpb3VzX3N0YXRlIiwgImFjdGlvbiJdLCAicmVhc29uIjogbnVsbCwgInNoYXBlIjogWzUsIDUsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNV0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzVdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJFIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm1vZGVsX25hbWUiOiAiQ3VyaW9zaXR5LURyaXZlbiBBY3RpdmUgSW5mZXJlbmNlIEFnZW50IiwgIm1vZGVsX3BhcmFtZXRlcnMiOiB7ImJfdGVuc29yX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNvbnRyb2xfZmFjdG9ycyI6IFt7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFs0XSwgImluZGV4IjogMCwgIm5hbWUiOiAiXHUwM2MwIiwgInJvbGUiOiAiYm9va2tlZXBpbmciLCAic2l6ZSI6IDQsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJTZWxlY3RlZCBhY3Rpb24iLCAiZGltZW5zaW9ucyI6IFsxXSwgImluZGV4IjogMSwgIm5hbWUiOiAidSIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogMSwgInR5cGUiOiAiZmxvYXQifV0sICJlcGlzdGVtaWNfd2VpZ2h0IjogMS4wLCAiaW5zdHJ1bWVudGFsX3dlaWdodCI6IDEuMCwgIm51bV9hY3Rpb25zIjogNCwgIm51bV9oaWRkZW5fc3RhdGVzIjogNSwgIm51bV9tb2RhbGl0aWVzIjogMSwgIm51bV9vYnMiOiA1LCAibnVtX3N0YXRlX2ZhY3RvcnMiOiAyLCAibnVtX3RpbWVzdGVwcyI6IDMwLCAib2JzZXJ2YXRpb25fbW9kYWxpdGllcyI6IFt7ImNvbW1lbnQiOiAiQ3VycmVudCBvYnNlcnZhdGlvbiIsICJkaW1lbnNpb25zIjogWzUsIDFdLCAiaW5kZXgiOiAwLCAibmFtZSI6ICJvIiwgInJvbGUiOiAiZmFjdG9yIiwgInNpemUiOiA1LCAidHlwZSI6ICJmbG9hdCJ9XSwgInBhc3NpdmVfbW9kZWwiOiBmYWxzZSwgInNpbXVsYXRpb25fcGFyYW1zIjoge30sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbNSwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInMiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDUsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBiZWxpZWYiLCAiZGltZW5zaW9ucyI6IFs1LCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic19wcmltZSIsICJyb2xlIjogImJvb2trZWVwaW5nIiwgInNpemUiOiA1LCAidHlwZSI6ICJmbG9hdCJ9XX0sICJuYW1lIjogIkN1cmlvc2l0eS1Ecml2ZW4gQWN0aXZlIEluZmVyZW5jZSBBZ2VudCIsICJvbnRvbG9neV9tYXBwaW5nIjogeyJBIjogIkxpa2VsaWhvb2RNYXRyaXgiLCAiQiI6ICJUcmFuc2l0aW9uTWF0cml4IiwgIkMiOiAiTG9nUHJlZmVyZW5jZVZlY3RvciIsICJEIjogIlByaW9yT3ZlckhpZGRlblN0YXRlcyIsICJFIjogIkhhYml0IiwgIkYiOiAiVmFyaWF0aW9uYWxGcmVlRW5lcmd5IiwgIkciOiAiRXhwZWN0ZWRGcmVlRW5lcmd5IiwgIkdfZXBpIjogIkVwaXN0ZW1pY1ZhbHVlIiwgIkdfaW5zIjogIkluc3RydW1lbnRhbFZhbHVlIiwgIm8iOiAiT2JzZXJ2YXRpb24iLCAicyI6ICJIaWRkZW5TdGF0ZSIsICJzX3ByaW1lIjogIk5leHRIaWRkZW5TdGF0ZSIsICJ0IjogIlRpbWUiLCAidSI6ICJBY3Rpb24iLCAiXHUwM2IzIjogIlByZWNpc2lvblBhcmFtZXRlciIsICJcdTAzYzAiOiAiUG9saWN5VmVjdG9yIn0sICJzdHJ1Y3R1cmVkX3BvbWRwIjogeyJhZGFwdGVyX25vdGVzIjogW10sICJjYW5vbmljYWxfYl9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiIsICJjb250cm9sX2ZhY3RvcnMiOiBbeyJjb21tZW50IjogIlBvbGljeSBkaXN0cmlidXRpb24gb3ZlciBhY3Rpb25zIiwgImRpbWVuc2lvbnMiOiBbNF0sICJpbmRleCI6IDAsICJuYW1lIjogIlx1MDNjMCIsICJyb2xlIjogImJvb2trZWVwaW5nIiwgInNpemUiOiA0LCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiU2VsZWN0ZWQgYWN0aW9uIiwgImRpbWVuc2lvbnMiOiBbMV0sICJpbmRleCI6IDEsICJuYW1lIjogInUiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDEsICJ0eXBlIjogImZsb2F0In1dLCAibWF0cmljZXMiOiB7IkEiOiBbWzAuOSwgMC4wMjUsIDAuMDI1LCAwLjAyNSwgMC4wMjVdLCBbMC4wMjUsIDAuOSwgMC4wMjUsIDAuMDI1LCAwLjAyNV0sIFswLjAyNSwgMC4wMjUsIDAuOSwgMC4wMjUsIDAuMDI1XSwgWzAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuOSwgMC4wMjVdLCBbMC4wMjUsIDAuMDI1LCAwLjAyNSwgMC4wMjUsIDAuOV1dLCAiQiI6IFtbWzAuOSwgMC45LCAxLjAsIDAuOV0sIFswLjEsIDAuMCwgMC4xLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdXSwgW1swLjEsIDAuMSwgMC4wLCAwLjBdLCBbMC44LCAwLjksIDAuOSwgMC45XSwgWzAuMSwgMC4wLCAwLjEsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4wLCAwLjAsIDAuMCwgMC4wXV0sIFtbMC4wLCAwLjAsIDAuMCwgMC4wXSwgWzAuMSwgMC4xLCAwLjAsIDAuMF0sIFswLjgsIDAuOSwgMC45LCAwLjldLCBbMC4xLCAwLjAsIDAuMSwgMC4wXSwgWzAuMCwgMC4wLCAwLjAsIDAuMF1dLCBbWzAuMCwgMC4wLCAwLjAsIDAuMF0sIFswLjAsIDAuMCwgMC4wLCAwLjBdLCBbMC4xLCAwLjEsIDAuMCwgMC4wXSwgWzAuOCwgMC45LCAwLjksIDAuOV0sIFswLjEsIDAuMCwgMC4xLCAwLjBdXSwgW1swLjAsIDAuMCwgMC4wLCAwLjFdLCBbMC4wLCAwLjAsIDAuMCwgMC4xXSwgWzAuMCwgMC4wLCAwLjAsIDAuMV0sIFswLjEsIDAuMSwgMC4wLCAwLjFdLCBbMC45LCAxLjAsIDAuOSwgMS4wXV1dLCAiQyI6IFstMi4wLCAtMi4wLCAtMi4wLCAtMi4wLCAyLjBdLCAiRCI6IFswLjIsIDAuMiwgMC4yLCAwLjIsIDAuMl0sICJFIjogWzAuMjUsIDAuMjUsIDAuMjUsIDAuMjVdfSwgIm1hdHJpeF9wcm92ZW5hbmNlIjogeyJBIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs1LCA1XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9LCAiQiI6IHsiY2Fub25pY2FsX29yZGVyIjogIm5leHRfc3RhdGVfcHJldmlvdXNfc3RhdGVfYWN0aW9uIiwgImNsYWltZWRfc2xpY2VfY29udmVudGlvbiI6IG51bGwsICJjb250cmFkaWN0aW9uIjogZmFsc2UsICJkZWNsYXJlZF9vcmRlciI6IFsibmV4dF9zdGF0ZSIsICJwcmV2aW91c19zdGF0ZSIsICJhY3Rpb24iXSwgImRlcml2ZWQiOiBmYWxzZSwgImRldGVjdGVkX29yZGVyIjogWyJuZXh0X3N0YXRlIiwgInByZXZpb3VzX3N0YXRlIiwgImFjdGlvbiJdLCAicmVhc29uIjogbnVsbCwgInNoYXBlIjogWzUsIDUsIDRdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIiwgInNvdXJjZV9vcmRlciI6ICJuZXh0X3N0YXRlX3ByZXZpb3VzX3N0YXRlX2FjdGlvbiJ9LCAiQyI6IHsiZGVyaXZlZCI6IGZhbHNlLCAic2hhcGUiOiBbNV0sICJzb3VyY2UiOiAiSW5pdGlhbFBhcmFtZXRlcml6YXRpb24ifSwgIkQiOiB7ImRlcml2ZWQiOiBmYWxzZSwgInNoYXBlIjogWzVdLCAic291cmNlIjogIkluaXRpYWxQYXJhbWV0ZXJpemF0aW9uIn0sICJFIjogeyJkZXJpdmVkIjogZmFsc2UsICJzaGFwZSI6IFs0XSwgInNvdXJjZSI6ICJJbml0aWFsUGFyYW1ldGVyaXphdGlvbiJ9fSwgIm9ic2VydmF0aW9uX21vZGFsaXRpZXMiOiBbeyJjb21tZW50IjogIkN1cnJlbnQgb2JzZXJ2YXRpb24iLCAiZGltZW5zaW9ucyI6IFs1LCAxXSwgImluZGV4IjogMCwgIm5hbWUiOiAibyIsICJyb2xlIjogImZhY3RvciIsICJzaXplIjogNSwgInR5cGUiOiAiZmxvYXQifV0sICJzdGF0ZV9mYWN0b3JzIjogW3siY29tbWVudCI6ICJIaWRkZW4gc3RhdGUgYmVsaWVmIiwgImRpbWVuc2lvbnMiOiBbNSwgMV0sICJpbmRleCI6IDAsICJuYW1lIjogInMiLCAicm9sZSI6ICJmYWN0b3IiLCAic2l6ZSI6IDUsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBiZWxpZWYiLCAiZGltZW5zaW9ucyI6IFs1LCAxXSwgImluZGV4IjogMSwgIm5hbWUiOiAic19wcmltZSIsICJyb2xlIjogImJvb2trZWVwaW5nIiwgInNpemUiOiA1LCAidHlwZSI6ICJmbG9hdCJ9XX0sICJ2YXJpYWJsZXMiOiBbeyJjb21tZW50IjogIkhpZGRlbiBzdGF0ZSBiZWxpZWYiLCAiZGltZW5zaW9ucyI6IFs1LCAxXSwgIm5hbWUiOiAicyIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJOZXh0IGhpZGRlbiBzdGF0ZSBiZWxpZWYiLCAiZGltZW5zaW9ucyI6IFs1LCAxXSwgIm5hbWUiOiAic19wcmltZSIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJEaXNjcmV0ZSB0aW1lIHN0ZXAiLCAiZGltZW5zaW9ucyI6IFsxXSwgIm5hbWUiOiAidCIsICJ0eXBlIjogImZsb2F0In0sIHsiY29tbWVudCI6ICJDdXJyZW50IG9ic2VydmF0aW9uIiwgImRpbWVuc2lvbnMiOiBbNSwgMV0sICJuYW1lIjogIm8iLCAidHlwZSI6ICJmbG9hdCJ9LCB7ImNvbW1lbnQiOiAiUG9saWN5IGRpc3RyaWJ1dGlvbiBvdmVyIGFjdGlvbnMiLCAiZGltZW5zaW9ucyI6IFs0XSwgIm5hbWUiOiAiXHUwM2MwIiwgInR5cGUiOiAiZmxvYXQifSwgeyJjb21tZW50IjogIlNlbGVjdGVkIGFjdGlvbiIsICJkaW1lbnNpb25zIjogWzFdLCAibmFtZSI6ICJ1IiwgInR5cGUiOiAiZmxvYXQifV19"
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
