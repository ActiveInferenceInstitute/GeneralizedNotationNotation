# RXINFER Rendering Results

Generated from GNN POMDP Model: **NEST D04 Synthetic 2-State LGSSM**

## Model Information

- **Model Name**: NEST D04 Synthetic 2-State LGSSM
- **Model Description**: Passive linear-Gaussian state-space model (no control input) generating the
summary indices of the D02 MESSAGEix wrapper blanket over T coupling iterations.
- Hidden state x = (decarb_rate_dev, demand_pressure), dimensionless deviations.
- Observation y = (emissions_index, price_index, objective_index, demand_index): relative indices of total
emissions e, mean commodity price p, objective J and mean demand d against
fixed reference values (see the fixture JSON, `blanket.references`).
- cap (emissions cap) is a declared exogenous schedule in the fixture, not a
variable of this model.
Deliverable D04 (AII); pattern for D07 (3–5 states) and D14 (GNN → RxInfer render).
- **Generation Date**: 2026-09-11 16:02:00

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 4
- **Number of Actions**: 0

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `NEST_D04_Synthetic_2-State_LGSSM_rxinfer.jl` - rxinfer simulation script


## Usage

Refer to the main rxinfer documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: rxinfer
- **File Extension**: .jl
- **Multi-Modality Support**: ❌
- **Multi-Factor Support**: ❌
