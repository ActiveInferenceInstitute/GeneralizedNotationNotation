# RXINFER Rendering Results

Generated from GNN POMDP Model: **NEST D03 Synthetic 2-State LGSSM**

## Model Information

- **Model Name**: NEST D03 Synthetic 2-State LGSSM
- **Model Description**: Passive linear-Gaussian state-space model (no control input) generating the
summary indices of the D03 STURM wrapper blanket over T coupling iterations.
- Hidden state x = (renovation_rate_dev, price_pressure), dimensionless deviations.
- Observation y = (price_index, demand_index, stock_index, turnover_index): relative indices of fuel prices p
(sensory, into STURM), final energy demand d (active, out of STURM), the
archetype stock aggregate and the turnover aggregate against fixed reference
values (see the fixture JSON, `blanket.references`).
- q_agg/r_agg are internal aggregates carried in the fixture for posterior
checks only; they never cross the blanket.
- There is NO counterpart to D02's cap: the cap is the Q-13 HIERARCHY imposed
constraint of the MESSAGEix side; STURM receives none.
Deliverable D03 (AII); mirror of the D04 fixture under the D03 blanket.
- **Generation Date**: 2026-09-11 16:50:47

## POMDP Dimensions

- **Number of States**: 2
- **Number of Observations**: 4
- **Number of Actions**: 0

## Active Inference Matrices

### Available Matrices/Vectors:


## Generated Files

- `NEST_D03_Synthetic_2-State_LGSSM_rxinfer.jl` - rxinfer simulation script


## Usage

Refer to the main rxinfer documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: rxinfer
- **File Extension**: .jl
- **Multi-Modality Support**: ❌
- **Multi-Factor Support**: ❌
