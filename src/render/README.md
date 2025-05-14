# Render Step

This step in the GNN processing pipeline is responsible for taking a GNN (Generalized Notation Notation) specification and rendering it into executable formats suitable for various simulation and modeling environments.

## Purpose

The primary goal of the `render` step is to translate the abstract GNN specification into concrete, runnable code or configuration files for target platforms. This allows the same GNN model to be executed in different environments, facilitating tasks such as:

- Single simulation runs
- Parameter sweeps
- Deployment to different computational backends

## Supported Target Formats

Initially, this step will focus on rendering GNN specifications for:

- **PyMDP (Python):** For active inference simulations in Python.
- **RxInfer.jl (Julia):** For probabilistic programming and message passing in Julia.

Further target formats may be added in the future. 