# GNN to RxInfer.jl Validation

This directory contains scripts and resources for validating the pipeline that translates Generalized Notation Notation (GNN) models into runnable RxInfer.jl simulations.

## `Multiagent_GNN_RxInfer.jl`

This is the primary validation script. Its purpose is to demonstrate that a configuration file (`config.toml`) generated from a GNN specification can successfully configure and run a complex RxInfer.jl simulation.

### Workflow

The script executes a two-stage process:

1.  **Baseline Simulation:**
    - It first locates the standard "Multi-agent Trajectory Planning" example within the `RxInferExamples.jl` package.
    - It runs this example using its **original, hand-written `config.toml` file**.
    - This step establishes a baseline for a successful run and its expected outputs.

2.  **GNN-Configured Simulation:**
    - The script then creates a new directory for the validation run.
    - It copies all the Julia script files from the original example into this new directory.
    - Crucially, it **discards the original `config.toml`** and instead copies the GNN-generated `rxinfer_multiagent_gnn_config.toml` into its place, renaming it to `config.toml`.
    - It then executes the simulation from this new directory, now driven entirely by the GNN-derived configuration.

### Purpose of Validation

Successfully completing both runs serves as a proof-of-concept, demonstrating that:
- The GNN parser and renderer can produce a syntactically correct and complete `config.toml` file for RxInfer.jl.
- The parameters specified in the GNN model are correctly translated into values that the RxInfer.jl simulation can understand and use.
- The end-to-end pipeline from GNN model -> TOML file -> RxInfer.jl simulation is functional.

### How to Run

Ensure that you have a Julia environment with all the necessary packages installed (as specified within the script and the RxInfer example). Then, execute the script from the repository root or its directory:

```sh
julia doc/rxinfer/Multiagent_GNN_RxInfer.jl
```

The script will log its progress to the console and produce output directories for both the original and the GNN-configured simulations, allowing for a direct comparison of results. 