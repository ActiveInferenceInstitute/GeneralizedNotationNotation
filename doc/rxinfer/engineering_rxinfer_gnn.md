# Engineering RxInfer.jl with Generalized Notation Notation (GNN)

## Overview

This document details the process of using Generalized Notation Notation (GNN) to specify, generate, and validate a configuration for an `RxInfer.jl` model. We use the "Multi-agent Trajectory Planning" example from `RxInferExamples.jl` to demonstrate a full round-trip engineering use case.

The core objective is to show that a model specified in GNN can produce a configuration file that is a drop-in replacement for the original, human-written configuration. This validates the GNN-to-RxInfer rendering pipeline and demonstrates a powerful workflow for model management and deployment.

## The Workflow

The process involves three main components:
1.  **The Original RxInfer Example**: A pre-existing, functional Julia project that serves as our baseline.
2.  **The GNN-Rendered Configuration**: A `config.toml` file generated from a formal GNN specification of the model.
3.  **The Validation Script**: A Julia script that automates the testing process to ensure the GNN-generated config works identically to the original.

### 1. The Original RxInfer Example

The base for our demonstration is the "Multi-agent Trajectory Planning" example. The original, unmodified source code for this example is located in the repository at:

```
doc/rxinfer/RxInferExamples.jl/scripts/Advanced Examples/Multi-agent Trajectory Planning/
```

This directory contains the Julia source code and an original `config.toml` which defines the model parameters, simulation settings, and environment.

### 2. The GNN-Rendered Configuration

Using the GNN toolchain, a formal GNN model description is created to represent the multi-agent trajectory model. The GNN renderer then processes this file to produce a compatible `config.toml`. For this example, the GNN-generated configuration is located at:

```
output/gnn_rendered_simulators/rxinfer_toml/rxinfer_multiagent_gnn/Multi-agent Trajectory Planning_config.toml
```

The goal is to prove that this file is functionally identical to the original `config.toml`.

### 3. The Validation Script: `Multiagent_GNN_RxInfer.jl`

To automate the validation, we use the `doc/rxinfer/Multiagent_GNN_RxInfer.jl` script. This script performs the following steps, logging its progress to the console:

1.  **Check Paths**: Verifies that the original RxInfer example script and the GNN-generated configuration file exist at their expected locations.
2.  **Run Original Script**: Executes the original "Multi-agent Trajectory Planning" script using its own `config.toml`. This serves as a baseline to ensure the original example runs correctly.
3.  **Validate GNN Config**: Loads and performs a basic validation of the GNN-generated `config.toml` to ensure it's well-formed and contains the expected sections.
4.  **Create a Test Environment**:
    *   Creates a new directory: `doc/rxinfer/multiagent_trajectory_planning/`.
    *   Copies all files from the original example directory into this new directory, *except* for the original `config.toml`.
5.  **Inject GNN Config**: Copies the GNN-generated `config.toml` into the new test directory. At this point, the test directory contains the original source code but with the GNN-generated configuration.
6.  **Run Modified Script**: Executes the Julia script from the test directory using the GNN configuration.

Successful execution of this final step demonstrates that the GNN-generated configuration is a valid, drop-in replacement for the original, thus verifying the integrity of the GNN-to-RxInfer pipeline for this use case.

## How to Run the Validation

To run the entire validation process, execute the main Julia script from the repository root:

```bash
julia doc/rxinfer/Multiagent_GNN_RxInfer.jl
```

The script will print logs to the console, showing the outcome of each step. A successful run will complete with an exit code of 0, confirming that the GNN-produced configuration works as expected.

## Conclusion

This engineering use case demonstrates a key capability of the GNN ecosystem: the ability to formally specify a complex model and reliably generate target-specific, executable configuration files. This workflow promotes reproducibility, modularity, and enables a robust, programmatic toolchain for managing and deploying scientific models across different simulation environments like `RxInfer.jl`.

There are connections to CEREBRUM and other topics, via GNN as well as directly/separately. 