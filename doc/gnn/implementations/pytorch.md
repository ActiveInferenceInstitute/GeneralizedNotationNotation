# PyTorch Framework Implementation

> **GNN Integration Layer**: Python / GPU-Accelerated Neural Active Inference
> **Framework Base**: `torch >= 2.0` with optional CUDA
> **Simulation Architecture**: Neural network–augmented POMDP agent
> **Documentation Version**: 2.0.0

## Overview

The PyTorch integration enables **neural Active Inference** from GNN specifications. Rather than implementing the classic categorical POMDP (as PyMDP does), the PyTorch renderer builds differentiable generative models where matrices A, B, C, D can be parameterised as learned neural modules.

This enables a continuous learning trajectory: start with hand-specified GNN matrices, optionally replace them with learnable `nn.Parameter` tensors, and train end-to-end.

## Architecture

| Stage | Module | Description |
|---|---|---|
| Rendering (Step 11) | `src/render/pytorch/pytorch_renderer.py` | GNN JSON → PyTorch Agent script |
| Execution (Step 12) | `src/execute/pytorch/pytorch_runner.py` | Subprocess launch, log persistence |
| Analysis (Step 16) | `src/analysis/pytorch/analyzer.py` | Loss curves, belief accuracy, action histograms |

## Model Initialization

GNN matrices are extracted and converted to PyTorch tensors:

```python
import torch
import torch.nn as nn

# Static matrices from GNN spec
A = torch.tensor(gnn_params["A"], dtype=torch.float32)   # [num_obs, num_states]
B = torch.tensor(gnn_params["B"], dtype=torch.float32)   # [num_states, num_states, num_actions]
C = torch.tensor(gnn_params["C"], dtype=torch.float32)   # [num_obs]
D = torch.tensor(gnn_params["D"], dtype=torch.float32)   # [num_states]

# Optionally make learnable
A_param = nn.Parameter(A.log())  # log-space for softmax stability
B_param = nn.Parameter(B.log())
```

## Perception-Action Loop

```python
for t in range(T):
    # Environment: stochastic observation from true state
    obs = torch.multinomial(A[:, true_state], 1).item()

    # Agent inference: softmax belief update
    log_likelihood = A[:, :].log()[:, :].T[obs]     # [num_states]
    log_prior = D.log() if t == 0 else belief.log()
    belief = torch.softmax(log_likelihood + log_prior, dim=0)

    # Policy evaluation: Expected Free Energy
    efe = torch.zeros(num_actions)
    for a in range(num_actions):
        predicted_state = B[:, :, a] @ belief           # [num_states]
        predicted_obs   = A @ predicted_state            # [num_obs]
        ambiguity = -(predicted_obs * predicted_obs.log()).sum()
        risk = (predicted_obs * (predicted_obs.log() - C.log())).sum()
        efe[a] = ambiguity + risk

    # Action selection
    action = torch.multinomial(torch.softmax(-efe, dim=0), 1).item()

    # Environment transition
    true_state = torch.multinomial(B[:, true_state, action], 1).item()
```

## Telemetry Output

```json
{
  "simulation_trace": {
    "observations":  [1, 0, 2, 1, 1],
    "beliefs":       [[0.05, 0.90, 0.05], ...],
    "actions":       [0, 0, 1, 0, 0],
    "efe_history":   [[-1.2, -0.8, -2.1], ...]
  },
  "training": {
    "loss_history":  [],
    "epochs_run":    0,
    "gradient_norm": null
  }
}
```

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

`torch` is an optional dependency — the pipeline gracefully skips PyTorch steps if not installed. Check with:

```bash
PYTHONPATH=src python -c "from execute.pytorch import check_pytorch; print(check_pytorch())"
```

## Run

```bash
# Render GNN to PyTorch script
python src/11_render.py --target-dir input/gnn_files/ --framework pytorch

# Execute PyTorch script
python src/12_execute.py --target-dir input/gnn_files/ --framework pytorch

# Or use the MCP tool
# Call: execute_gnn_model(path="...", framework="pytorch")
```

## Correlation Results

During the **March 6, 2026** pipeline benchmarking audit, the PyTorch integration was verified as **Fully Operational**, achieving a pristine `1.0` correlation baseline against both PyMDP and JAX reference implementations:

| Pair | Correlation |
|---|---|
| PyTorch ↔ PyMDP | 1.0 (deterministic init) |
| PyTorch ↔ JAX | 1.0 (same algorithm) |

*Cross-framework correlation is confirmed when both use the same random seed and identical A/B/C/D matrices.*

## Source Code Connections

| Stage | Module | Key Function |
|---|---|---|
| Rendering | [pytorch_renderer.py](../../../src/render/pytorch/pytorch_renderer.py) | `render_gnn_to_pytorch()` |
| Execution | [pytorch_runner.py](../../../src/execute/pytorch/pytorch_runner.py) | `execute_pytorch_script()` |
| Analysis | [analyzer.py](../../../src/analysis/pytorch/analyzer.py) | `generate_analysis_from_logs()` |

## Improvement Opportunities

| ID | Area | Description | Impact |
|---|---|---|---|
| PT-1 | Rendering | No gradient checkpointing for long sequences | Low |
| PT-2 | Execution | Training loop not yet parameterisable from GNN spec | Medium |
| PT-3 | Analysis | Loss curve visualisation only generated if training runs | Low |

## See Also / Next Steps

- **[Cross-Framework Methodology](../integration/cross_framework_methodology.md)**: Details on the correlation methodology and benchmarking metrics.
- **[Architecture Reference](../reference/architecture_reference.md)**: Deep dive into the pipeline orchestrator and module integration.
- **[GNN Implementations Index](README.md)**: Return to the master framework implementer manifest.
- **[Back to GNN START_HERE](../../START_HERE.md)**
