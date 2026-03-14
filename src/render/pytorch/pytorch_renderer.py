#!/usr/bin/env python3
"""
PyTorch Renderer for GNN Specifications

Renders GNN POMDP models to standalone PyTorch simulation scripts.
Generates the standard generative loop (same environment dynamics as PyMDP/JAX)
using torch.tensor operations.

@Web: https://pytorch.org/docs/stable/
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def render_gnn_to_pytorch(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, str]:
    """Render a GNN specification to a PyTorch POMDP simulation script.

    Args:
        gnn_spec: Parsed GNN model specification.
        output_path: Path to write the generated PyTorch code.
        options: Optional rendering options.

    Returns:
        Tuple of (success: bool, message: str, output_file_path: str)
    """
    try:
        model_name = gnn_spec.get("modelName", "pytorch_pomdp")
        logger.info(f"Rendering GNN spec to PyTorch: {model_name}")

        # Extract matrices
        A, B, C, D = _extract_matrices(gnn_spec)

        # Validate shapes
        from render.matrix_utils import validate_abcd_shapes
        valid, msg = validate_abcd_shapes(A, B, C, D)
        if not valid:
            logger.warning(f"Shape validation warning: {msg}")

        # Generate code
        code = _generate_pytorch_code(model_name, A, B, C, D, options)

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import os as _os, tempfile as _tempfile
        with _tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output_path.parent, delete=False) as _tmp:
            _tmp.write(code)
        _os.replace(_tmp.name, str(output_path))

        logger.info(f"✅ PyTorch script written to: {output_path}")
        return True, f"PyTorch script generated: {output_path}", str(output_path)

    except Exception as e:
        logger.error(f"❌ PyTorch rendering failed: {e}")
        return False, f"PyTorch rendering failed: {e}", ""


def _extract_matrices(gnn_spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract A, B, C, D matrices from GNN spec."""
    params = gnn_spec.get("stateSpace", {}).get("parameters", {})
    if not params:
        params = gnn_spec.get("initialparameterization", {})
    if not params:
        params = gnn_spec.get("parameters", {})

    def _parse_matrix(raw, default):
        if raw is None:
            return default
        if isinstance(raw, (list, np.ndarray)):
            return np.array(raw, dtype=float)
        if isinstance(raw, str):
            try:
                import ast
                parsed = ast.literal_eval(raw)
                return np.array(parsed, dtype=float)
            except Exception:
                return default
        return default

    num_states = gnn_spec.get("stateSpace", {}).get("size", None)
    if num_states is None:
        num_states = gnn_spec.get("model_parameters", {}).get("num_hidden_states", 2)
    num_obs = gnn_spec.get("observationSpace", {}).get("size", None)
    if num_obs is None:
        num_obs = gnn_spec.get("model_parameters", {}).get("num_obs", num_states)

    default_A = np.eye(num_obs, num_states)
    default_B = np.eye(num_states)
    default_C = np.zeros(num_obs)
    default_C[0] = 1.0
    default_D = np.ones(num_states) / num_states

    A = _parse_matrix(params.get("A"), default_A)
    B = _parse_matrix(params.get("B"), default_B)
    C = _parse_matrix(params.get("C"), default_C)
    D = _parse_matrix(params.get("D"), default_D)

    # Normalize
    from render.matrix_utils import normalize_columns
    A = normalize_columns(A)
    if B.ndim == 2:
        B = normalize_columns(B)

    D = D / D.sum() if D.sum() > 0 else D

    return A, B, C, D


def _format_tensor(arr: np.ndarray, indent: int = 4) -> str:
    """Format a numpy array as a torch.tensor() literal."""
    prefix = " " * indent
    if arr.ndim == 1:
        vals = ", ".join(f"{v:.6f}" for v in arr)
        return f"torch.tensor([{vals}], dtype=torch.float64)"
    elif arr.ndim == 2:
        rows = []
        for row in arr:
            vals = ", ".join(f"{v:.6f}" for v in row)
            rows.append(f"{prefix}    [{vals}]")
        inner = ",\n".join(rows)
        return f"torch.tensor([\n{inner}\n{prefix}], dtype=torch.float64)"
    else:
        return f"torch.tensor({arr.tolist()}, dtype=torch.float64)"


def _generate_pytorch_code(
    model_name: str,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    options: Optional[Dict[str, Any]] = None
) -> str:
    """Generate standalone PyTorch POMDP simulation script."""
    num_timesteps = (options or {}).get("num_timesteps", 10)
    num_states = A.shape[1] if A.ndim == 2 else 2
    num_obs = A.shape[0] if A.ndim == 2 else num_states
    num_actions = B.shape[2] if B.ndim == 3 else 2

    A_str = _format_tensor(A)
    B_str = _format_tensor(B if B.ndim == 2 else B[:, :, 0])
    C_str = _format_tensor(C)
    D_str = _format_tensor(D)

    B_full_init = ""
    if B.ndim == 3:
        slices = []
        for a in range(B.shape[2]):
            slices.append(f"    B_slices.append({_format_tensor(B[:, :, a], indent=4)})")
        B_full_init = "\n    B_slices = []\n" + "\n".join(slices) + "\n    B = torch.stack(B_slices, dim=2)"
    else:
        B_full_init = f"\n    B = {B_str}.unsqueeze(2).expand(-1, -1, {num_actions}).clone()"

    code = f'''\
#!/usr/bin/env python3
"""
PyTorch POMDP Simulation: {model_name}

Auto-generated by GNN Pipeline — PyTorch renderer.
Implements the standard Active Inference generative loop using PyTorch tensors.
"""
import json
import os
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Install with: uv sync --extra ml-ai")
    sys.exit(1)

import numpy as np


def run_simulation():
    """Run POMDP simulation with PyTorch tensors."""
    start_time = time.time()

    # --- Model Parameters ---
    num_states = {num_states}
    num_obs = {num_obs}
    num_actions = {num_actions}
    T = {num_timesteps}

    A = {A_str}
    C = {C_str}
    D = {D_str}
    {B_full_init}

    # --- Simulation State ---
    beliefs_history = []
    actions_history = []
    observations_history = []
    efe_history = []

    # Initialize true state from prior
    true_state = torch.multinomial(D, 1).item()
    beliefs = D.clone()

    for t in range(T):
        # 1. Generate observation from true state
        obs_probs = A[:, true_state]
        obs_probs = obs_probs / obs_probs.sum()
        observation = torch.multinomial(obs_probs, 1).item()

        # 2. Belief update (Bayesian filtering)
        likelihood = A[observation, :]
        posterior = likelihood * beliefs
        posterior = posterior / (posterior.sum() + 1e-16)
        beliefs = posterior

        # 3. Compute Expected Free Energy (EFE) for each action
        efe = torch.zeros(num_actions, dtype=torch.float64)
        for a_idx in range(num_actions):
            B_a = B[:, :, a_idx]
            predicted_state = B_a @ beliefs
            predicted_obs = A @ predicted_state

            # Ambiguity (expected entropy of observations)
            log_A = torch.log(A + 1e-16)
            ambiguity = -(predicted_state * (A * log_A).sum(dim=0)).sum()

            # Risk (KL from preferred observations)
            log_pred = torch.log(predicted_obs + 1e-16)
            C_norm = torch.softmax(C, dim=0)
            log_pref = torch.log(C_norm + 1e-16)
            risk = (predicted_obs * (log_pred - log_pref)).sum()

            efe[a_idx] = ambiguity + risk

        # 4. Action selection (softmax over negative EFE)
        action_probs = torch.softmax(-efe, dim=0)
        action = torch.multinomial(action_probs, 1).item()

        # 5. State transition
        B_a = B[:, :, action]
        transition_probs = B_a[:, true_state]
        transition_probs = transition_probs / (transition_probs.sum() + 1e-16)
        true_state = torch.multinomial(transition_probs, 1).item()

        # Record history
        beliefs_history.append(beliefs.cpu().numpy().tolist())
        actions_history.append(action)
        observations_history.append(observation)
        efe_history.append(efe.cpu().numpy().tolist())

    elapsed = time.time() - start_time

    # --- Validation ---
    beliefs_arr = np.array(beliefs_history)
    validation = {{
        "beliefs_in_range": bool(np.all((beliefs_arr >= 0) & (beliefs_arr <= 1))),
        "beliefs_sum_to_one": bool(np.allclose(beliefs_arr.sum(axis=1), 1.0, atol=1e-6)),
        "actions_in_range": all(0 <= a < num_actions for a in actions_history),
        "all_valid": True
    }}
    validation["all_valid"] = all(validation.values())

    results = {{
        "model_name": "{model_name}",
        "framework": "pytorch",
        "num_timesteps": T,
        "num_states": num_states,
        "num_observations": num_obs,
        "num_actions": num_actions,
        "beliefs": beliefs_history,
        "actions": actions_history,
        "observations": observations_history,
        "efe_history": efe_history,
        "validation": validation,
        "execution_time_seconds": round(elapsed, 4),
        "torch_version": torch.__version__
    }}

    # Save results
    output_dir = Path(os.environ.get("PYTORCH_OUTPUT_DIR", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "simulation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"PyTorch POMDP simulation complete: {{T}} timesteps in {{elapsed:.3f}}s")
    print(f"Results saved to: {{output_file}}")
    print(f"Validation: {{validation}}")
    return results


if __name__ == "__main__":
    run_simulation()
'''
    return code
