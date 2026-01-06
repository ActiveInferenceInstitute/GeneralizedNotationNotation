
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from src.render.rxinfer.rxinfer_renderer import render_gnn_to_rxinfer

# Hardcoded GNN spec to verify the renderer
gnn_spec = {
    "name": "Classic Active Inference POMDP Agent v1",
    "model_name": "Classic Active Inference POMDP Agent v1",
    "description": "RxInfer verification",
    "model_parameters": {
        "num_hidden_states": 3,
        "num_obs": 3,
        "num_actions": 3
    }
}

output_path = Path("output/11_render_output/actinf_pomdp_agent/rxinfer/Classic Active Inference POMDP Agent v1_rxinfer.jl")

success, msg, warnings = render_gnn_to_rxinfer(gnn_spec, output_path)
print(f"Success: {success}")
print(f"Message: {msg}")
if warnings:
    print(f"Warnings: {warnings}")
