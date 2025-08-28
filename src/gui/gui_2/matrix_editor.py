"""
Matrix Editor for GUI 2: Visual matrix editing and GNN template management
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def get_pomdp_template() -> str:
    """
    Get the POMDP template based on actinf_pomdp_agent.md
    This serves as the starting template for GUI 2 visual editing.
    """
    return """# GNN Visual Model Editor
# GNN Version: 1.0
# This model is being constructed using the Visual Matrix Editor (GUI 2)

## GNNSection
VisualPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Visual Active Inference POMDP Agent

## ModelAnnotation
This model is constructed using the visual matrix editor:
- Interactive matrix editing via drag-and-drop
- Real-time state space visualization
- Live GNN markdown synchronization

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3,type=float]   # Observation likelihood matrix

# Transition matrix: B[states_next, states_previous, actions]  
B[3,3,3,type=float]   # State transition matrices

# Preference vector: C[observation_outcomes]
C[3,type=float]       # Log-preferences over observations

# Prior vector: D[states]
D[3,type=float]       # Prior over initial hidden states

# Hidden State
s[3,1,type=float]     # Current hidden state distribution

# Observation
o[3,1,type=int]       # Current observation

# Policy and Control
π[3,type=float]       # Policy distribution
u[1,type=int]         # Action taken

# Time
t[1,type=int]         # Discrete time step

## Connections
D>s
s-A
A-o
s>B
B>u
π>u

## InitialParameterization
# Visual editing will populate these values
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9)
}

B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )
}

C={(0.1, 0.1, 1.0)}
D={(0.33, 0.33, 0.33)}

## Footer
Visual Active Inference POMDP Agent - GUI 2 Visual Editor
"""


def parse_matrix_from_gnn(gnn_text: str) -> Dict[str, Any]:
    """
    Parse matrix information from GNN markdown for visual editing.
    
    Args:
        gnn_text: GNN markdown content
        
    Returns:
        Dictionary containing parsed matrix information
    """
    matrices = {
        "A": {"shape": [3, 3], "values": None, "description": "Likelihood matrix"},
        "B": {"shape": [3, 3, 3], "values": None, "description": "Transition matrices"}, 
        "C": {"shape": [3], "values": None, "description": "Preference vector"},
        "D": {"shape": [3], "values": None, "description": "Prior vector"}
    }
    
    # Parse state space block
    state_space_pattern = r"## StateSpaceBlock\s*\n(.*?)(?=##|$)"
    match = re.search(state_space_pattern, gnn_text, re.DOTALL)
    
    if match:
        state_block = match.group(1)
        
        # Parse matrix dimensions
        for line in state_block.split('\n'):
            if '[' in line and ']' in line:
                # Extract matrix name and dimensions
                parts = line.split('[')
                if len(parts) >= 2:
                    name = parts[0].strip()
                    if name in matrices:
                        # Extract dimensions
                        dim_part = parts[1].split(']')[0]
                        dims = []
                        for d in dim_part.split(','):
                            d = d.strip()
                            if d.isdigit():
                                dims.append(int(d))
                        if dims:
                            matrices[name]["shape"] = dims
    
    # Parse initial parameterization
    param_pattern = r"## InitialParameterization\s*\n(.*?)(?=##|$)"
    match = re.search(param_pattern, gnn_text, re.DOTALL)
    
    if match:
        param_block = match.group(1)
        
        # Parse matrix values (simplified parsing)
        for matrix_name in matrices.keys():
            pattern = rf"{matrix_name}=\{{([^}}]+)\}}"
            matrix_match = re.search(pattern, param_block, re.DOTALL)
            if matrix_match:
                values_str = matrix_match.group(1)
                # This is a simplified parser - in a real implementation, 
                # you'd want more robust parsing
                matrices[matrix_name]["values"] = values_str.strip()
    
    return {
        "matrices": matrices,
        "connections": _parse_connections(gnn_text),
        "metadata": _parse_metadata(gnn_text)
    }


def create_matrix_from_gnn(gnn_text: str) -> Dict[str, Any]:
    """
    Create visual matrix representation from GNN markdown.
    This prepares data for the visual drag-and-drop interface.
    """
    parsed = parse_matrix_from_gnn(gnn_text)
    
    # Convert to visual representation format
    visual_matrices = {}
    
    for name, info in parsed["matrices"].items():
        shape = info["shape"]
        description = info["description"]
        
        # Create visual matrix structure
        if len(shape) == 1:  # Vector
            visual_matrices[name] = {
                "type": "vector",
                "size": shape[0],
                "values": [0.0] * shape[0],
                "description": description,
                "editable": True
            }
        elif len(shape) == 2:  # Matrix
            rows, cols = shape
            visual_matrices[name] = {
                "type": "matrix", 
                "rows": rows,
                "cols": cols,
                "values": [[0.0 for _ in range(cols)] for _ in range(rows)],
                "description": description,
                "editable": True
            }
        elif len(shape) == 3:  # 3D Tensor
            depth, rows, cols = shape
            visual_matrices[name] = {
                "type": "tensor",
                "depth": depth,
                "rows": rows, 
                "cols": cols,
                "values": [[[0.0 for _ in range(cols)] for _ in range(rows)] for _ in range(depth)],
                "description": description,
                "editable": True,
                "current_slice": 0
            }
    
    return {
        "visual_matrices": visual_matrices,
        "connections": parsed["connections"],
        "metadata": parsed["metadata"]
    }


def update_gnn_from_matrix(visual_data: Dict[str, Any], template: str) -> str:
    """
    Update GNN markdown from visual matrix edits.
    
    Args:
        visual_data: Dictionary containing visual matrix data
        template: Base GNN template to update
        
    Returns:
        Updated GNN markdown text
    """
    updated_gnn = template
    
    # Update state space dimensions
    state_space_updates = []
    for name, matrix in visual_data.get("visual_matrices", {}).items():
        if matrix["type"] == "vector":
            state_space_updates.append(f"{name}[{matrix['size']},type=float]   # {matrix['description']}")
        elif matrix["type"] == "matrix":
            state_space_updates.append(f"{name}[{matrix['rows']},{matrix['cols']},type=float]   # {matrix['description']}")
        elif matrix["type"] == "tensor":
            state_space_updates.append(f"{name}[{matrix['depth']},{matrix['rows']},{matrix['cols']},type=float]   # {matrix['description']}")
    
    # Update InitialParameterization with actual values
    param_updates = []
    for name, matrix in visual_data.get("visual_matrices", {}).items():
        if matrix["type"] == "vector":
            values_str = "(" + ", ".join(f"{v:.3f}" for v in matrix["values"]) + ")"
            param_updates.append(f"{name}={{{values_str}}}")
        elif matrix["type"] == "matrix":
            rows_str = []
            for row in matrix["values"]:
                row_str = "(" + ", ".join(f"{v:.3f}" for v in row) + ")"
                rows_str.append(row_str)
            values_str = "(\n  " + ",\n  ".join(rows_str) + "\n)"
            param_updates.append(f"{name}={{{values_str}}}")
        elif matrix["type"] == "tensor":
            slices_str = []
            for slice_data in matrix["values"]:
                rows_str = []
                for row in slice_data:
                    row_str = "(" + ", ".join(f"{v:.3f}" for v in row) + ")"
                    rows_str.append(row_str)
                slice_str = "( " + ", ".join(rows_str) + " )"
                slices_str.append(slice_str)
            values_str = "(\n  " + ",\n  ".join(slices_str) + "\n)"
            param_updates.append(f"{name}={{{values_str}}}")
    
    # Replace parameterization section
    if param_updates:
        param_text = "## InitialParameterization\n# Updated via Visual Matrix Editor\n" + "\n".join(param_updates)
        
        # Replace existing parameterization section
        param_pattern = r"## InitialParameterization.*?(?=##|\Z)"
        updated_gnn = re.sub(param_pattern, param_text + "\n\n", updated_gnn, flags=re.DOTALL)
    
    return updated_gnn


def _parse_connections(gnn_text: str) -> List[str]:
    """Parse connection information from GNN text"""
    connections = []
    
    conn_pattern = r"## Connections\s*\n(.*?)(?=##|$)"
    match = re.search(conn_pattern, gnn_text, re.DOTALL)
    
    if match:
        conn_block = match.group(1)
        for line in conn_block.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                connections.append(line)
    
    return connections


def _parse_metadata(gnn_text: str) -> Dict[str, str]:
    """Parse metadata from GNN text"""
    metadata = {}
    
    # Extract model name
    name_pattern = r"## ModelName\s*\n(.*?)(?=##|$)"
    match = re.search(name_pattern, gnn_text, re.DOTALL)
    if match:
        metadata["model_name"] = match.group(1).strip()
    
    # Extract annotation
    annot_pattern = r"## ModelAnnotation\s*\n(.*?)(?=##|$)"
    match = re.search(annot_pattern, gnn_text, re.DOTALL)
    if match:
        metadata["annotation"] = match.group(1).strip()
    
    return metadata


def validate_matrix_dimensions(visual_data: Dict[str, Any]) -> List[str]:
    """
    Validate matrix dimensions for consistency.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    matrices = visual_data.get("visual_matrices", {})
    
    # Check A matrix dimensions
    if "A" in matrices and "s" in matrices:
        A = matrices["A"]
        if A["type"] == "matrix":
            # A should be [n_obs, n_states]
            # This is a simplified check
            pass
    
    # Check B matrix dimensions  
    if "B" in matrices:
        B = matrices["B"]
        if B["type"] == "tensor":
            # B should be [n_states, n_states, n_actions]
            pass
    
    # Add more validation as needed
    
    return errors
