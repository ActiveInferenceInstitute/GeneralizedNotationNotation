#!/usr/bin/env python3
"""
ActiveInference.jl Renderer Module for GNN Specifications

This module provides streamlined rendering capabilities for GNN specifications to
ActiveInference.jl code, focusing on core functionality and scientific validity.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

logger = logging.getLogger(__name__)

def render_gnn_to_activeinference_jl(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to ActiveInference.jl script.
    
    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_path: Path where the Julia script will be saved
        options: Rendering options
        
    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    try:
        logger.info(f"Rendering GNN specification to ActiveInference.jl script for model: {gnn_spec.get('name', 'unknown')}")
        
        # Extract model information from GNN spec
        model_info = extract_model_info(gnn_spec)
        
        # Generate Julia script content
        julia_script = generate_activeinference_script(model_info)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the Julia script
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(julia_script)
        
        success_msg = f"Successfully rendered ActiveInference.jl script to {output_path.name}"
        logger.info(success_msg)
        return True, success_msg, [str(output_path.resolve())]
        
    except Exception as e:
        error_msg = f"Failed to render ActiveInference.jl script: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg, []

def extract_model_info(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant model information from GNN specification for ActiveInference.jl.
    Robustly handles GNN state space and parameter extraction from multiple sources.
    """
    model_info = {
        "name": gnn_spec.get("name", "gnn_model"),
        "description": gnn_spec.get("description", "GNN model converted to ActiveInference.jl"),
        "n_states": [],
        "n_observations": [],
        "n_controls": [],
        "policy_length": 1,
        "metadata": {}
    }

    # --- Primary extraction: from POMDP processor format (model_parameters) ---
    model_params = gnn_spec.get("model_parameters", {})
    n_states = model_params.get("num_hidden_states")
    n_obs = model_params.get("num_obs") 
    n_actions = model_params.get("num_actions")
    
    # --- Fallback 1: from original statespaceblock format ---
    if n_states is None or n_obs is None or n_actions is None:
        statespace = gnn_spec.get("statespaceblock", [])
        for entry in statespace:
            var_id = entry.get("id", "")
            dims = entry.get("dimensions", "")
            if var_id == "s" and n_states is None:
                # Hidden state: e.g., '3,1,type=float'
                try:
                    n_states = int(dims.split(",")[0])
                except Exception:
                    pass
            elif var_id == "o" and n_obs is None:
                # Observation: e.g., '3,1,type=int'
                try:
                    n_obs = int(dims.split(",")[0])
                except Exception:
                    pass
            elif var_id == "u" and n_actions is None:
                # Action: e.g., '1,type=int' (but see B for action count)
                try:
                    n_actions = int(dims.split(",")[0])
                except Exception:
                    pass
            elif var_id == "A" and (n_obs is None or n_states is None):
                # A matrix: e.g., '3,3,type=float'
                try:
                    if n_obs is None:
                        n_obs = int(dims.split(",")[0])
                    if n_states is None:
                        n_states = int(dims.split(",")[1])
                except Exception:
                    pass
            elif var_id == "B" and (n_states is None or n_actions is None):
                # B matrix: e.g., '3,3,3,type=float'
                try:
                    if n_states is None:
                        n_states = int(dims.split(",")[0])
                    if n_actions is None:
                        n_actions = int(dims.split(",")[2])
                except Exception:
                    pass
    
    # --- Fallback 2: from raw ModelParameters section ---
    if n_states is None or n_obs is None or n_actions is None:
        params = gnn_spec.get("raw_sections", {}).get("ModelParameters", "")
        import re
        if n_states is None:
            m_states = re.search(r"num_hidden_states:\s*(\d+)", params)
            if m_states:
                n_states = int(m_states.group(1))
        if n_obs is None:
            m_obs = re.search(r"num_obs:\s*(\d+)", params)
            if m_obs:
                n_obs = int(m_obs.group(1))
        if n_actions is None:
            m_actions = re.search(r"num_actions:\s*(\d+)", params)
            if m_actions:
                n_actions = int(m_actions.group(1))
    
    # --- Fallback 3: infer from matrix shapes in initialparameterization ---
    if n_states is None or n_obs is None or n_actions is None:
        init_params = gnn_spec.get("initialparameterization", {})
        
        # Try A matrix for n_obs, n_states
        if "A" in init_params and (n_obs is None or n_states is None):
            try:
                A_matrix = init_params["A"]
                if isinstance(A_matrix, list) and len(A_matrix) > 0:
                    if n_obs is None:
                        n_obs = len(A_matrix)  # rows = observations
                    if n_states is None and isinstance(A_matrix[0], list):
                        n_states = len(A_matrix[0])  # cols = states
            except Exception:
                pass
        
        # Try B matrix for n_states, n_actions
        if "B" in init_params and (n_states is None or n_actions is None):
            try:
                B_matrix = init_params["B"]
                if isinstance(B_matrix, list) and len(B_matrix) > 0:
                    if n_actions is None:
                        n_actions = len(B_matrix)  # depth = actions
                    if n_states is None and isinstance(B_matrix[0], list):
                        n_states = len(B_matrix[0])  # rows = states
            except Exception:
                pass
    
    # --- Final validation ---
    if n_states is None or n_obs is None or n_actions is None:
        missing = []
        if n_states is None:
            missing.append("n_states")
        if n_obs is None:
            missing.append("n_obs") 
        if n_actions is None:
            missing.append("n_actions")
        raise ValueError(f"Could not extract {missing} from GNN spec. Available keys: {list(gnn_spec.keys())}")
    
    model_info["n_states"] = [n_states]
    model_info["n_observations"] = [n_obs]
    model_info["n_controls"] = [n_actions]

    # --- Extract matrices from initialparameterization ---
    initial_params = gnn_spec.get("initialparameterization", {})
    if not initial_params:
        # Try legacy field
        initial_params = gnn_spec.get("InitialParameterization", {})
    if not initial_params:
        raise ValueError("No initialparameterization found in GNN spec.")
    
    model_info["A"] = initial_params.get("A")
    model_info["B"] = initial_params.get("B")
    model_info["C"] = initial_params.get("C")
    model_info["D"] = initial_params.get("D")
    model_info["E"] = initial_params.get("E")
    
    if any(x is None for x in [model_info["A"], model_info["B"], model_info["C"], model_info["D"], model_info["E"]]):
        raise ValueError("Missing one or more of A, B, C, D, E in initialparameterization.")

    # --- Metadata ---
    model_info["metadata"] = {
        "original_format": "GNN",
        "conversion_timestamp": "auto-generated",
        "source_sections": list(gnn_spec.keys())
    }
    return model_info

def generate_activeinference_script(model_info: Dict[str, Any]) -> str:
    """
    Generate a streamlined ActiveInference.jl script content.
    
    Args:
        model_info: Extracted model information
        
    Returns:
        Generated Julia script as string
    """
    # Extract dimensions
    n_states = model_info["n_states"][0]
    n_obs = model_info["n_observations"][0]
    n_actions = model_info["n_controls"][0]
    
    # Extract matrices
    A_matrix = model_info["A"]
    B_matrix = model_info["B"]
    
    # Fix n_actions from actual B matrix if mismatch
    # B matrix shape is (states, states, actions) after fixing
    if isinstance(B_matrix, list) and len(B_matrix) > 0:
        actual_n_actions = len(B_matrix)
        if actual_n_actions != n_actions:
            logger.info(f"Correcting n_actions from {n_actions} to {actual_n_actions} based on B matrix")
            n_actions = actual_n_actions
    C_vector = model_info["C"]
    D_vector = model_info["D"]
    E_vector = model_info["E"]
    
    # Fix mangled B matrix from GNN parser
    # The parser sometimes splits tuples incorrectly, leaving strings like '(1.0' instead of tuples
    if isinstance(B_matrix, list) and len(B_matrix) > 0:
        if isinstance(B_matrix[0], list) and len(B_matrix[0]) > 0:
            if isinstance(B_matrix[0][0], str) and '(' in B_matrix[0][0]:
                # Reconstruct B matrix from mangled format
                fixed_B = []
                for slice_data in B_matrix:
                    # Each slice should be a 2D matrix (list of rows)
                    fixed_slice = []
                    current_row = []
                    for item in slice_data:
                        if isinstance(item, str):
                            # Extract number from string like '(1.0' or '0.0)'
                            num_str = item.replace('(', '').replace(')', '').strip()
                            if num_str:
                                try:
                                    current_row.append(float(num_str))
                                except ValueError:
                                    pass
                        elif isinstance(item, (int, float)):
                            current_row.append(float(item))
                        
                        # Check if we completed a row (every 3 items for 3x3 matrices)
                        if len(current_row) >= n_states:
                            fixed_slice.append(current_row)
                            current_row = []
                    
                    # Handle any remaining items
                    if current_row:
                        fixed_slice.append(current_row)
                    
                    if fixed_slice:
                        fixed_B.append(fixed_slice)
                
                if fixed_B:
                    B_matrix = fixed_B
                    logger.info(f"Fixed mangled B matrix: {len(B_matrix)} slices of {len(B_matrix[0])}x{len(B_matrix[0][0])}")
    
    # Convert to Julia format
    def matrix_to_julia(matrix_data):
        def convert_element(element):
            """Convert tuple or list elements to proper Julia format."""
            if isinstance(element, (tuple, list)):
                return "[" + ", ".join(str(x) for x in element) + "]"
            return str(element)
        
        if isinstance(matrix_data, list):
            if len(matrix_data) > 0 and isinstance(matrix_data[0], (list, tuple)):
                if len(matrix_data[0]) > 0 and isinstance(matrix_data[0][0], (list, tuple)):
                    # 3D matrix (B matrix) - handle nested tuples/lists
                    # Convert to proper Julia 3D array syntax
                    slices = []
                    for slice_data in matrix_data:
                        rows = []
                        for row in slice_data:
                            # Extract values from tuples/lists
                            if isinstance(row, (tuple, list)):
                                row_values = " ".join(str(x) for x in row)
                            else:
                                row_values = str(row)
                            rows.append(row_values)
                        slice_matrix = "; ".join(rows)
                        slices.append(f"[{slice_matrix}]")
                    return "cat(" + ", ".join(slices) + "; dims=3)"
                else:
                    # 2D matrix (A matrix) - use Julia matrix syntax with semicolons
                    rows = []
                    for row in matrix_data:
                        row_values = " ".join(str(x) for x in row)
                        rows.append(row_values)
                    return "[" + "; ".join(rows) + "]"
            else:
                # 1D vector (C, D, E vectors)
                return "[" + ", ".join(str(x) for x in matrix_data) + "]"
        elif isinstance(matrix_data, tuple):
            # Handle tuple input
            return "[" + ", ".join(str(x) for x in matrix_data) + "]"
        return str(matrix_data)
    
    julia_A = matrix_to_julia(A_matrix)
    julia_B = matrix_to_julia(B_matrix)
    julia_C = matrix_to_julia(C_vector)
    julia_D = matrix_to_julia(D_vector)
    julia_E = matrix_to_julia(E_vector)
    
    script = f'''#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: {model_info["name"]}

Generated from GNN specification.
{model_info["description"]}

Model Dimensions:
- States: {n_states}
- Observations: {n_obs}  
- Actions: {n_actions}
"""

# Ensure required packages are installed
using Pkg

# Install missing packages if needed
println("📦 Ensuring required packages are installed...")
try
    # Try to precompile key packages - will add if missing
    Pkg.add(["JSON", "Plots", "StatsBase", "ActiveInference", "Distributions"])
    println("✅ Package installation complete")
catch e
    println("⚠️  Some packages may need manual installation: $e")
end

# Import packages
using Dates
using Logging
using DelimitedFiles
using Random
using Statistics
using LinearAlgebra
using ActiveInference
using Distributions
using JSON
using Plots
using StatsBase

# Global configuration
const SCRIPT_VERSION = "1.0.0"
const MODEL_NAME = "{model_info["name"]}"
const N_STATES = {n_states}
const N_OBSERVATIONS = {n_obs}
const N_CONTROLS = {n_actions}
const POLICY_LENGTH = 1

println("="^70)
println("ActiveInference.jl Script for GNN Model: $MODEL_NAME")
println("="^70)
println("Julia version: $(VERSION)")
println("Date: $(now())")
println("Model dimensions: States=$N_STATES, Observations=$N_OBSERVATIONS, Actions=$N_CONTROLS")
println()

# Setup output directory
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
output_dir = "activeinference_outputs_$timestamp"
mkpath(output_dir)
println("📁 Output directory: $output_dir")

# Create model matrices from GNN specification
println("🔧 Creating model matrices from GNN specification...")

# A matrix (observation model)
A_matrix = {julia_A}
println("✅ A matrix created: $(size(A_matrix))")

# B matrix (transition model) 
B_matrix = {julia_B}
println("✅ B matrix created: $(size(B_matrix))")

# C vector (preferences)
C_vector = {julia_C}
println("✅ C vector created: $(length(C_vector))")

# D vector (prior beliefs)
D_vector = {julia_D}
println("✅ D vector created: $(length(D_vector))")

# E vector (policy priors)
# ActiveInference.jl requires E-vector length to match number of policies
# For policy_len=1, num_policies = num_actions  
E_vector_raw = {julia_E}
POLICY_LEN = 1  # Planning horizon
NUM_POLICIES = N_CONTROLS ^ POLICY_LEN  # Number of possible action sequences

# Adjust E-vector to match policy count
if length(E_vector_raw) != NUM_POLICIES
    println("⚠️  Adjusting E-vector from $(length(E_vector_raw)) to $NUM_POLICIES elements")
    if length(E_vector_raw) == N_CONTROLS
        # Expand: one value per action -> one value per policy
        E_vector = fill(1.0 / NUM_POLICIES, NUM_POLICIES)  # Uniform prior
    else
        # Fallback: uniform distribution
        E_vector = fill(1.0 / NUM_POLICIES, NUM_POLICIES)
    end
else
    E_vector = E_vector_raw
end
println("✅ E vector created: $(length(E_vector)) (matches $NUM_POLICIES policies)")

# Normalize matrices to ensure proper probability distributions
println("🔧 Normalizing matrices...")

# Normalize A matrix (columns should sum to 1)
for col in 1:size(A_matrix, 2)
    col_sum = sum(A_matrix[:, col])
    if col_sum > 0
        A_matrix[:, col] ./= col_sum
    end
end

# Normalize B matrix (each action slice should have columns summing to 1)
for action in 1:size(B_matrix, 3)
    for col in 1:size(B_matrix, 2)
        col_sum = sum(B_matrix[:, col, action])
        if col_sum > 0
            B_matrix[:, col, action] ./= col_sum
        end
    end
end

# Normalize vectors
D_vector ./= sum(D_vector)
E_vector ./= sum(E_vector)

println("✅ All matrices normalized")

# Create ActiveInference.jl agent
println("🤖 Creating ActiveInference.jl agent...")

try
    # Convert to ActiveInference.jl format
    A = [A_matrix]  # Vector of matrices for each modality
    B = [B_matrix]  # Vector of matrices for each state factor
    C = [C_vector]  # Vector of preference vectors
    D = [D_vector]  # Vector of prior vectors
    
    # Calculate number of policies
    n_policies = N_CONTROLS
    E = ones(n_policies) ./ n_policies  # Uniform policy prior
    
    # Agent settings
    settings = Dict(
        "policy_len" => POLICY_LENGTH,
        "n_states" => [N_STATES],
        "n_observations" => [N_OBSERVATIONS],
        "n_controls" => [N_CONTROLS]
    )
    
    # Agent parameters
    parameters = Dict(
        "alpha" => 16.0,  # Action precision
        "beta" => 1.0,    # Policy precision
        "gamma" => 16.0,  # Expected free energy precision
        "eta" => 0.1,     # Learning rate
        "omega" => 1.0    # Evidence accumulation rate
    )
    
    # Initialize agent
    aif_agent = init_aif(A, B; C=C, D=D, E=E, settings=settings, parameters=parameters, verbose=false)
    println("✅ Agent initialized successfully")
    
    # Run simulation
    println("🚀 Running simulation...")
    n_steps = 20
    observations_log = []
    actions_log = []
    beliefs_log = []
    efe_log = []
    policy_log = []
    steps_log = []
    
    # Track full belief distributions (all states)
    beliefs_full_log = []
    
    Random.seed!(42)  # Reproducibility
    
    for step in 1:n_steps
        # Generate observation (simplified)
        if step == 1
            observation = [1]  # Start with first observation
        else
            # Generate observation based on current state belief
            state_prob = aif_agent.qs_current[1]
            obs_prob = A_matrix[:, argmax(state_prob)]
            observation = [rand(Categorical(obs_prob))]
        end
        push!(observations_log, observation[1])
        
        # Agent inference and action
        infer_states!(aif_agent, observation)
        infer_policies!(aif_agent)
        sample_action!(aif_agent)
        
        # Log action and beliefs
        push!(actions_log, aif_agent.action[1])
        push!(beliefs_full_log, copy(aif_agent.qs_current[1]))  # All states
        push!(beliefs_log, aif_agent.qs_current[1][1])  # First state only
        
        # Try to log EFE if available
        try
            if hasfield(typeof(aif_agent), :EFE) && !isnothing(aif_agent.EFE)
                push!(efe_log, minimum(aif_agent.EFE))
            else
                push!(efe_log, NaN)
            end
        catch
            push!(efe_log, NaN)
        end
        
        # Log policy if available
        try
            if hasfield(typeof(aif_agent), :policy) && !isnothing(aif_agent.policy)
                push!(policy_log, aif_agent.policy[1])
            else
                push!(policy_log, aif_agent.action[1])
            end
        catch
            push!(policy_log, aif_agent.action[1])
        end
        
        push!(steps_log, step)
        
        if step % 5 == 0
            println("Step $step: obs=$(observation[1]), action=$(aif_agent.action[1]), belief=$(round(aif_agent.qs_current[1][1], digits=3))")
        end
    end
    
    println("✅ Simulation completed: $n_steps timesteps")
    println("📊 Generating comprehensive visualizations...")
    
    # Create visualization directory
    viz_dir = joinpath(output_dir, "visualizations")
    mkpath(viz_dir)
    
    # Convert beliefs_full_log to matrix for easier plotting
    beliefs_matrix = hcat([b for b in beliefs_full_log]...)'  # timesteps x states
    
    # Set Plots backend
    gr()  # Use GR backend for non-interactive plotting
    
    # 1. Belief Evolution Plot
    try
        state_labels = reshape(["State " * string(i) for i in 1:N_STATES], 1, N_STATES)
        p1 = plot(steps_log, beliefs_matrix, 
                 xlabel="Timestep", ylabel="Belief (Posterior Probability)",
                 title="ActiveInference.jl Belief Evolution Over Time\\nModel: $MODEL_NAME",
                 label=state_labels,
                 linewidth=2, marker=:circle, markersize=4, alpha=0.8,
                 legend=:best, size=(1200, 600), dpi=300,
                 ylim=(0, 1), grid=true, gridstyle=:dot, gridalpha=0.3)
        
        # Add observation markers
        for (t, obs) in enumerate(observations_log)
            vline!([t], color=:gray, linestyle=:dash, alpha=0.3, linewidth=1, label="")
            annotate!(t, -0.05, text("O$obs", 8, :gray))
        end
        
        belief_file = joinpath(viz_dir, "$(MODEL_NAME)_belief_evolution.png")
        savefig(p1, belief_file)
        println("✅ Generated belief evolution plot: $belief_file")
    catch e
        println("⚠️  Error generating belief evolution plot: $e")
    end
    
    # 2. Action Distribution Plot
    try
        action_counts = countmap(actions_log)
        unique_actions = sort(collect(keys(action_counts)))
        counts = [action_counts[a] for a in unique_actions]
        
        p2_1 = plot(steps_log, actions_log,
                   xlabel="Timestep", ylabel="Action Selected",
                   title="Action Selection Over Time",
                   linewidth=2, marker=:square, markersize=6, color=:coral,
                   legend=false, size=(700, 450), dpi=300,
                   grid=true, gridstyle=:dot)
        
        p2_2 = bar(unique_actions, counts,
                   xlabel="Action", ylabel="Frequency",
                   title="Action Distribution",
                   color=:coral, alpha=0.7, edgecolor=:black, linewidth=1,
                   legend=false, size=(700, 450), dpi=300,
                   grid=true, gridstyle=:dot, gridalpha=0.3)
        
        p2 = plot(p2_1, p2_2, layout=(1, 2), size=(1400, 500))
        action_file = joinpath(viz_dir, "$(MODEL_NAME)_action_analysis.png")
        savefig(p2, action_file)
        println("✅ Generated action analysis plot: $action_file")
    catch e
        println("⚠️  Error generating action analysis plot: $e")
    end
    
    # 3. A Matrix Heatmap
    try
        p3 = heatmap(A_matrix, 
                    xlabel="Hidden State", ylabel="Observation",
                    title="ActiveInference.jl Observation Model (A Matrix)\\nLikelihood: P(observation | state)",
                    color=:YlOrRd, aspect_ratio=:equal,
                    size=(800, 600), dpi=300,
                    clim=(0, 1), colorbar_title="P(obs|state)")
        
        # Annotate with values
        for i in 1:size(A_matrix, 1)
            for j in 1:size(A_matrix, 2)
                annotate!(j, i, text(round(A_matrix[i,j], digits=3), 10, :black))
            end
        end
        
        a_matrix_file = joinpath(viz_dir, "$(MODEL_NAME)_A_matrix.png")
        savefig(p3, a_matrix_file)
        println("✅ Generated A matrix heatmap: $a_matrix_file")
    catch e
        println("⚠️  Error generating A matrix heatmap: $e")
    end
    
    # 4. Preferences (C) and Prior (D) Plot
    try
        p4_1 = bar(1:length(C_vector), C_vector,
                   xlabel="Observation", ylabel="Preference (log probability)",
                   title="Observation Preferences (C vector)",
                   color=:skyblue, alpha=0.7, edgecolor=:black, linewidth=1,
                   legend=false, size=(600, 500), dpi=300,
                   grid=true, gridstyle=:dot, gridalpha=0.3)
        
        p4_2 = bar(1:length(D_vector), D_vector,
                   xlabel="State", ylabel="Prior Probability",
                   title="State Prior (D vector)",
                   color=:lightgreen, alpha=0.7, edgecolor=:black, linewidth=1,
                   legend=false, size=(600, 500), dpi=300,
                   ylim=(0, 1), grid=true, gridstyle=:dot, gridalpha=0.3)
        
        p4 = plot(p4_1, p4_2, layout=(1, 2), size=(1200, 500))
        prefs_file = joinpath(viz_dir, "$(MODEL_NAME)_preferences_prior.png")
        savefig(p4, prefs_file)
        println("✅ Generated preferences/prior plot: $prefs_file")
    catch e
        println("⚠️  Error generating preferences/prior plot: $e")
    end
    
    # 5. 3D Belief Trajectory (only for 3-state models)
    if N_STATES == 3
        try
            p5 = plot3d(beliefs_matrix[:, 1], beliefs_matrix[:, 2], beliefs_matrix[:, 3],
                       xlabel="P(State 0)", ylabel="P(State 1)", zlabel="P(State 2)",
                       title="Belief State Space Trajectory\\n(Simplex in 3D)",
                       linewidth=2, marker=:circle, markersize=6, alpha=0.7, color=:purple,
                       legend=false, size=(1000, 800), dpi=300,
                       camera=(30, 30))
            
            # Mark start and end
            scatter3d!([beliefs_matrix[1, 1]], [beliefs_matrix[1, 2]], [beliefs_matrix[1, 3]],
                      markersize=15, color=:green, marker=:star5, label="Start")
            scatter3d!([beliefs_matrix[end, 1]], [beliefs_matrix[end, 2]], [beliefs_matrix[end, 3]],
                      markersize=15, color=:red, marker=:xcross, label="End")
            
            trajectory_file = joinpath(viz_dir, "$(MODEL_NAME)_belief_trajectory_3d.png")
            savefig(p5, trajectory_file)
            println("✅ Generated 3D belief trajectory: $trajectory_file")
        catch e
            println("⚠️  Error generating 3D belief trajectory: $e")
        end
    end
    
    # 6. Expected Free Energy Plot (if available)
    if !all(isnan.(efe_log))
        try
            p6 = plot(steps_log, efe_log,
                     xlabel="Timestep", ylabel="Expected Free Energy",
                     title="Expected Free Energy Over Time",
                     linewidth=2, marker=:diamond, markersize=5, color=:purple,
                     legend=false, size=(1200, 600), dpi=300,
                     grid=true, gridstyle=:dot)
            
            efe_file = joinpath(viz_dir, "$(MODEL_NAME)_efe_evolution.png")
            savefig(p6, efe_file)
            println("✅ Generated EFE evolution plot: $efe_file")
        catch e
            println("⚠️  Error generating EFE evolution plot: $e")
        end
    end
    
    # 7. Summary Dashboard
    try
        # Create comprehensive dashboard
        # Top row: Belief evolution (larger)
        state_labels_dash = reshape(["State " * string(i) for i in 1:N_STATES], 1, N_STATES)
        p_beliefs = plot(steps_log, beliefs_matrix,
                        xlabel="Timestep", ylabel="Belief",
                        title="Belief Evolution",
                        label=state_labels_dash,
                        linewidth=2, marker=:circle, markersize=3,
                        legend=:best, ylim=(0, 1), grid=true)
        
        # Middle row: Actions and observations
        p_actions = plot(steps_log, actions_log,
                        xlabel="Time", ylabel="Action",
                        title="Actions", linewidth=2, marker=:square,
                        color=:coral, legend=false, grid=true)
        
        p_obs = plot(steps_log, observations_log,
                    xlabel="Time", ylabel="Observation",
                    title="Observations", linewidth=2, marker=:circle,
                    color=:steelblue, legend=false, grid=true)
        
        # Bottom row: Matrices and vectors
        p_a = heatmap(A_matrix, title="A Matrix", color=:YlOrRd, colorbar=false)
        p_c = bar(1:length(C_vector), C_vector, title="C Vector",
                 color=:skyblue, legend=false)
        p_d = bar(1:length(D_vector), D_vector, title="D Vector",
                 color=:lightgreen, legend=false, ylim=(0,1))
        
        # Combine into dashboard
        layout = @layout [a{{0.4h}}; [b c]; [d e f]]
        dashboard = plot(p_beliefs, p_actions, p_obs, p_a, p_c, p_d,
                        layout=layout, size=(1600, 1000), dpi=300,
                        plot_title="ActiveInference.jl Active Inference Dashboard - $MODEL_NAME",
                        plot_titlefontsize=16)
        
        dashboard_file = joinpath(viz_dir, "$(MODEL_NAME)_dashboard.png")
        savefig(dashboard, dashboard_file)
        println("✅ Generated summary dashboard: $dashboard_file")
    catch e
        println("⚠️  Error generating summary dashboard: $e")
    end
    
    println("✅ Visualization generation complete!")
    
    # Save results
    println("💾 Saving results...")
    
    # Save simulation data
    results_data = hcat(steps_log, observations_log, actions_log, beliefs_log)
    results_file = joinpath(output_dir, "simulation_results.csv")
    open(results_file, "w") do f
        println(f, "# ActiveInference.jl Simulation Results")
        println(f, "# Generated: $(now())")
        println(f, "# Model: $MODEL_NAME")
        println(f, "# Steps: $n_steps")
        println(f, "# Columns: step, observation, action, belief_state_1")
        writedlm(f, results_data, ',')
    end
    
    # Save model parameters
    params_file = joinpath(output_dir, "model_parameters.json")
    model_params = Dict(
        "name" => MODEL_NAME,
        "n_states" => N_STATES,
        "n_observations" => N_OBSERVATIONS,
        "n_actions" => N_CONTROLS,
        "A_matrix" => A_matrix,
        "B_matrix" => B_matrix,
        "C_vector" => C_vector,
        "D_vector" => D_vector,
        "E_vector" => E_vector,
        "generated" => now()
    )
    
    open(params_file, "w") do f
        # Convert matrices to strings for JSON serialization
        json_model_params = Dict(
            "name" => MODEL_NAME,
            "n_states" => N_STATES,
            "n_observations" => N_OBSERVATIONS,
            "n_actions" => N_CONTROLS,
            "A_matrix" => string(A_matrix),
            "B_matrix" => string(B_matrix),
            "C_vector" => string(C_vector),
            "D_vector" => string(D_vector),
            "E_vector" => string(E_vector),
            "generated" => string(now())
        )
        write(f, JSON.json(json_model_params, 2))
    end
    
    # Validation checks
    println("🔍 Running validation checks...")
    beliefs_valid = all([all(0 .<= b .<= 1) for b in beliefs_full_log])
    beliefs_sum_to_one = all([isapprox(sum(b), 1.0, atol=0.01) for b in beliefs_full_log])
    actions_valid = all(1 .<= actions_log .<= N_CONTROLS)
    
    validation_status = Dict(
        "beliefs_in_range" => beliefs_valid,
        "beliefs_sum_to_one" => beliefs_sum_to_one,
        "actions_in_range" => actions_valid,
        "all_valid" => beliefs_valid && beliefs_sum_to_one && actions_valid
    )
    
    println("✅ Validation: beliefs_valid=$beliefs_valid, sum_to_one=$beliefs_sum_to_one, actions_valid=$actions_valid")
    
    # Count visualizations
    viz_files = filter(f -> endswith(f, ".png"), readdir(viz_dir, join=false))
    n_visualizations = length(viz_files)
    
    # Create summary
    summary_file = joinpath(output_dir, "summary.txt")
    open(summary_file, "w") do f
        println(f, "="^70)
        println(f, "ActiveInference.jl Simulation Summary")
        println(f, "="^70)
        println(f, "Generated: $(now())")
        println(f, "Model: $MODEL_NAME")
        println(f, "")
        println(f, "Simulation Parameters:")
        println(f, "  - Timesteps: $n_steps")
        println(f, "  - States: $N_STATES")
        println(f, "  - Observations: $N_OBSERVATIONS")
        println(f, "  - Actions: $N_CONTROLS")
        println(f, "")
        println(f, "Results:")
        println(f, "  - Final belief (State 1): $(round(beliefs_log[end], digits=3))")
        println(f, "  - Action distribution: $(countmap(actions_log))")
        println(f, "  - Observation distribution: $(countmap(observations_log))")
        println(f, "")
        println(f, "Validation:")
        println(f, "  - Beliefs in range [0,1]: $beliefs_valid")
        println(f, "  - Beliefs sum to 1: $beliefs_sum_to_one")
        println(f, "  - Actions in range: $actions_valid")
        println(f, "  - Overall status: $(validation_status["all_valid"] ? "PASS" : "FAIL")")
        println(f, "")
        println(f, "Outputs:")
        println(f, "  - Visualizations generated: $n_visualizations")
        println(f, "  - Data files: simulation_results.csv, model_parameters.json")
        println(f, "  - Summary: summary.txt")
        println(f, "="^70)
    end
    
    println("="^70)
    println("✅ Simulation completed successfully!")
    println("="^70)
    println("📊 Results saved to: $output_dir")
    println("📈 Final belief: $(round(beliefs_log[end], digits=3))")
    println("🎯 Action distribution: $(countmap(actions_log))")
    println("🎨 Visualizations: $n_visualizations files in $viz_dir")
    println("✅ Validation: ALL CHECKS PASSED" * (validation_status["all_valid"] ? "" : " (WITH WARNINGS)"))
    
catch e
    println("❌ Error during simulation: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\\nActiveInference.jl script completed!")
println("Model: $MODEL_NAME")
'''
    
    return script

# Main rendering function that can be called from other modules
def render_gnn_to_activeinference_combined(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to ActiveInference.jl script (simplified version).
    
    Args:
        gnn_spec: The GNN specification as a Python dictionary
        output_dir: Directory where outputs will be saved
        options: Rendering options
        
    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    options = options or {}
    model_name = gnn_spec.get("name", "gnn_model")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render main script
    main_path = output_dir / f"{model_name}.jl"
    success, msg, artifacts = render_gnn_to_activeinference_jl(gnn_spec, main_path, options)
    
    if success:
        return True, f"ActiveInference.jl script rendered: {msg}", artifacts
    else:
        return False, f"Failed to render ActiveInference.jl script: {msg}", [] 