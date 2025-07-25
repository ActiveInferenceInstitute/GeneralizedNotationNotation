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
    Robustly handles GNN state space and parameter extraction.
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

    # --- Robust state/obs/action extraction ---
    statespace = gnn_spec.get("statespaceblock", [])
    n_states = None
    n_obs = None
    n_actions = None
    
    for entry in statespace:
        var_id = entry.get("id", "")
        dims = entry.get("dimensions", "")
        if var_id == "s":
            # Hidden state: e.g., '3,1,type=float'
            try:
                n_states = int(dims.split(",")[0])
            except Exception:
                pass
        elif var_id == "o":
            # Observation: e.g., '3,1,type=int'
            try:
                n_obs = int(dims.split(",")[0])
            except Exception:
                pass
        elif var_id == "u":
            # Action: e.g., '1,type=int' (but see B for action count)
            try:
                n_actions = int(dims.split(",")[0])
            except Exception:
                pass
        elif var_id == "A":
            # A matrix: e.g., '3,3,type=float'
            try:
                n_obs = int(dims.split(",")[0])
                n_states = int(dims.split(",")[1])
            except Exception:
                pass
        elif var_id == "B":
            # B matrix: e.g., '3,3,3,type=float'
            try:
                n_states = int(dims.split(",")[0])
                n_actions = int(dims.split(",")[2])
            except Exception:
                pass
    
    # Fallback to ModelParameters if needed
    if n_states is None or n_obs is None or n_actions is None:
        params = gnn_spec.get("raw_sections", {}).get("ModelParameters", "")
        import re
        m_states = re.search(r"num_hidden_states:\s*(\d+)", params)
        m_obs = re.search(r"num_obs:\s*(\d+)", params)
        m_actions = re.search(r"num_actions:\s*(\d+)", params)
        if m_states:
            n_states = int(m_states.group(1))
        if m_obs:
            n_obs = int(m_obs.group(1))
        if m_actions:
            n_actions = int(m_actions.group(1))
    
    # Final fallback: error if any are missing
    if n_states is None or n_obs is None or n_actions is None:
        raise ValueError("Could not extract n_states, n_obs, n_actions from GNN spec. Please check the GNN file.")
    
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
    C_vector = model_info["C"]
    D_vector = model_info["D"]
    E_vector = model_info["E"]
    
    # Convert to Julia format
    def matrix_to_julia(matrix_data):
        if isinstance(matrix_data, list):
            if isinstance(matrix_data[0], list):
                if isinstance(matrix_data[0][0], list):
                    # 3D matrix (B matrix)
                    return "[" + ", ".join([
                        "[" + ", ".join([
                            "[" + ", ".join(str(x) for x in row) + "]"
                            for row in slice
                        ]) + "]"
                        for slice in matrix_data
                    ]) + "]"
                else:
                    # 2D matrix (A matrix)
                    return "[" + ", ".join([
                        "[" + ", ".join(str(x) for x in row) + "]"
                        for row in matrix_data
                    ]) + "]"
            else:
                # 1D vector (C, D, E vectors)
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

using Pkg
using Dates
using Logging
using DelimitedFiles
using Random
using Statistics
using LinearAlgebra
using ActiveInference
using Distributions

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
E_vector = {julia_E}
println("✅ E vector created: $(length(E_vector))")

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
    steps_log = []
    
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
        
        push!(actions_log, aif_agent.action[1])
        push!(beliefs_log, aif_agent.qs_current[1][1])  # First state belief
        push!(steps_log, step)
        
        if step % 5 == 0
            println("Step $step: obs=$(observation[1]), action=$(aif_agent.action[1]), belief=$(round(aif_agent.qs_current[1][1], digits=3))")
        end
    end
    
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
        JSON.print(f, model_params, 2)
    end
    
    # Create summary
    summary_file = joinpath(output_dir, "summary.txt")
    open(summary_file, "w") do f
        println(f, "ActiveInference.jl Simulation Summary")
        println(f, "Generated: $(now())")
        println(f, "Model: $MODEL_NAME")
        println(f, "Steps: $n_steps")
        println(f, "Final belief: $(round(beliefs_log[end], digits=3))")
        println(f, "Action distribution: $(countmap(actions_log))")
        println(f, "Observation distribution: $(countmap(observations_log))")
    end
    
    println("✅ Simulation completed successfully!")
    println("📊 Results saved to: $output_dir")
    println("📈 Final belief: $(round(beliefs_log[end], digits=3))")
    println("🎯 Action distribution: $(countmap(actions_log))")
    
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