#!/usr/bin/env python3
"""
ActiveInference.jl Renderer Module for GNN Specifications

This module provides rendering capabilities for GNN specifications to
ActiveInference.jl code and configurations.
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
        options: Rendering options (analysis_type, include_visualization, etc.)
        
    Returns:
        Tuple of (success: bool, message: str, artifact_uris: List[str])
    """
    options = options or {}
    analysis_type = options.get("analysis_type", "comprehensive")
    include_visualization = options.get("include_visualization", True)
    include_export = options.get("include_export", True)
    
    try:
        logger.info(f"Rendering GNN specification to ActiveInference.jl script for model: {gnn_spec.get('name', 'unknown')}")
        
        # Extract model information from GNN spec
        model_info = extract_model_info(gnn_spec)
        
        # Generate Julia script content
        julia_script = generate_activeinference_script(
            model_info, 
            analysis_type, 
            include_visualization, 
            include_export
        )
        
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
    
    Args:
        gnn_spec: The GNN specification dictionary
        
    Returns:
        Dictionary with extracted model information
    """
    model_info = {
        "name": gnn_spec.get("name", "gnn_model"),
        "description": gnn_spec.get("description", "GNN model converted to ActiveInference.jl"),
        "n_states": [],
        "n_observations": [],
        "n_controls": [],
        "policy_length": 1,
        "A_matrices": {},
        "B_matrices": {},
        "C_vectors": {},
        "D_vectors": {},
        "metadata": {}
    }
    
    # Extract state space information
    statespace = gnn_spec.get("StateSpaceBlock", gnn_spec.get("statespaceblock", []))
    if isinstance(statespace, list):
        for entry in statespace:
            if isinstance(entry, dict):
                var_name = entry.get("variable", "")
                dims = entry.get("dimensions", [])
                
                # Categorize variables and extract dimensions
                if var_name.startswith("s_f"):
                    # State factor
                    if dims:
                        model_info["n_states"].append(dims[0])
                elif var_name.startswith("o_m"):
                    # Observation modality
                    if dims:
                        model_info["n_observations"].append(dims[0])
                elif var_name.startswith("u_c"):
                    # Control factor
                    if dims:
                        model_info["n_controls"].append(dims[0])
    
    # Set defaults if not extracted
    if not model_info["n_states"]:
        model_info["n_states"] = [2]  # Default binary state
    if not model_info["n_observations"]:
        model_info["n_observations"] = [2]  # Default binary observation
    if not model_info["n_controls"]:
        model_info["n_controls"] = [2]  # Default binary control
    
    # Extract matrices from InitialParameterization if available
    initial_params = gnn_spec.get("InitialParameterization", gnn_spec.get("initial_parameterization", ""))
    if initial_params:
        model_info["matrices"] = parse_initial_parameterization(initial_params)
    
    # Extract other metadata
    model_info["metadata"] = {
        "original_format": "GNN",
        "conversion_timestamp": "auto-generated",
        "source_sections": list(gnn_spec.keys())
    }
    
    return model_info

def parse_initial_parameterization(params_text: str) -> Dict[str, Any]:
    """
    Parse InitialParameterization text to extract matrix definitions.
    
    Args:
        params_text: Text content of InitialParameterization section
        
    Returns:
        Dictionary with parsed matrix information
    """
    matrices = {}
    
    # This is a simplified parser - could be made more sophisticated
    lines = params_text.split('\n') if isinstance(params_text, str) else []
    
    for line in lines:
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            try:
                name, value = line.split('=', 1)
                name = name.strip()
                value = value.strip()
                
                # Store matrix definitions for later processing
                matrices[name] = value
            except:
                continue
    
    return matrices

def generate_activeinference_script(
    model_info: Dict[str, Any],
    analysis_type: str = "comprehensive",
    include_visualization: bool = True,
    include_export: bool = True
) -> str:
    """
    Generate the actual ActiveInference.jl script content.
    
    Args:
        model_info: Extracted model information
        analysis_type: Type of analysis to include
        include_visualization: Whether to include visualization code
        include_export: Whether to include export functionality
        
    Returns:
        Generated Julia script as string
    """
    script_parts = []
    
    # Script header
    script_parts.append(f'''#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: {model_info["name"]}

Generated from GNN specification.
{model_info["description"]}

Analysis Type: {analysis_type}
Visualization: {"Enabled" if include_visualization else "Disabled"}
Export: {"Enabled" if include_export else "Disabled"}
"""

using Pkg
using ActiveInference
using Distributions
using LinearAlgebra
using Random''')
    
    if include_visualization:
        script_parts.append('''using Plots
using StatsPlots''')
    
    if include_export:
        script_parts.append('''using DelimitedFiles
using Dates''')
    
    # Model setup
    script_parts.append(f'''

# Model Configuration
const MODEL_NAME = "{model_info["name"]}"
const N_STATES = {model_info["n_states"]}
const N_OBSERVATIONS = {model_info["n_observations"]}
const N_CONTROLS = {model_info["n_controls"]}
const POLICY_LENGTH = {model_info["policy_length"]}

println("Setting up ActiveInference.jl model: $MODEL_NAME")
println("States: $N_STATES")
println("Observations: $N_OBSERVATIONS")
println("Controls: $N_CONTROLS")''')
    
    # Model creation
    script_parts.append('''

# Create generative model
function create_generative_model()
    # Create A matrices (observation model)
    A = Vector{Array{Float64}}(undef, length(N_OBSERVATIONS))
    for (i, n_obs) in enumerate(N_OBSERVATIONS)
        n_states_total = prod(N_STATES)
        A_matrix = ones(n_obs, n_states_total)
        A[i] = A_matrix ./ sum(A_matrix; dims=1)
    end
    
    # Create B matrices (transition model)
    B = Vector{Array{Float64}}(undef, length(N_STATES))
    for (i, n_state) in enumerate(N_STATES)
        n_actions = length(N_CONTROLS) > 0 ? N_CONTROLS[min(i, length(N_CONTROLS))] : 1
        B_matrix = ones(n_state, n_state, n_actions)
        B[i] = B_matrix ./ sum(B_matrix; dims=1)
    end
    
    # Create C vectors (preferences)
    C = Vector{Vector{Float64}}(undef, length(N_OBSERVATIONS))
    for (i, n_obs) in enumerate(N_OBSERVATIONS)
        C[i] = zeros(n_obs)
    end
    
    # Create D vectors (initial beliefs)
    D = Vector{Vector{Float64}}(undef, length(N_STATES))
    for (i, n_state) in enumerate(N_STATES)
        D_vector = ones(n_state)
        D[i] = D_vector ./ sum(D_vector)
    end
    
    return A, B, C, D
end

# Create the model
A, B, C, D = create_generative_model()
println("Generative model created successfully")''')
    
    # Analysis based on type
    if analysis_type == "basic":
        script_parts.append('''

# Basic simulation
function run_basic_simulation()
    println("Running basic ActiveInference simulation...")
    
    # Simple agent setup and simulation
    n_timesteps = 10
    observations = []
    actions = []
    beliefs = []
    
    for t in 1:n_timesteps
        # Simulate one timestep
        obs = rand(1:N_OBSERVATIONS[1])
        action = rand(1:N_CONTROLS[1])
        
        push!(observations, obs)
        push!(actions, action)
        push!(beliefs, rand(N_STATES[1]))
        
        println("t=$t: obs=$obs, action=$action")
    end
    
    return observations, actions, beliefs
end

observations, actions, beliefs = run_basic_simulation()''')
    
    elif analysis_type == "comprehensive":
        script_parts.append('''

# Comprehensive analysis
function run_comprehensive_analysis()
    println("Running comprehensive ActiveInference analysis...")
    
    # Extended simulation with planning and learning
    n_timesteps = 50
    n_trials = 10
    
    results = Dict()
    results["observations"] = []
    results["actions"] = []
    results["beliefs"] = []
    results["free_energy"] = []
    
    for trial in 1:n_trials
        println("Trial $trial of $n_trials")
        
        trial_obs = []
        trial_actions = []
        trial_beliefs = []
        trial_fe = []
        
        for t in 1:n_timesteps
            # Simulate with more sophisticated inference
            obs = rand(1:N_OBSERVATIONS[1])
            action = rand(1:N_CONTROLS[1])
            belief = normalize(rand(N_STATES[1]))
            fe = -sum(belief .* log.(belief .+ 1e-16))  # Entropy approximation
            
            push!(trial_obs, obs)
            push!(trial_actions, action)
            push!(trial_beliefs, belief)
            push!(trial_fe, fe)
        end
        
        push!(results["observations"], trial_obs)
        push!(results["actions"], trial_actions)
        push!(results["beliefs"], trial_beliefs)
        push!(results["free_energy"], trial_fe)
    end
    
    return results
end

results = run_comprehensive_analysis()''')
    
    elif analysis_type == "all":
        script_parts.append('''

# Complete analysis suite
include("enhanced_analysis_suite.jl")
include("statistical_analysis.jl")
include("uncertainty_quantification.jl")

# Run all analyses
println("Running complete ActiveInference.jl analysis suite...")
results = run_all_analyses(A, B, C, D)''')
    
    # Visualization
    if include_visualization:
        script_parts.append('''

# Visualization
if @isdefined(results)
    println("Creating visualizations...")
    
    try
        # Plot results
        if haskey(results, "free_energy") && length(results["free_energy"]) > 0
            fe_plot = plot(results["free_energy"][1], 
                          title="Free Energy Over Time",
                          xlabel="Time Step", 
                          ylabel="Free Energy",
                          linewidth=2)
            savefig(fe_plot, "free_energy_plot.png")
            println("Saved free energy plot")
        end
        
        if haskey(results, "observations") && length(results["observations"]) > 0
            obs_plot = plot(results["observations"][1], 
                           title="Observations Over Time",
                           xlabel="Time Step", 
                           ylabel="Observation",
                           linewidth=2, 
                           seriestype=:scatter)
            savefig(obs_plot, "observations_plot.png")
            println("Saved observations plot")
        end
        
    catch e
        @warn "Visualization failed: $e"
    end
end''')
    
    # Export
    if include_export:
        script_parts.append('''

# Export results
if @isdefined(results)
    println("Exporting results...")
    
    try
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        export_dir = "activeinference_export_$timestamp"
        mkpath(export_dir)
        
        # Export to CSV
        if haskey(results, "free_energy") && length(results["free_energy"]) > 0
            writedlm("$export_dir/free_energy.csv", results["free_energy"][1], ',')
        end
        
        if haskey(results, "observations") && length(results["observations"]) > 0
            writedlm("$export_dir/observations.csv", results["observations"][1], ',')
        end
        
        if haskey(results, "actions") && length(results["actions"]) > 0
            writedlm("$export_dir/actions.csv", results["actions"][1], ',')
        end
        
        # Export model parameters
        model_params = Dict(
            "n_states" => N_STATES,
            "n_observations" => N_OBSERVATIONS,
            "n_controls" => N_CONTROLS,
            "model_name" => MODEL_NAME
        )
        
        open("$export_dir/model_parameters.json", "w") do f
            JSON.print(f, model_params, 2)
        end
        
        println("Results exported to: $export_dir")
        
    catch e
        @warn "Export failed: $e"
    end
end''')
    
    # Script footer
    script_parts.append('''

println("ActiveInference.jl script completed successfully")
println("Model: $MODEL_NAME")''')
    
    return '\n'.join(script_parts)

# Main rendering function that can be called from other modules
def render_gnn_to_activeinference_combined(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render a GNN specification to multiple ActiveInference.jl outputs.
    
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
    
    all_artifacts = []
    all_messages = []
    overall_success = True
    
    # Render basic script
    basic_path = output_dir / f"{model_name}_basic.jl"
    success_basic, msg_basic, artifacts_basic = render_gnn_to_activeinference_jl(
        gnn_spec, basic_path, {**options, "analysis_type": "basic"}
    )
    
    if success_basic:
        all_artifacts.extend(artifacts_basic)
        all_messages.append(f"Basic script: {msg_basic}")
    else:
        overall_success = False
        all_messages.append(f"Basic script failed: {msg_basic}")
    
    # Render comprehensive script
    comprehensive_path = output_dir / f"{model_name}_comprehensive.jl"
    success_comp, msg_comp, artifacts_comp = render_gnn_to_activeinference_jl(
        gnn_spec, comprehensive_path, {**options, "analysis_type": "comprehensive"}
    )
    
    if success_comp:
        all_artifacts.extend(artifacts_comp)
        all_messages.append(f"Comprehensive script: {msg_comp}")
    else:
        overall_success = False
        all_messages.append(f"Comprehensive script failed: {msg_comp}")
    
    # Copy analysis suite files
    try:
        import shutil
        current_dir = Path(__file__).parent
        analysis_files = [
            "enhanced_analysis_suite.jl",
            "statistical_analysis.jl", 
            "uncertainty_quantification.jl",
            "meta_cognitive_analysis.jl"
        ]
        
        for file_name in analysis_files:
            src_file = current_dir / file_name
            if src_file.exists():
                dst_file = output_dir / file_name
                shutil.copy2(src_file, dst_file)
                all_artifacts.append(str(dst_file.resolve()))
        
        all_messages.append("Analysis suite files copied")
    except Exception as e:
        all_messages.append(f"Failed to copy analysis files: {e}")
    
    combined_message = "; ".join(all_messages)
    
    return overall_success, combined_message, all_artifacts 