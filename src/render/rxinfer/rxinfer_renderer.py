#!/usr/bin/env python3
"""
RxInfer.jl Renderer

Renders GNN specifications to RxInfer.jl simulation code using probabilistic programming.
This renderer creates executable RxInfer.jl simulations configured from parsed GNN POMDP specifications.

Features:
- GNN-to-RxInfer parameter extraction
- Julia probabilistic programming code generation
- Bayesian Active Inference model specification
- Pipeline integration support

Author: GNN RxInfer Integration
Date: 2024
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime


class RxInferRenderer:
    """
    RxInfer.jl renderer for generating Julia probabilistic programming code from GNN specifications.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize RxInfer renderer.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.logger = logging.getLogger(__name__)
    
    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to RxInfer.jl simulation code.
        
        Args:
            gnn_file_path: Path to GNN file
            output_path: Path for output RxInfer script
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read GNN file
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse GNN content (simplified for now)
            gnn_spec = self._parse_gnn_content(content, gnn_file_path.stem)
            
            # Generate RxInfer.jl simulation code
            rxinfer_code = self._generate_rxinfer_simulation_code(gnn_spec, gnn_file_path.stem)
            
            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rxinfer_code)
            
            self.logger.info(f"Generated RxInfer.jl simulation: {output_path}")
            return True, f"Successfully generated RxInfer.jl simulation code"
            
        except Exception as e:
            error_msg = f"Error rendering {gnn_file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _parse_gnn_content(self, content: str, model_name: str) -> Dict[str, Any]:
        """Parse GNN content into a structured dictionary (simplified parser)."""
        gnn_spec = {
            'model_name': model_name,
            'variables': [],
            'model_parameters': {},
            'initial_parameterization': {}
        }
        
        # Simple parser for key sections
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                current_section = line[3:].strip()
            elif current_section == 'ModelParameters' and ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    if '.' in value:
                        gnn_spec['model_parameters'][key] = float(value)
                    else:
                        gnn_spec['model_parameters'][key] = int(value)
                except ValueError:
                    gnn_spec['model_parameters'][key] = value
        
        return gnn_spec
    
    def _generate_rxinfer_simulation_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """
        Generate executable RxInfer.jl simulation code from GNN specification.
        
        Args:
            gnn_spec: Parsed GNN specification
            model_name: Name of the model
            
        Returns:
            Generated Julia code string
        """
        # Extract key information from GNN spec
        model_display_name = gnn_spec.get('model_name', model_name)
        
        # Extract dimensions from model parameters
        model_params = gnn_spec.get('model_parameters', {})
        num_states = model_params.get('num_hidden_states', 3)
        num_observations = model_params.get('num_obs', 3)
        num_actions = model_params.get('num_actions', 3)
        
        # Try to extract from variables if available
        variables = gnn_spec.get('variables', [])
        for var in variables:
            if var.get('name') == 'A' and 'dimensions' in var:
                dims = var['dimensions']
                if len(dims) >= 2:
                    num_observations = dims[0]
                    num_states = dims[1]
            elif var.get('name') == 'B' and 'dimensions' in var:
                dims = var['dimensions']
                if len(dims) >= 3:
                    num_actions = dims[2]
        
        # Get initial parameterization if available
        initial_params = gnn_spec.get('initial_parameterization', {})
        
        # Generate the Julia code
        code = f'''# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: {model_display_name}
# Generated: {self._get_timestamp()}

using RxInfer
using Distributions
using LinearAlgebra
using Plots
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Model parameters extracted from GNN specification
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = 20

println("🔬 RxInfer.jl Active Inference Simulation")
println("📊 State Space: $NUM_STATES states, $NUM_OBSERVATIONS observations, $NUM_ACTIONS actions")

# Define the Active Inference model using GraphPPL
@model function active_inference_model(n_steps, observations)
    
    # Hyperparameters for priors
    α_A = ones(NUM_OBSERVATIONS, NUM_STATES)  # Prior for A matrix
    α_B = ones(NUM_STATES, NUM_STATES, NUM_ACTIONS)  # Prior for B tensor
    α_D = ones(NUM_STATES)  # Prior for initial state distribution
    
    # Model parameters
    A ~ MatrixDirichlet(α_A)  # Observation model P(o|s)
    B ~ ArrayDirichlet(α_B)   # Transition model P(s'|s,a) 
    D ~ Dirichlet(α_D)        # Initial state distribution
    
    # Preference parameters (can be learned or fixed)
    C = zeros(NUM_OBSERVATIONS)
    C[end] = 2.0  # Prefer last observation state
    
    # State sequence
    s = Vector{{RandomVariable}}(undef, n_steps)
    
    # Initial state
    s[1] ~ Categorical(D)
    
    # State transitions (simplified - assumes action selection)
    for t in 2:n_steps
        # For now, assume optimal action selection (can be extended)
        action_idx = 1  # Default action
        s[t] ~ Categorical(B[:, s[t-1], action_idx])
    end
    
    # Observations
    for t in 1:n_steps
        observations[t] ~ Categorical(A[:, s[t]])
    end
    
    return (states=s, A=A, B=B, D=D)
end

# Generate synthetic observations for demonstration
function generate_observations(n_steps::Int)
    # Simple observation sequence (can be replaced with real data)
    obs = Vector{{Int}}(undef, n_steps)
    for t in 1:n_steps
        if t <= n_steps ÷ 2
            obs[t] = 1  # First half: observation 1
        else
            obs[t] = NUM_OBSERVATIONS  # Second half: final observation
        end
    end
    return obs
end

# Run inference
function run_active_inference_simulation()
    println("\\n🚀 Starting Active Inference simulation...")
    
    # Generate observations
    observations_data = generate_observations(TIME_STEPS)
    println("📋 Generated observation sequence: $observations_data")
    
    # Create data for inference
    data = (observations = observations_data,)
    
    # Perform inference
    println("\\n🧠 Running variational inference...")
    result = infer(
        model = active_inference_model(TIME_STEPS, observations_data),
        data = data,
        options = (
            iterations = 50,
            showprogress = true,
            free_energy = true
        )
    )
    
    # Extract results
    println("\\n📊 Inference Results:")
    
    # Extract posterior marginals
    states_marginals = result.posteriors[:states]
    A_marginal = result.posteriors[:A]
    B_marginal = result.posteriors[:B]
    D_marginal = result.posteriors[:D]
    
    println("✓ Successfully computed posterior marginals")
    println("  - State posteriors: ${{length(states_marginals)}} time steps")
    
    # Compute free energy if available
    if haskey(result, :free_energy)
        free_energy = result.free_energy
        println("🎯 Free Energy: $free_energy")
    end
    
    # Display state beliefs over time
    println("\\n📈 State beliefs over time:")
    for (t, state_belief) in enumerate(states_marginals)
        belief_mode = mode(state_belief)
        belief_prob = pdf(state_belief, belief_mode)
        println("  Step $t: Most likely state = $belief_mode (prob ≈ $(round(belief_prob, digits=3)))")
    end
    
    return result
end

# Visualization function
function plot_results(result, observations_data)
    println("\\n📊 Creating visualization...")
    
    try
        # Extract state posteriors
        states_marginals = result.posteriors[:states]
        
        # Create state probability matrix
        state_probs = zeros(TIME_STEPS, NUM_STATES)
        for (t, marginal) in enumerate(states_marginals)
            for s in 1:NUM_STATES
                state_probs[t, s] = pdf(marginal, s)
            end
        end
        
        # Plot state beliefs over time
        p1 = heatmap(
            1:TIME_STEPS, 1:NUM_STATES, state_probs',
            title="State Beliefs Over Time",
            xlabel="Time Step", ylabel="State",
            color=:viridis
        )
        
        # Plot observations
        p2 = plot(
            1:TIME_STEPS, observations_data,
            title="Observation Sequence",
            xlabel="Time Step", ylabel="Observation",
            marker=:circle, linewidth=2
        )
        
        # Combine plots
        combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
        
        # Save plot
        output_file = "rxinfer_active_inference_results.png"
        savefig(combined_plot, output_file)
        println("💾 Saved visualization to: $output_file")
        
    catch e
        println("⚠️  Visualization failed: $e")
    end
end

# Main execution
function main()
    println("="^60)
    println("RxInfer.jl Active Inference - GNN Generated Simulation")
    println("Model: {model_display_name}")
    println("="^60)
    
    try
        # Run the simulation
        result = run_active_inference_simulation()
        
        # Generate synthetic observations for plotting
        observations_data = generate_observations(TIME_STEPS)
        
        # Create visualizations
        plot_results(result, observations_data)
        
        println("\\n✅ Simulation completed successfully!")
        println("🎉 Active Inference with RxInfer.jl finished.")
        
        return 0
        
    catch e
        println("❌ Simulation failed: $e")
        println("🔍 Stack trace:")
        println(sprint(showerror, e, catch_backtrace()))
        return 1
    end
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
'''
        
        return code
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_gnn_to_rxinfer(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to RxInfer.jl simulation script.
    
    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_script_path: Path for output RxInfer script
        options: Optional rendering options
        
    Returns:
        Tuple of (success, message, warnings)
    """
    try:
        renderer = RxInferRenderer(options)
        
        # Generate simulation code directly from spec
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        rxinfer_code = renderer._generate_rxinfer_simulation_code(gnn_spec, model_name)
        
        # Write output file
        output_script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_script_path, 'w', encoding='utf-8') as f:
            f.write(rxinfer_code)
        
        message = f"Generated RxInfer.jl simulation script: {output_script_path}"
        warnings = []
        
        # Check for potential issues
        if not gnn_spec.get('initial_parameterization'):
            warnings.append("No initial parameterization found - using defaults")
        
        if not gnn_spec.get('model_parameters'):
            warnings.append("No model parameters found - using inferred dimensions")
        
        return True, message, warnings
        
    except Exception as e:
        return False, f"Error generating RxInfer.jl script: {e}", [] 