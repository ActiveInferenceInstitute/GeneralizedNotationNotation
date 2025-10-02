#!/usr/bin/env python3
"""
ActiveInference.jl Renderer

Renders GNN specifications to ActiveInference.jl simulation code for discrete Active Inference.
This renderer creates executable ActiveInference.jl simulations configured from parsed GNN POMDP specifications.

Features:
- GNN-to-ActiveInference.jl parameter extraction
- Julia Active Inference code generation
- Discrete POMDP model specification
- Pipeline integration support

Author: GNN ActiveInference.jl Integration
Date: 2024
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime


class ActiveInferenceJlRenderer:
    """
    ActiveInference.jl renderer for generating Julia Active Inference code from GNN specifications.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize ActiveInference.jl renderer.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.logger = logging.getLogger(__name__)
    
    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to ActiveInference.jl simulation code.
        
        Args:
            gnn_file_path: Path to GNN file
            output_path: Path for output ActiveInference.jl script
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Read GNN file
            with open(gnn_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse GNN content (simplified for now)
            gnn_spec = self._parse_gnn_content(content, gnn_file_path.stem)
            
            # Generate ActiveInference.jl simulation code
            activeinference_code = self._generate_activeinference_simulation_code(gnn_spec, gnn_file_path.stem)
            
            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(activeinference_code)
            
            self.logger.info(f"Generated ActiveInference.jl simulation: {output_path}")
            return True, f"Successfully generated ActiveInference.jl simulation code"
            
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
    
    def _generate_activeinference_simulation_code(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """
        Generate executable ActiveInference.jl simulation code from GNN specification.
        
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

        # Validate initial parameters and extract usable values
        A_values = None
        B_values = None
        C_values = None
        D_values = None
        E_values = None

        try:
            if 'A' in initial_params:
                A_data = initial_params['A']
                if isinstance(A_data, list) and len(A_data) > 0:
                    # Try to convert to numeric matrix, filtering out non-numeric values
                    A_flat = []
                    for row in A_data:
                        if isinstance(row, list):
                            for val in row:
                                try:
                                    # Try to convert to float, skip if not numeric
                                    float_val = float(val)
                                    A_flat.append(float_val)
                                except (ValueError, TypeError):
                                    # Skip non-numeric values like "amplified pain"
                                    continue
                        else:
                            try:
                                float_val = float(row)
                                A_flat.append(float_val)
                            except (ValueError, TypeError):
                                continue

                    if len(A_flat) >= num_observations * num_states:
                        A_values = A_flat[:num_observations * num_states]
                        A_values = [A_values[i:i+num_states] for i in range(0, len(A_values), num_states)]
            # Add similar validation for other matrices
            if 'C' in initial_params:
                C_data = initial_params['C']
                if isinstance(C_data, list) and len(C_data) > 0:
                    C_numeric = []
                    for val in C_data:
                        try:
                            C_numeric.append(float(val))
                        except (ValueError, TypeError):
                            continue
                    if len(C_numeric) >= num_observations:
                        C_values = C_numeric[:num_observations]

            if 'D' in initial_params:
                D_data = initial_params['D']
                if isinstance(D_data, list) and len(D_data) > 0:
                    D_numeric = []
                    for val in D_data:
                        try:
                            D_numeric.append(float(val))
                        except (ValueError, TypeError):
                            continue
                    if len(D_numeric) >= num_states:
                        D_values = D_numeric[:num_states]

        except Exception as e:
            print(f"Warning: Failed to parse initial parameters: {e}. Using defaults.")
        
        # Generate the Julia code
        code = f'''# ActiveInference.jl Simulation
# Generated from GNN Model: {model_display_name}
# Generated: {self._get_timestamp()}

using ActiveInference
using LinearAlgebra
using Random
using Plots
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Model parameters extracted from GNN specification
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = 20

println("🔬 ActiveInference.jl Simulation")
println("📊 State Space: $NUM_STATES states, $NUM_OBSERVATIONS observations, $NUM_ACTIONS actions")

# Initialize model matrices from GNN specification
function initialize_matrices()
    println("\\n🏗️  Initializing model matrices...")
    
    # A matrix: Observation model P(o|s)
    # Use provided values if available, otherwise create default
    if A_values is not None and len(A_values) == NUM_OBSERVATIONS and len(A_values[0]) == NUM_STATES:
        A = Matrix{Float64}(hcat([A_values[i] for i in 1:length(A_values)]...))
        A = A ./ sum(A, dims=1)  # Normalize columns
        println("✓ Using provided A matrix values")
    else:
        # Create identity-like mapping with some noise for realism
        A = Matrix{Float64}(I, NUM_OBSERVATIONS, NUM_STATES)
        if NUM_OBSERVATIONS != NUM_STATES
            # If dimensions don't match, create appropriate mapping
            A = rand(NUM_OBSERVATIONS, NUM_STATES)
            A = A ./ sum(A, dims=1)  # Normalize columns
        else
            # Add small noise to diagonal
            A += 0.1 * rand(NUM_OBSERVATIONS, NUM_STATES)
            A = A ./ sum(A, dims=1)  # Normalize columns
        end
        println("✓ Using default A matrix (random initialization)")
    end
    
    # B matrix: Transition model P(s'|s,a)
    B = zeros(NUM_STATES, NUM_STATES, NUM_ACTIONS)
    for action in 1:NUM_ACTIONS
        # Create different transition patterns for each action
        if action == 1
            # Action 1: Stay in same state (identity + noise)
            B[:, :, action] = Matrix{Float64}(I, NUM_STATES, NUM_STATES)
            B[:, :, action] += 0.1 * rand(NUM_STATES, NUM_STATES)
        else
            # Other actions: Move to next state (cyclical)
            for s in 1:NUM_STATES
                next_state = (s % NUM_STATES) + 1
                B[next_state, s, action] = 0.8
                B[s, s, action] = 0.2  # Some probability of staying
            end
        end
        # Normalize columns for each action
        for s in 1:NUM_STATES
            B[:, s, action] = B[:, s, action] ./ sum(B[:, s, action])
        end
    end
    
    # C vector: Preferences over observations
    if C_values is not None and len(C_values) >= NUM_OBSERVATIONS:
        C = Vector{Float64}(C_values[:NUM_OBSERVATIONS])
        println("✓ Using provided C vector values")
    else:
        C = zeros(NUM_OBSERVATIONS)
        C[end] = 2.0  # Prefer last observation state
        if NUM_OBSERVATIONS > 1
            C[1] = -1.0  # Avoid first observation state
        end
        println("✓ Using default C vector")
    end

    # D vector: Prior beliefs over initial states
    if D_values is not None and len(D_values) >= NUM_STATES:
        D = Vector{Float64}(D_values[:NUM_STATES])
        D = D ./ sum(D)  # Normalize to probability distribution
        println("✓ Using provided D vector values")
    else:
        D = ones(NUM_STATES) / NUM_STATES  # Uniform prior
        println("✓ Using default D vector (uniform prior)")
    end
    
    println("✓ Matrices initialized successfully")
    println("  - A matrix shape: $(size(A))")
    println("  - B matrix shape: $(size(B))")
    println("  - C vector length: $(length(C))")
    println("  - D vector length: $(length(D))")
    
    return A, B, C, D
end

# Initialize the matrices
A, B, C, D = initialize_matrices()

# Create agent
function create_agent(A, B, C, D)
    println("\\n🤖 Creating Active Inference agent...")
    
    try
        # Create agent with initialized matrices
        agent = Agent(
            A = A,
            B = B, 
            C = C,
            D = D,
            planning_horizon = 3,
            action_selection = "deterministic"
        )
        
        println("✓ Agent created successfully")
        return agent
        
    catch e
        println("❌ Failed to create agent: $e")
        
        # Fallback: create agent with different parameters
        println("🔄 Trying fallback agent creation...")
        agent = Agent(A, B, C, D)
        println("✓ Fallback agent created")
        return agent
    end
end

# Environment simulation
mutable struct SimpleEnvironment
    state::Int
    num_states::Int
    A::Matrix{Float64}
    B::Array{Float64, 3}
    
    function SimpleEnvironment(initial_state::Int, A::Matrix, B::Array)
        new(initial_state, size(A, 2), A, B)
    end
end

function step!(env::SimpleEnvironment, action::Int)
    # Sample next state according to transition model
    transition_probs = env.B[:, env.state, action]
    next_state = sample_from_categorical(transition_probs)
    env.state = next_state
    
    # Generate observation according to observation model
    obs_probs = env.A[:, env.state]
    observation = sample_from_categorical(obs_probs)
    
    return observation
end

function sample_from_categorical(probs::Vector)
    cumsum_probs = cumsum(probs)
    rand_val = rand()
    return findfirst(x -> x >= rand_val, cumsum_probs)
end

# Run simulation
function run_simulation()
    println("\\n🚀 Starting Active Inference simulation...")
    
    # Create agent
    agent = create_agent(A, B, C, D)
    
    # Create environment
    initial_state = 1
    env = SimpleEnvironment(initial_state, A, B)
    
    # Storage for results
    observations = Int[]
    actions = Int[]
    states = Int[]
    beliefs = Vector{{Vector{{Float64}}}}()
    
    println("\\n📈 Running simulation for $TIME_STEPS steps...")
    
    for t in 1:TIME_STEPS
        # Current state and observation
        push!(states, env.state)
        
        # Generate observation
        observation = step!(env, 1)  # Default action for first step
        push!(observations, observation)
        
        # Agent inference
        try
            # Infer states given observation
            qs = infer_states(agent, [observation])
            push!(beliefs, qs[1])  # Store belief over first state factor
            
            # Infer policies and select action
            q_pi = infer_policies(agent)
            action = sample_action(agent, q_pi)
            push!(actions, action[1])  # Take first action component
            
            # Update environment with selected action
            if t < TIME_STEPS
                step!(env, action[1])
            end
            
            # Print progress
            if t % 5 == 0 || t <= 5
                belief_max = argmax(qs[1])
                println("  Step $t: obs=$observation, action=$(action[1]), state=$(env.state), belief_peak=$belief_max")
            end
            
        catch e
            println("⚠️  Error at step $t: $e")
            # Fallback: random action
            random_action = rand(1:NUM_ACTIONS)
            push!(actions, random_action)
            push!(beliefs, ones(NUM_STATES) / NUM_STATES)
        end
    end
    
    println("✅ Simulation completed!")
    
    return (
        observations = observations,
        actions = actions,
        states = states,
        beliefs = beliefs
    )
end

# Visualization function
function plot_results(results)
    println("\\n📊 Creating visualizations...")
    
    try
        observations, actions, states, beliefs = results.observations, results.actions, results.states, results.beliefs
        
        # Convert beliefs to matrix for plotting
        belief_matrix = hcat(beliefs...)'
        
        # Create plots
        p1 = plot(1:length(observations), observations,
                 title="Observations Over Time",
                 xlabel="Time Step", ylabel="Observation",
                 marker=:circle, linewidth=2, label="Observations")
        
        p2 = plot(1:length(actions), actions,
                 title="Actions Over Time", 
                 xlabel="Time Step", ylabel="Action",
                 marker=:square, linewidth=2, label="Actions")
        
        p3 = plot(1:length(states), states,
                 title="True States Over Time",
                 xlabel="Time Step", ylabel="State", 
                 marker=:diamond, linewidth=2, label="States")
        
        p4 = heatmap(1:TIME_STEPS, 1:NUM_STATES, belief_matrix',
                    title="State Beliefs Over Time",
                    xlabel="Time Step", ylabel="State",
                    color=:viridis)
        
        # Combine plots
        combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
        
        # Save plot
        output_file = "activeinference_jl_results.png"
        savefig(combined_plot, output_file)
        println("💾 Saved visualization to: $output_file")
        
        # Print summary statistics
        println("\\n📊 Summary Statistics:")
        println("  - Average state: $(round(mean(states), digits=2))")
        println("  - Most common observation: $(mode(observations))")
        println("  - Most common action: $(mode(actions))")
        println("  - Final state: $(states[end])")
        
    catch e
        println("⚠️  Visualization failed: $e")
    end
end

# Main execution function
function main()
    println("="^60)
    println("ActiveInference.jl - GNN Generated Simulation")
    println("Model: {model_display_name}")
    println("="^60)
    
    try
        # Run the simulation
        results = run_simulation()
        
        # Create visualizations
        plot_results(results)
        
        println("\\n🎉 ActiveInference.jl simulation completed successfully!")
        
        return 0
        
    catch e
        println("❌ Simulation failed: $e")
        println("🔍 Stack trace:")
        println(sprint(showerror, e, catch_backtrace()))
        return 1
    end
end

# Helper function to handle mode calculation
function mode(arr)
    counts = Dict{{eltype(arr), Int}}()
    for item in arr
        counts[item] = get(counts, item, 0) + 1
    end
    return argmax(counts)
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


def render_gnn_to_activeinference_jl(
    gnn_spec: Dict[str, Any],
    output_script_path: Path,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to ActiveInference.jl simulation script.
    
    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_script_path: Path for output ActiveInference.jl script
        options: Optional rendering options
        
    Returns:
        Tuple of (success, message, warnings)
    """
    try:
        renderer = ActiveInferenceJlRenderer(options)
        
        # Generate simulation code directly from spec
        model_name = gnn_spec.get('model_name', 'GNN_Model')
        activeinference_code = renderer._generate_activeinference_simulation_code(gnn_spec, model_name)
        
        # Write output file
        output_script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_script_path, 'w', encoding='utf-8') as f:
            f.write(activeinference_code)
        
        message = f"Generated ActiveInference.jl simulation script: {output_script_path}"
        warnings = []
        
        # Check for potential issues
        if not gnn_spec.get('initial_parameterization'):
            warnings.append("No initial parameterization found - using defaults")
        
        if not gnn_spec.get('model_parameters'):
            warnings.append("No model parameters found - using inferred dimensions")
        
        return True, message, warnings
        
    except Exception as e:
        return False, f"Error generating ActiveInference.jl script: {e}", [] 