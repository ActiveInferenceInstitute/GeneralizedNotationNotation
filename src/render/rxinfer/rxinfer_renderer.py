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
            rxinfer_code = self._generate_rxinfer_simulation_code_simple(gnn_spec, gnn_file_path.stem)
            
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
    
    def _generate_rxinfer_simulation_code_simple(self, gnn_spec: Dict[str, Any], model_name: str) -> str:
        """Generate simplified RxInfer code that actually works with modern API."""
        from datetime import datetime
        
        try:
            # Get model parameters with defaults
            model_display_name = gnn_spec.get('model_name', model_name)
            model_params = gnn_spec.get('model_parameters', {})
            num_states = model_params.get('num_hidden_states', 3)
            num_observations = model_params.get('num_obs', 3)
            
            # Extract num_actions from multiple possible sources
            initial_params = gnn_spec.get('initial_parameterization', {}) or gnn_spec.get('initialparameterization', {})
            B_data = initial_params.get('B', [])
            
            # Infer from B matrix dimensions if available
            inferred_actions = len(B_data) if B_data and isinstance(B_data, list) else None
            
            num_actions = (
                model_params.get('num_actions') or 
                model_params.get('num_controls') or 
                model_params.get('n_actions') or
                inferred_actions or
                3  # Default to 3 for proper POMDP
            )
            
            # Validate parameters
            if not isinstance(num_states, int) or num_states < 1:
                num_states = 3
            if not isinstance(num_observations, int) or num_observations < 1:
                num_observations = 3
            if not isinstance(num_actions, int) or num_actions < 1:
                num_actions = 3
            
            # Read the minimal working template (no deprecated APIs)
            template_path = Path(__file__).parent / 'minimal_template.jl'
            if not template_path.exists():
                raise FileNotFoundError(f"Template file not found: {template_path}")
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Fill in the template - use careful formatting to avoid Julia syntax conflicts
            # Replace placeholders one at a time to avoid issues with curly braces
            code = template
            code = code.replace('{model_name}', model_display_name)
            code = code.replace('{timestamp}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            code = code.replace('{num_states}', str(num_states))
            code = code.replace('{num_observations}', str(num_observations))
            code = code.replace('{num_actions}', str(num_actions))
            
            return code
        except Exception as e:
            raise RuntimeError(f"Failed to generate RxInfer code template: {e}") from e
    
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
        
        # Extract num_actions from multiple possible sources (GNN specs vary)
        # Priority: explicit model param > B matrix dimensions > default
        initial_params = gnn_spec.get('initial_parameterization', {}) or gnn_spec.get('initialparameterization', {})
        B_data = initial_params.get('B', [])
        
        # Try to infer num_actions from B matrix if available
        inferred_actions = None
        if B_data and isinstance(B_data, list) and len(B_data) > 0:
            # B is typically [action][next_state][prev_state] or similar
            inferred_actions = len(B_data)
        
        num_actions = (
            model_params.get('num_actions') or 
            model_params.get('num_controls') or 
            model_params.get('n_actions') or
            inferred_actions or
            3  # Default to 3 for proper POMDP simulation
        )
        A_data = initial_params.get('A', [])
        B_data = initial_params.get('B', [])
        C_data = initial_params.get('C', [])
        D_data = initial_params.get('D', [])
        
        # Generate the Julia code
        code = f'''#!/usr/bin/env julia
# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: {model_display_name}
# Generated: {self._get_timestamp()}

using Pkg

println("📦 Ensuring required packages are installed...")
try
    Pkg.add(["RxInfer", "Distributions", "Plots", "LinearAlgebra", "Random", "StatsBase"])
catch e
    println("⚠️  Package install error (might be already installed): $e")
end

using RxInfer
using Distributions
using LinearAlgebra
using Plots
using Random
using StatsBase
using JSON

Random.seed!(42)

# --- Model Parameters ---
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = 20

# Parameter Matrices (from GNN)
# We use raw Vector of Vectors and convert to Matrix/Tensor for RxInfer
A_raw = {A_data if A_data else "fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS, NUM_STATES)"}
B_raw = {B_data if B_data else "fill(1.0/NUM_STATES, NUM_STATES, NUM_STATES, NUM_ACTIONS)"}
D_raw = {D_data if D_data else "fill(1.0/NUM_STATES, NUM_STATES)"}

# Convert to Julia Matrices
function to_matrix(raw)
    try
        if raw isa Matrix
            return raw
        end
        # Handle Tuple or Vector of Tuples/Vectors
        arr = collect(raw)
        if !isempty(arr) && (arr[1] isa Tuple || arr[1] isa Vector)
            rows = [collect(r) for r in arr]
            return hcat(rows...)'
        end
    catch e
        println("to_matrix warning: $e")
    end
    return raw
end

function to_tensor(raw)
    try
        if raw isa Array{{Float64, 3}}
            return raw
        end
        # Handle Tuple or Vector structure for B[action][row][col]
        arr = collect(raw)
        if !isempty(arr)
            first_action = collect(arr[1])
            if !isempty(first_action) && (first_action[1] isa Tuple || first_action[1] isa Vector)
                n_actions = length(arr)
                n_rows = length(first_action)
                first_row = collect(first_action[1])
                n_cols = length(first_row)
                
                tensor = zeros(n_rows, n_cols, n_actions)
                for a in 1:n_actions
                    action_data = collect(arr[a])
                    for r in 1:n_rows
                        row_data = collect(action_data[r])
                        for c in 1:n_cols
                            tensor[r, c, a] = row_data[c]
                        end
                    end
                end
                return tensor
            end
        end
    catch e
        println("to_tensor warning: $e")
    end
    return raw
end

A_matrix = to_matrix(A_raw)
B_matrix = to_tensor(B_raw) # Handling GNN B format
D_vector = Vector{{Float64}}(collect(D_raw))

# Normalize D_vector to ensure it sums exactly to 1.0 (required for Categorical)
D_vector = D_vector ./ sum(D_vector)

# Normalize A_matrix columns (each column should sum to 1)
for j in 1:size(A_matrix, 2)
    A_matrix[:, j] = A_matrix[:, j] ./ sum(A_matrix[:, j])
end

println("A matrix size: $(size(A_matrix))")
println("B matrix size: $(size(B_matrix))")
println("D vector size: $(size(D_vector))")


# --- RxInfer Model ---
# Fixed parameters version (active inference with known model)
@model function active_inference_model(observations, n_steps, A, B, D)
    
    # State sequence 
    # GraphPPL v3/v4 supports auto-collection of variables via indexing.
    # No randomvar declaration needed.
    
    # Initial state
    s[1] ~ Categorical(D)
    
    # First observation
    observations[1] ~ DiscreteTransition(s[1], A)
    
    # State transitions and observations
    for t in 2:n_steps
        # Action selection (Random policy for now)
        action_idx = rand(1:NUM_ACTIONS)
        
        # State transition 
        # B is [Next, Prev, Action]
        # We slice B by action to get a transition matrix B_a
        B_a = B[:, :, action_idx] 
        s[t] ~ DiscreteTransition(s[t-1], B_a)
        
        # Observation
        observations[t] ~ DiscreteTransition(s[t], A)
    end
    
    return s
end

# --- Simulation & Inference ---
function run_simulation()
    
    # Generate synthetic observations using the generative model (manual)
    # We use the same A and B matrices
    real_states = Vector{{Int}}(undef, TIME_STEPS)
    real_obs = Vector{{Int}}(undef, TIME_STEPS)
    
    # Initial
    current_state = rand(Categorical(D_vector))
    real_states[1] = current_state
    # Manual categorical logic for observations (using probability vector)
    real_obs[1] = rand(Categorical(A_matrix[:, current_state]))
    
    for t in 2:TIME_STEPS
        action = rand(1:NUM_ACTIONS)
        # B_matrix is [Next, Prev, Action]
        next_probs = B_matrix[:, current_state, action]
        current_state = rand(Categorical(next_probs))
        real_states[t] = current_state
        real_obs[t] = rand(Categorical(A_matrix[:, current_state]))
    end
    
    println("Observation sequence: $real_obs")
    
    # One-hot encode observations (required for DiscreteTransition in RxInfer v4)
    function one_hot(idx, n)
        v = zeros(n)
        v[idx] = 1.0
        return v
    end
    
    obs_one_hot = [one_hot(o, NUM_OBSERVATIONS) for o in real_obs]
    println("Observation sequence (one-hot) prepared.")
    
    # Run Inference
    result = infer(
        model = active_inference_model(n_steps=TIME_STEPS, A=A_matrix, B=B_matrix, D=D_vector),
        data = (observations = obs_one_hot,),
        iterations = 10 
    )
    
    println("Inference complete.")
    
    # Standardized result extraction
    # Convert belief traces (Categorical) to probability vectors
    all_iterations = result.posteriors[:s]
    final_iteration = all_iterations[end]
    beliefs = [probvec(final_iteration[t]) for t in 1:TIME_STEPS]
    
    results_data = Dict(
        "framework" => "rxinfer",
        "model_name" => "{model_display_name}",
        "time_steps" => TIME_STEPS,
        "true_states" => real_states,
        "observations" => real_obs,
        "beliefs" => beliefs,
        "num_states" => NUM_STATES,
        "num_observations" => NUM_OBSERVATIONS
    )
    
    open("simulation_results.json", "w") do f
        JSON.print(f, results_data, 4)
    end
    println("✅ Standardized results saved to simulation_results.json")
    
    return result, real_states, real_obs
end

# --- Main ---
function main()
    try
        result, true_states, obs = run_simulation()
        
        # Visualize
        # result.posteriors[:s] returns a vector of iterations, 
        # each being a vector of Categorical distributions over time.
        all_iterations = result.posteriors[:s]
        final_iteration_posteriors = all_iterations[end]
        
        # Extract belief trace for state 1 over time
        belief_trace = [pdf(final_iteration_posteriors[t], 1) for t in 1:TIME_STEPS]
        
        p = plot(belief_trace, label="Belief(State 1)", title="State Inference", xlabel="Time", ylabel="Probability")
        scatter!(p, true_states .== 1, label="True State 1", markershape=:star)
        
        out_path = "rxinfer_results.png"
        savefig(p, out_path)
        println("Saved plot to $out_path")
        println("✅ RxInfer simulation successful")
        return 0
    catch e
        println("❌ Simulation failed: $e")
        # print stacktrace
        showerror(stdout, e, catch_backtrace())
        return 1
    end
end

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
    logger = logging.getLogger(__name__)
    
    try:
        # Validate input
        if not isinstance(gnn_spec, dict):
            return False, "Invalid GNN specification: must be a dictionary", []
        
        renderer = RxInferRenderer(options)
        
        # Get model name safely
        model_name = gnn_spec.get('name') or gnn_spec.get('model_name', 'GNN_Model')
        
        # Generate simulation code directly from spec (using simplified working version)
        try:
            # Use the full generator with updated syntax
            rxinfer_code = renderer._generate_rxinfer_simulation_code(gnn_spec, model_name)
        except Exception as gen_error:
            logger.error(f"Code generation failed: {gen_error}")
            return False, f"Error generating RxInfer.jl code: {gen_error}", []
        
        # Write output file
        try:
            output_script_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_script_path, 'w', encoding='utf-8') as f:
                f.write(rxinfer_code)
        except Exception as write_error:
            logger.error(f"Failed to write output file: {write_error}")
            return False, f"Error writing RxInfer.jl script: {write_error}", []
        
        message = f"Generated RxInfer.jl simulation script: {output_script_path}"
        warnings = []
        
        # Check for potential issues
        if not gnn_spec.get('initial_parameterization'):
            warnings.append("No initial parameterization found - using defaults")
        
        if not gnn_spec.get('model_parameters'):
            warnings.append("No model parameters found - using inferred dimensions")
        
        logger.info(f"Successfully generated RxInfer.jl script for {model_name}")
        return True, message, warnings
        
    except Exception as e:
        logger.error(f"Unexpected error in render_gnn_to_rxinfer: {e}", exc_info=True)
        return False, f"Error generating RxInfer.jl script: {e}", [] 