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
            num_states = gnn_spec.get('model_parameters', {}).get('num_hidden_states', 3)
            num_observations = gnn_spec.get('model_parameters', {}).get('num_obs', 3)
            
            # Validate parameters
            if not isinstance(num_states, int) or num_states < 1:
                num_states = 3
            if not isinstance(num_observations, int) or num_observations < 1:
                num_observations = 3
            
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
        num_actions = model_params.get('num_actions', 3) # Make sure this is correct key
        
        # Get initial parameterization if available
        initial_params = gnn_spec.get('initial_parameterization', {})
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
        elseif raw isa Vector && raw[1] isa Vector
            return hcat(raw...)
        end
    catch
    end
    return raw
end

function to_tensor(raw)
    try
        # B is [actions][prev][next] or similar. 
        # GNN Standard: B[action][next_state][prev_state] usually?
        # Let's assume input matches expected dimensions or is list of lists of lists
        if raw isa Array{{Float64, 3}}
            return raw
        elseif raw isa Vector && raw[1] isa Vector && raw[1][1] isa Vector
            # Dimensions: Action x Next x Prev ?
            # RxInfer expects: Next x Prev x Action (or similar, checking dims)
            # Let's construct generic 3D array
            n_actions = length(raw)
            n_next = length(raw[1])
            n_prev = length(raw[1][1])
            
            # Create tensor
            tensor = zeros(n_next, n_prev, n_actions)
            for a in 1:n_actions
                for n in 1:n_next
                    for p in 1:n_prev
                        tensor[n, p, a] = raw[a][n][p]
                    end
                end
            end
            return tensor
        end
    catch
    end
    return raw
end

A_matrix = to_matrix(A_raw)
B_matrix = to_tensor(B_raw) # Handling GNN B format
D_vector = Vector{{Float64}}(D_raw)

println("A matrix size: $(size(A_matrix))")
println("B matrix size: $(size(B_matrix))")
println("D vector size: $(size(D_vector))")


# --- RxInfer Model ---
# Fixed parameters version (active inference with known model)
@model function active_inference_model(observations, n_steps, A, B, D)
    
    # State sequence
    s = Vector{{Any}}(undef, n_steps)
    
    # Initial state
    s_init ~ Categorical(D)
    s[1] = s_init
    
    # First observation
    observations[1] ~ Categorical(s[1], copy(A))
    
    # State transitions and observations
    for t in 2:n_steps
        # Action selection (Random policy for now)
        action_idx = rand(1:NUM_ACTIONS)
        
        # State transition 
        # B is [Next, Prev, Action]
        # We slice B by action to get a transition matrix B_a
        B_a = B[:, :, action_idx] 
        s_next ~ DiscreteTransition(s[t-1], copy(B_a)) 
        s[t] = s_next
        
        # Observation
        # Try Categorical with two arguments (aliased to DiscreteTransition or similar?)
        observations[t] ~ Categorical(s[t], copy(A))
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
    
    # Run Inference
    result = infer(
        model = active_inference_model(n_steps=TIME_STEPS, A=A_matrix, B=B_matrix, D=D_vector),
        data = (observations = real_obs,),
        iterations = 10 # Not needed for exact inference in discrete case usually, but good for stability if loops
    )
    
    println("Inference complete.")
    return result, real_states, real_obs
end

# --- Main ---
function main()
    try
        result, true_states, obs = run_simulation()
        
        # Visualize
        posteriors = result.posteriors[:s]
        
        # Extract belief trace for state 1 over time
        belief_trace = [pdf(posteriors[t], 1) for t in 1:TIME_STEPS]
        
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