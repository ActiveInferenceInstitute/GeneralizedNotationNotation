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
            return True, "Successfully generated RxInfer.jl simulation code"

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

            # Extract num_timesteps from model parameters (default 20 for backward compat)
            num_timesteps = model_params.get('num_timesteps', 20)

            # Validate parameters
            if not isinstance(num_states, int) or num_states < 1:
                num_states = 3
            if not isinstance(num_observations, int) or num_observations < 1:
                num_observations = 3
            if not isinstance(num_actions, int) or num_actions < 1:
                num_actions = 3
            if not isinstance(num_timesteps, int) or num_timesteps < 1:
                num_timesteps = 20

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
            code = code.replace('{num_timesteps}', str(num_timesteps))

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

        # Extract num_timesteps from model parameters (default 20 for backward compat)
        num_timesteps = model_params.get('num_timesteps', 20)

        # Extract action_precision from GNN ModelParameters (RX-3: was hardcoded 4.0)
        action_precision = model_params.get('action_precision', model_params.get('gamma', 4.0))
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
    Pkg.add(["RxInfer", "Distributions", "LinearAlgebra", "Random", "StatsBase"])
catch e
    println("⚠️  Package install error (might be already installed): $e")
end

using RxInfer
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON

Random.seed!(42)

# --- Model Parameters ---
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = {num_timesteps}

# Parameter Matrices (from GNN)
# We use raw Vector of Vectors and convert to Matrix/Tensor for RxInfer
A_raw = {A_data if A_data else "fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS, NUM_STATES)"}
B_raw = {B_data if B_data else "fill(1.0/NUM_STATES, NUM_STATES, NUM_STATES, NUM_ACTIONS)"}
C_raw = {C_data if C_data else "fill(0.0, NUM_OBSERVATIONS)"}
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
        @warn "to_matrix conversion failed" exception=e
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
                # 3D case: each element is a matrix (list of rows)
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
            elseif first_action[1] isa Number
                # 2D case: each element is a flat vector (row of transition matrix)
                # This is a passive model (HMM/Markov Chain) with no action dimension
                println("ℹ️  B is 2D (passive model) — expanding to 3D with single action")
                n_rows = length(arr)
                n_cols = length(first_action)
                mat = zeros(n_rows, n_cols)
                for r in 1:n_rows
                    row_data = collect(arr[r])
                    for c in 1:n_cols
                        mat[r, c] = row_data[c]
                    end
                end
                # Normalize columns
                for c in 1:n_cols
                    cs = sum(mat[:, c])
                    if cs > 0
                        mat[:, c] ./= cs
                    end
                end
                # Expand to 3D with single action
                return reshape(mat, n_rows, n_cols, 1)
            end
        end
    catch e
        @warn "to_tensor conversion failed" exception=e
        println("to_tensor warning: $e")
    end
    return raw
end

A_matrix = to_matrix(A_raw)
B_matrix = to_tensor(B_raw) # Handling GNN B format
C_vector = Vector{{Float64}}(collect(C_raw))
D_vector = Vector{{Float64}}(collect(D_raw))

# Normalize D_vector to ensure it sums exactly to 1.0 (required for Categorical)
D_vector = D_vector ./ sum(D_vector)

# Normalize A_matrix columns (each column should sum to 1)
for j in 1:size(A_matrix, 2)
    A_matrix[:, j] = A_matrix[:, j] ./ sum(A_matrix[:, j])
end

# Softmax utility function
function softmax(x)
    ex = exp.(x .- maximum(x))
    return ex ./ sum(ex)
end

# Convert C (log-preferences) to preferred observation distribution
C_preferred = softmax(C_vector)

println("A matrix size: $(size(A_matrix))")
println("B matrix size: $(size(B_matrix))")
println("C vector (preferences): $C_vector")
println("C preferred (softmax): $C_preferred")
println("D vector size: $(size(D_vector))")


# --- RxInfer Single-Step Inference Model ---
# Used for belief updating given a single observation
@model function belief_update_model(observation, A, prior)
    # State prior from previous belief
    s ~ Categorical(prior)
    # Observation likelihood
    observation ~ DiscreteTransition(s, A)
    return s
end

# --- Expected Free Energy (EFE) Computation ---
# G(a) = ambiguity + risk
# ambiguity: expected uncertainty about observations given predicted states
# risk: KL divergence between expected observations and preferred observations
function compute_efe(belief, action_idx, A, B, C_pref)
    # Predicted next state distribution: s' = B[:,:,a] * belief
    B_a = B[:, :, action_idx]
    predicted_state = B_a * belief
    
    # Normalize (handle numerical issues)
    predicted_state = max.(predicted_state, 1e-16)
    predicted_state = predicted_state ./ sum(predicted_state)
    
    # Expected observation distribution: o' = A * s'
    predicted_obs = A * predicted_state
    predicted_obs = max.(predicted_obs, 1e-16)
    predicted_obs = predicted_obs ./ sum(predicted_obs)
    
    # Ambiguity: expected entropy of observations conditioned on states
    # H[P(o|s)] weighted by predicted state
    ambiguity = 0.0
    for j in 1:length(predicted_state)
        if predicted_state[j] > 1e-16
            # Entropy of column j of A (observation distribution for state j)
            col = A[:, j]
            col = max.(col, 1e-16)
            ambiguity -= predicted_state[j] * sum(col .* log.(col))
        end
    end
    
    # Risk: KL divergence D_KL(predicted_obs || C_preferred)
    C_safe = max.(C_pref, 1e-16)
    risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(C_safe)))
    
    # EFE = ambiguity + risk (lower is better)
    return ambiguity + risk
end

# --- Active Inference Action Selection ---
function select_action(belief, A, B, C_pref; action_precision={action_precision})
    n_actions = size(B, 3)
    efe_values = zeros(n_actions)
    
    for a in 1:n_actions
        efe_values[a] = compute_efe(belief, a, A, B, C_pref)
    end
    
    # Policy via softmax over negative EFE (lower EFE = higher probability)
    neg_efe = -action_precision .* efe_values
    action_probs = softmax(neg_efe)
    
    # Sample action from policy
    action = rand(Categorical(action_probs))
    
    return action, efe_values, action_probs
end

# --- One-hot encoding ---
function one_hot(idx, n)
    v = zeros(n)
    v[idx] = 1.0
    return v
end

# --- Active Inference Simulation Loop ---
function run_simulation()
    println("\\n🧠 Running Active Inference simulation with EFE-based action selection...")
    
    # Storage
    true_states = Vector{{Int}}(undef, TIME_STEPS)
    observations = Vector{{Int}}(undef, TIME_STEPS)
    actions = Vector{{Int}}(undef, TIME_STEPS)
    beliefs = Vector{{Vector{{Float64}}}}(undef, TIME_STEPS)
    efe_history = Vector{{Vector{{Float64}}}}(undef, TIME_STEPS)
    action_probs_history = Vector{{Vector{{Float64}}}}(undef, TIME_STEPS)
    
    # Initialize environment
    current_state = rand(Categorical(D_vector))
    current_belief = copy(D_vector)
    
    for t in 1:TIME_STEPS
        # 1. Environment generates observation
        true_states[t] = current_state
        obs = rand(Categorical(A_matrix[:, current_state]))
        observations[t] = obs
        
        # 2. Infer beliefs using RxInfer (single-step Bayesian inference)
        obs_one_hot = one_hot(obs, NUM_OBSERVATIONS)
        try
            result = infer(
                model = belief_update_model(A=A_matrix, prior=current_belief),
                data = (observation = obs_one_hot,),
                iterations = 5
            )
            # Extract posterior belief
            posterior = result.posteriors[:s]
            final_posterior = posterior[end]
            current_belief = probvec(final_posterior)
        catch e
            # Fallback: manual Bayesian update if RxInfer fails
            println("  Step $t: RxInfer inference fallback - $e")
            likelihood = A_matrix[obs, :]
            unnormalized = current_belief .* likelihood
            current_belief = unnormalized ./ sum(unnormalized)
        end
        
        # Ensure belief is valid
        current_belief = max.(current_belief, 1e-16)
        current_belief = current_belief ./ sum(current_belief)
        beliefs[t] = copy(current_belief)
        
        # 3. Compute EFE and select action (Active Inference!)
        action, efe_values, action_probs = select_action(
            current_belief, A_matrix, B_matrix, C_preferred
        )
        actions[t] = action
        efe_history[t] = copy(efe_values)
        action_probs_history[t] = copy(action_probs)
        
        # 4. Environment transitions based on selected action
        next_probs = B_matrix[:, current_state, action]
        next_probs = max.(next_probs, 1e-16)
        next_probs = next_probs ./ sum(next_probs)
        current_state = rand(Categorical(next_probs))
        
        # 5. Update belief for next timestep (predictive prior)
        B_a = B_matrix[:, :, action]
        current_belief = B_a * current_belief
        current_belief = max.(current_belief, 1e-16)
        current_belief = current_belief ./ sum(current_belief)
        
        println("  Step $t: obs=$obs, action=$action, belief_max=$(round(maximum(beliefs[t]), digits=3)), EFE=$(round.(efe_values, digits=3))")
    end
    
    println("\\n✅ Active Inference simulation complete")
    println("Action distribution: ", StatsBase.countmap(actions))
    
    # Compute per-step EFE of selected action
    selected_efe = [efe_history[t][actions[t]] for t in 1:TIME_STEPS]
    
    # Save results
    results_data = Dict(
        "framework" => "rxinfer",
        "model_name" => "{model_display_name}",
        "time_steps" => TIME_STEPS,
        "true_states" => true_states,
        "observations" => observations,
        "actions" => actions,
        "beliefs" => beliefs,
        "efe_history" => selected_efe,
        "efe_per_action" => efe_history,
        "action_probabilities" => action_probs_history,
        "num_states" => NUM_STATES,
        "num_observations" => NUM_OBSERVATIONS,
        "num_actions" => NUM_ACTIONS,
        "preferences" => C_vector,
        "validation" => Dict(
            "all_beliefs_valid" => all(b -> all(x -> 0.0 <= x <= 1.0, b), beliefs),
            "beliefs_sum_to_one" => all(b -> abs(sum(b) - 1.0) < 0.01, beliefs),
            "actions_in_range" => all(a -> 1 <= a <= NUM_ACTIONS, actions)
        )
    )
    
    open("simulation_results.json", "w") do f
        JSON.print(f, results_data, 4)
    end
    println("✅ Standardized results saved to simulation_results.json")
    
    return beliefs, actions, efe_history
end

# --- Main ---
function main()
    try
        beliefs, actions, efe_hist = run_simulation()
        
        println("✅ RxInfer Active Inference simulation successful")
        println("📊 Visualizations will be generated by the analysis step")
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
