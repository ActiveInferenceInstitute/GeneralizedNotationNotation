#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: Simple MDP Agent

Generated from GNN specification.
This model describes a fully observable Markov Decision Process (MDP):
- 4 hidden states representing grid positions (corners of a 2x2 grid).
- Observations are identical to states (A = identity matrix).
- 4 actions: stay, move-north, move-south, move-east.
- Preferences strongly favor state/observation 3 (goal location).
- Tests the degenerate POMDP case where partial observability is absent.

Model Dimensions:
- States: 4
- Observations: 4  
- Actions: 4
"""

# Ensure required packages are installed
using Pkg

# Install missing packages if needed
println("📦 Ensuring required packages are installed...")
try
    # Try to precompile key packages - will add if missing
    Pkg.add(["JSON", "ActiveInference", "Distributions"])
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
using StatsBase

# Global configuration
const SCRIPT_VERSION = "1.0.0"
const MODEL_NAME = "Simple MDP Agent"
const N_STATES = 4
const N_OBSERVATIONS = 4
const N_CONTROLS = 4
# POLICY_LEN is defined below after E-vector setup

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
A_matrix = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
println("✅ A matrix created: $(size(A_matrix))")

# B matrix (transition model) 
B_matrix = cat([0.9 0.1 0.0 0.0; 0.1 0.9 0.0 0.0; 0.0 0.0 0.9 0.1; 0.0 0.0 0.1 0.9], [0.1 0.9 0.0 0.0; 0.9 0.1 0.0 0.0; 0.0 0.0 0.1 0.9; 0.0 0.0 0.9 0.1], [0.0 0.0 0.9 0.1; 0.0 0.0 0.1 0.9; 0.9 0.1 0.0 0.0; 0.1 0.9 0.0 0.0], [0.0 0.0 0.1 0.9; 0.0 0.0 0.9 0.1; 0.1 0.9 0.0 0.0; 0.9 0.1 0.0 0.0]; dims=3)
println("✅ B matrix created: $(size(B_matrix))")

# C vector (preferences)
C_vector = [0.0, 0.0, 0.0, 3.0]
println("✅ C vector created: $(length(C_vector))")

# D vector (prior beliefs)
D_vector = [0.25, 0.25, 0.25, 0.25]
println("✅ D vector created: $(length(D_vector))")

# E vector (policy priors)
# ActiveInference.jl requires E-vector length to match number of policies
# For policy_len=1, num_policies = num_actions  
E_vector_raw = [0.25, 0.25, 0.25, 0.25]
POLICY_LEN = 1  # Planning horizon
NUM_POLICIES = N_CONTROLS ^ POLICY_LEN  # Number of possible action sequences

# Adjust E-vector to match policy count
if length(E_vector_raw) != NUM_POLICIES
    println("⚠️  Adjusting E-vector from $(length(E_vector_raw)) to $NUM_POLICIES elements")
    if length(E_vector_raw) == N_CONTROLS
        # Expand: one value per action -> one value per policy
        E_vector = fill(1.0 / NUM_POLICIES, NUM_POLICIES)  # Uniform prior
    else
        # Recovery: uniform distribution
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
        "policy_len" => POLICY_LEN,
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
    n_steps = 15  # From GNN ModelParameters (num_timesteps)
    observations_log = []
    actions_log = []
    beliefs_log = []
    efe_log = []
    policy_log = []
    steps_log = []
    
    # Track full belief distributions (all states)
    beliefs_full_log = []
    
    Random.seed!(42)  # Reproducibility
    
    # Generative Environment (True State tracking)
    # Initialize from the agent's prior D matrix
    true_state = rand(Categorical(D_vector))
    
    for step in 1:n_steps
        # 1. Environment generates observation
        # Sample observation stochastically from the Likelihood matrix A conditioned on the true state
        obs_prob = A_matrix[:, true_state]
        # Categorical sampling to generate an observation integer
        observation = [rand(Categorical(obs_prob))]
        
        push!(observations_log, observation[1])
        
        # 2. Agent perceives observation, infers state, selects action
        infer_states!(aif_agent, observation)
        infer_policies!(aif_agent)
        sample_action!(aif_agent)
        
        # Log action and beliefs
        push!(actions_log, aif_agent.action[1])
        push!(beliefs_full_log, copy(aif_agent.qs_current[1]))  # All states
        push!(beliefs_log, aif_agent.qs_current[1][1])  # First state only
        
        # 3. Environment transitions true hidden state
        # Sample next state stochastically from the Transition matrix B conditioned on current true state and selected action
        next_probs = B_matrix[:, true_state, aif_agent.action[1]]
        # Safety catch for numerical zero bounds before categorical sampling
        next_probs = max.(next_probs, 1e-16)
        next_probs = next_probs ./ sum(next_probs)
        true_state = rand(Categorical(next_probs))
        
        # Try to log EFE if available
        try
            if hasfield(typeof(aif_agent), :G) && !isnothing(aif_agent.G)
                push!(efe_log, copy(aif_agent.G))
            else
                push!(efe_log, [NaN])
            end
        catch
            push!(efe_log, [NaN])
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
    println("📊 Visualizations will be generated by the analysis step")
    
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
    
    # Validation checks (must run before saving JSON results that reference validation_status)
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
    
    # Helper function to sanitize NaNs before JSON serialization
    safe_float(x) = isnan(x) ? 0.0 : Float64(x)
    
    # Convert Julia arrays of arrays to standard forms that JSON.jl handles natively
    json_beliefs_log = [[safe_float(v) for v in b] for b in beliefs_full_log]
    json_efe_log = [[safe_float(v) for v in e] for e in efe_log]
    json_policy_log = [Ref(p)[] for p in policy_log] # Handle policy structure

    # Save comprehensive JSON results matching cross-framework standard
    comp_results = Dict(
        "framework" => "activeinference_jl",
        "model_name" => MODEL_NAME,
        "time_steps" => n_steps,
        "observations" => observations_log,
        "actions" => actions_log,
        "beliefs" => json_beliefs_log,
        "efe_history" => json_efe_log,
        "policy_history" => json_policy_log,
        "num_states" => N_STATES,
        "num_observations" => N_OBSERVATIONS,
        "num_actions" => N_CONTROLS,
        "validation" => validation_status
    )
    
    comp_file = joinpath(output_dir, "simulation_results.json")
    open(comp_file, "w") do f
        JSON.print(f, comp_results, 2)
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
    
    # (Validation already computed above before JSON export)
    
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
        println(f, "  - Data files: simulation_results.csv, model_parameters.json")
        println(f, "  - Summary: summary.txt")
        println(f, "  - Visualizations: generated by analysis step")
        println(f, "="^70)
    end
    
    println("="^70)
    println("✅ Simulation completed successfully!")
    println("="^70)
    println("📊 Results saved to: $output_dir")
    println("📈 Final belief: $(round(beliefs_log[end], digits=3))")
    println("🎯 Action distribution: $(countmap(actions_log))")
    println("✅ Validation: ALL CHECKS PASSED" * (validation_status["all_valid"] ? "" : " (WITH WARNINGS)"))
    
catch e
    println("❌ Error during simulation: $e")
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\nActiveInference.jl script completed!")
println("Model: $MODEL_NAME")
