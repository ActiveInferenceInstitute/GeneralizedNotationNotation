#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: Active Inference Neural Response Model v1

Generated from GNN specification.
This model describes how a neuron responds to stimuli using Active Inference principles:
- One primary observation modality (firing_rate) with 4 possible activity levels
- Two auxiliary observation modalities (postsynaptic_potential, calcium_signal) for comprehensive monitoring
- Five hidden state factors representing different aspects of neural computation
- Three control factors for plasticity, channel modulation, and metabolic allocation
- The model captures key neural phenomena: membrane potential dynamics, synaptic plasticity (STDP-like), activity-dependent adaptation, homeostatic regulation, and metabolic constraints
- Preferences encode biologically realistic goals: stable firing rates, energy efficiency, and synaptic balance

Model Dimensions:
- States: 3
- Observations: 3  
- Actions: 3
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
const MODEL_NAME = "Active Inference Neural Response Model v1"
const N_STATES = 3
const N_OBSERVATIONS = 3
const N_CONTROLS = 3
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
A_matrix = [0.05 0.15 0.25 0.55 0.4 0.4 0.2 0.1 0.35 0.55 0.3 0.45; 0.1 0.2 0.3 0.4 0.35 0.45 0.2 0.15 0.4 0.45 0.25 0.4; 0.15 0.25 0.35 0.25 0.3 0.5 0.2 0.2 0.45 0.35 0.2 0.35]
println("✅ A matrix created: $(size(A_matrix))")

# B matrix (transition model) 
B_matrix = []
println("✅ B matrix created: $(size(B_matrix))")

# C vector (preferences)
C_vector = [0.1, 0.2, 0.4, 0.3, 0.15, 0.35, 0.5, 0.25, 0.35, 0.4, 0.25, 0.2]
println("✅ C vector created: $(length(C_vector))")

# D vector (prior beliefs)
D_vector = [0.05 0.15 0.35 0.35 0.1 # V_m distribution 0.20 0.4 0.3 0.1 # W distribution 0.40 0.4 0.2 # A distribution 0.20 0.6 0.2 # H distribution 0.15 0.7 0.15]
println("✅ D vector created: $(length(D_vector))")

# E vector (policy priors)
E_vector = [0.2, 0.3, 0.5, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.3, 0.4, 0.3, 0.25, 0.5, 0.25, 0.3, 0.4, 0.3, 0.35, 0.4, 0.25, 0.3, 0.45, 0.25, 0.35, 0.4, 0.25]
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

println("\nActiveInference.jl script completed!")
println("Model: $MODEL_NAME")
