#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: Active Inference Chronic Pain Multi-Theory Model v1

Generated from GNN specification.
This model integrates multiple coherent theories of chronic pain mechanisms across THREE NESTED CONTINUOUS TIMESCALES:
**Multi-Theory Integration:**
- Peripheral Sensitization: Enhanced nociceptor responsiveness and reduced thresholds (slow timescale)
- Central Sensitization: Amplified CNS processing and reduced inhibition (slow timescale, one-way process)
- Gate Control Theory: Spinal modulation of ascending pain signals (fast timescale)
- Neuromatrix Theory: Distributed network generating pain experience (fast-medium coupling)
- Predictive Coding: Pain as precision-weighted prediction error (all timescales)
- Biopsychosocial Integration: Cognitive, emotional, and behavioral factors (medium timescale)
**Three Nested Timescales:**
1. Fast (ms-s): Neural signaling, gate control, descending modulation, acute pain perception
2. Medium (min-hrs): Cognitive-affective processes, behavioral strategies, functional capacity
3. Slow (hrs-days): Tissue healing, peripheral/central sensitization, chronic adaptations
**State Space Structure:**
- Six hidden state factors (378 combinations): tissue state (slow), peripheral sensitivity (slow), spinal gate (fast), central sensitization (slow), descending modulation (fast), cognitive-affective state (medium)
- Four observation modalities (72 outcomes): pain intensity (fast), pain quality (fast), functional capacity (medium), autonomic response (fast)
- Four control factors (81 actions): attention allocation (medium), behavioral strategy (medium), cognitive reappraisal (medium), descending control (fast)
**Key Features:**
- Timescale separation: ε (fast/medium) ≈ 10^-3, δ (medium/slow) ≈ 10^-2
- Cross-timescale coupling: slow states modulate fast dynamics, fast observations (averaged) drive medium cognition, medium behaviors (averaged) influence slow healing
- Testable predictions about pain chronification pathways across multiple timescales
- Intervention targets at each timescale: fast (descending control), medium (CBT/behavioral), slow (prevent sensitization)

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
const MODEL_NAME = "Active Inference Chronic Pain Multi-Theory Model v1"
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
A_matrix = [0.7 0.25 0.05 0.0 0.8 0.15 0.05 0.9 0.08 0.02 0.85 0.15; 0.6 0.3 0.1 0.0 0.75 0.2 0.05 0.85 0.1 0.05 0.8 0.2; 0.45 0.35 0.15 0.05 0.7 0.2 0.1 0.75 0.15 0.1 0.7 0.3; 0.2 0.3 0.35 0.15 0.5 0.3 0.2 0.5 0.3 0.2 0.4 0.6; amplified pain; 0.3 0.4 0.25 0.05 0.6 0.25 0.15 0.6 0.25 0.15 0.5 0.5; 0.4 0.35 0.2 0.05 0.65 0.25 0.1 0.7 0.2 0.1 0.6 0.4; 0.55 0.3 0.12 0.03 0.7 0.22 0.08 0.8 0.15 0.05 0.75 0.25; 0.8 0.15 0.05 0.0 0.85 0.12 0.03 0.95 0.04 0.01 0.9 0.1; 0.75 0.2 0.05 0.0 0.8 0.15 0.05 0.9 0.08 0.02 0.85 0.15; 0.65 0.25 0.08 0.02 0.75 0.18 0.07 0.85 0.12 0.03 0.8 0.2; 0.9 0.08 0.02 0.0 0.9 0.08 0.02 0.98 0.02 0.0 0.95 0.05; gate closed; 0.85 0.12 0.03 0.0 0.88 0.1 0.02 0.95 0.04 0.01 0.92 0.08; 0.05 0.15 0.45 0.35 0.3 0.4 0.3 0.2 0.4 0.4 0.25 0.75; 0.02 0.1 0.4 0.48 0.2 0.35 0.45 0.1 0.35 0.55 0.15 0.85; 0.35 0.4 0.2 0.05 0.7 0.25 0.05 0.7 0.2 0.1 0.75 0.25; 0.2 0.35 0.3 0.15 0.55 0.3 0.15 0.55 0.3 0.15 0.6 0.4; 0.15 0.3 0.4 0.15 0.4 0.35 0.25 0.4 0.35 0.25 0.45 0.55; 0.05 0.2 0.45 0.3 0.3 0.4 0.3 0.3 0.4 0.3 0.35 0.65; 0.4 0.4 0.15 0.05 0.75 0.2 0.05 0.8 0.15 0.05 0.8 0.2; 0.25 0.4 0.25 0.1 0.65 0.25 0.1 0.65 0.25 0.1 0.7 0.3; 0.2 0.35 0.3 0.15 0.55 0.3 0.15 0.5 0.3 0.2 0.6 0.4; 0.08 0.25 0.4 0.27 0.4 0.35 0.25 0.35 0.4 0.25 0.45 0.55; 0.05 0.2 0.4 0.35 0.3 0.4 0.3 0.25 0.45 0.3 0.35 0.65; 0.02 0.1 0.35 0.53 0.2 0.35 0.45 0.15 0.45 0.4 0.2 0.8; 0.02 0.08 0.3 0.6 0.15 0.3 0.55 0.1 0.4 0.5 0.15 0.85; all risk factors; 0.1 0.3 0.4 0.2 0.5 0.35 0.15 0.45 0.35 0.2 0.5 0.5; 0.05 0.25 0.45 0.25 0.4 0.4 0.2 0.4 0.4 0.2 0.45 0.55; 0.03 0.15 0.42 0.4 0.3 0.4 0.3 0.3 0.45 0.25 0.35 0.65; 0.01 0.08 0.35 0.56 0.2 0.35 0.45 0.2 0.45 0.35 0.25 0.75; 0.01 0.05 0.3 0.64 0.15 0.3 0.55 0.15 0.5 0.35 0.2 0.8; 0.0 0.02 0.2 0.78 0.1 0.25 0.65 0.08 0.45 0.47 0.1 0.9; 0.7 0.25 0.05 0.0 0.8 0.15 0.05 0.8 0.15 0.05 0.85 0.15; 0.55 0.35 0.08 0.02 0.75 0.2 0.05 0.75 0.2 0.05 0.8 0.2; 0.15 0.3 0.4 0.15 0.45 0.35 0.2 0.5 0.35 0.15 0.55 0.45; 0.05 0.2 0.45 0.3 0.3 0.4 0.3 0.35 0.4 0.25 0.4 0.6]
println("✅ A matrix created: $(size(A_matrix))")

# B matrix (transition model) 
B_matrix = []
println("✅ B matrix created: $(size(B_matrix))")

# C vector (preferences)
C_vector = [2.2, 1.4, 0.5, 0.7, -0.1, -0.8, -0.5, -1.3, -2.2, -1.7, -2.5, -3.4, 2.0, 1.2, 0.3, 0.5, -0.3, -1.0, -0.7, -1.5, -2.4, -1.9, -2.7, -3.6, 1.6, 0.8, -0.1, 0.1, -0.7, -1.4, -1.1, -1.9, -2.8, -2.3, -3.1, -4.0, 1.0, 0.2, -0.7, -0.5, -1.3, -2.0, -1.7, -2.5, -3.4, -2.9, -3.7, -4.6, 0.5, -0.3, -1.2, -1.0, -1.8, -2.5, -2.2, -3.0, -3.9, -3.4, -4.2, -5.1, -0.2, -1.0, -1.9, -1.7, -2.5, -3.2, -2.9, -3.7, -4.6, -4.1, -4.9, -5.8]
println("✅ C vector created: $(length(C_vector))")

# D vector (prior beliefs)
D_vector = [0.1 0.5 0.4 # T distribution (healed inflamed damaged) 0.60 0.3 0.1 # P_sens distribution (normal moderate severe) 0.40 0.4 0.2 # G distribution (open modulated closed) 0.90 0.1 # C_sens distribution (absent present) 0.20 0.6 0.2 # D_mod distribution (facilitation neutral inhibition) 0.30 0.2 0.15 0.1 0.1 0.1 0.05; adaptive → alexithymic]
println("✅ D vector created: $(length(D_vector))")

# E vector (policy priors)
E_vector = [0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015, 0.03, 0.045, 0.03, 0.05, 0.075, 0.05, 0.03, 0.045, 0.03, 0.015, 0.023, 0.015, 0.025, 0.038, 0.025, 0.015, 0.023, 0.015, 0.01, 0.015, 0.01, 0.017, 0.025, 0.017, 0.01, 0.015, 0.01, 0.02, 0.03, 0.02, 0.033, 0.05, 0.033, 0.02, 0.03, 0.02, 0.01, 0.015, 0.01, 0.017, 0.025, 0.017, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006, 0.012, 0.018, 0.012, 0.02, 0.03, 0.02, 0.012, 0.018, 0.012, 0.006, 0.009, 0.006, 0.01, 0.015, 0.01, 0.006, 0.009, 0.006]
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
