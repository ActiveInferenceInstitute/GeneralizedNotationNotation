#!/usr/bin/env julia

# RxInfer.jl Active Inference POMDP Simulation
# Generated from GNN Model: {model_name}
# Generated: {timestamp}

# Ensure required packages are installed
using Pkg

# Install missing packages if needed
println("📦 Ensuring required packages are installed...")
try
    Pkg.add(["RxInfer", "Distributions", "StatsBase", "Plots", "JSON", "LinearAlgebra", "Random"])
    println("✅ Package installation complete")
catch e
    println("⚠️  Some packages may need manual installation: $e")
end

using RxInfer
using Distributions
using StatsBase
using Plots
using JSON
using LinearAlgebra
using Random

Random.seed!(42)

println("============================================================")
println("RxInfer.jl Active Inference POMDP - GNN Generated")
println("Model: {model_name}")
println("============================================================")

# Model parameters from GNN specification
const NUM_STATES = {num_states}
const NUM_OBSERVATIONS = {num_observations}
const NUM_ACTIONS = {num_actions}
const TIME_STEPS = {num_timesteps}

println("📊 State Space: $NUM_STATES states, $NUM_OBSERVATIONS observations, $NUM_ACTIONS actions")
println("⏱️  Time Steps: $TIME_STEPS")

# Initialize parameter matrices with proper dimensions
# A: Observation model P(o|s) - shape (obs, states)
A_matrix = fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS, NUM_STATES)
for i in 1:min(NUM_STATES, NUM_OBSERVATIONS)
    A_matrix[i, i] = 0.8  # Higher probability for matching state-observation
end
# Normalize columns to sum to 1
for j in 1:NUM_STATES
    A_matrix[:, j] ./= sum(A_matrix[:, j])
end

# B: Transition model P(s'|s, a) - shape (states, states, actions)
B_matrix = zeros(NUM_STATES, NUM_STATES, NUM_ACTIONS)
for a in 1:NUM_ACTIONS
    for s in 1:NUM_STATES
        # Create action-dependent transitions
        next_s = mod1(s + a - 1, NUM_STATES)  # Action shifts state
        B_matrix[next_s, s, a] = 0.8
        # Add some uncertainty
        for other_s in 1:NUM_STATES
            if other_s != next_s
                B_matrix[other_s, s, a] = 0.2 / (NUM_STATES - 1)
            end
        end
    end
end

# D: Prior over initial states (uniform)
D_vector = fill(1.0/NUM_STATES, NUM_STATES)

println("✅ A matrix size: $(size(A_matrix))")
println("✅ B matrix size: $(size(B_matrix))")
println("✅ D vector size: $(size(D_vector))")

# POMDP model with action-conditioned transitions
@model function pomdp_active_inference(observations, n_steps, A, B, D)
    # State sequence
    states = Vector(undef, n_steps)
    
    # Initial state with prior
    states[1] ~ Categorical(D)
    
    # First observation conditioned on initial state
    observations[1] ~ DiscreteTransition(states[1], A)
    
    # Subsequent time steps with action-conditioned transitions
    for t in 2:n_steps
        # Select action (for inference, we use random policy)
        action_idx = rand(1:NUM_ACTIONS)
        
        # B slice for this action
        B_a = B[:, :, action_idx]
        
        # State transition conditioned on action
        states[t] ~ DiscreteTransition(states[t-1], B_a)
        
        # Observation conditioned on current state
        observations[t] ~ DiscreteTransition(states[t], A)
    end
    
    return states
end

# Run simulation and inference
function run_simulation()
    println("\n🚀 Generating synthetic POMDP trajectory...")
    
    # Generate ground truth trajectory
    real_states = Vector{Int}(undef, TIME_STEPS)
    real_obs = Vector{Int}(undef, TIME_STEPS)
    real_actions = Vector{Int}(undef, TIME_STEPS)
    
    # Initial state
    current_state = rand(Categorical(D_vector))
    real_states[1] = current_state
    real_obs[1] = rand(Categorical(A_matrix[:, current_state]))
    real_actions[1] = 1
    
    # Generate trajectory with random policy
    for t in 2:TIME_STEPS
        action = rand(1:NUM_ACTIONS)
        real_actions[t] = action
        
        # State transition
        B_a = B_matrix[:, current_state, action]
        current_state = rand(Categorical(B_a))
        real_states[t] = current_state
        
        # Observation
        real_obs[t] = rand(Categorical(A_matrix[:, current_state]))
    end
    
    println("📋 Observations: $real_obs")
    println("🎯 True States: $real_states")
    println("🎮 Actions: $real_actions")
    
    # One-hot encode observations for RxInfer
    function one_hot(idx, n)
        v = zeros(n)
        v[idx] = 1.0
        return v
    end
    
    obs_one_hot = [one_hot(o, NUM_OBSERVATIONS) for o in real_obs]
    
    # Run Bayesian inference
    println("\n🔮 Running Bayesian state inference...")
    
    result = infer(
        model = pomdp_active_inference(n_steps=TIME_STEPS, A=A_matrix, B=B_matrix, D=D_vector),
        data = (observations = obs_one_hot,),
        iterations = 20
    )
    
    println("✅ Inference completed!")
    
    # Extract posterior beliefs
    all_iterations = result.posteriors[:states]
    final_iteration = all_iterations[end]
    beliefs = [probvec(final_iteration[t]) for t in 1:TIME_STEPS]
    
    # Calculate belief confidence (max probability)
    belief_confidence = [maximum(b) for b in beliefs]
    
    println("\n📊 Belief Analysis:")
    println("   Mean confidence: $(round(mean(belief_confidence), digits=3))")
    println("   Final belief: $(round.(beliefs[end], digits=3))")
    
    # Save standardized results
    results_data = Dict(
        "framework" => "rxinfer",
        "model_name" => "{model_name}",
        "time_steps" => TIME_STEPS,
        "num_states" => NUM_STATES,
        "num_observations" => NUM_OBSERVATIONS,
        "num_actions" => NUM_ACTIONS,
        "true_states" => real_states,
        "observations" => real_obs,
        "actions" => real_actions,
        "beliefs" => beliefs,
        "belief_confidence" => belief_confidence,
        "validation" => Dict(
            "beliefs_valid" => all(b -> all(0 .<= b .<= 1), beliefs),
            "beliefs_sum_to_one" => all(b -> isapprox(sum(b), 1.0, atol=0.01), beliefs)
        )
    )
    
    open("simulation_results.json", "w") do f
        JSON.print(f, results_data, 4)
    end
    println("💾 Results saved to simulation_results.json")
    
    return result, real_states, real_obs, beliefs
end

# Visualization
function create_visualization(beliefs, real_states)
    println("\n🎨 Creating visualization...")
    
    # Extract belief for state 1 over time
    belief_traces = hcat(beliefs...)'  # TIME_STEPS x NUM_STATES
    
    p = plot(1:TIME_STEPS, belief_traces,
             xlabel="Time Step", ylabel="Posterior Probability",
             title="RxInfer.jl POMDP State Inference\nModel: {model_name}",
             label=hcat(["State $i" for i in 1:NUM_STATES]...),
             linewidth=2, marker=:circle, markersize=4,
             legend=:best, size=(1000, 600), dpi=150)
    
    # Mark true states
    for t in 1:TIME_STEPS
        scatter!([t], [1.0], marker=:star, markersize=8,
                color=real_states[t] == 1 ? :green : :gray, alpha=0.5, label="")
    end
    
    savefig(p, "rxinfer_results.png")
    println("💾 Visualization saved to rxinfer_results.png")
end

# Main execution
function main()
    try
        result, true_states, obs, beliefs = run_simulation()
        create_visualization(beliefs, true_states)
        
        println("\n" * "="^60)
        println("🎉 RxInfer.jl POMDP simulation completed successfully!")
        println("="^60)
        return 0
    catch e
        println("\n❌ Simulation failed: $e")
        showerror(stdout, e, catch_backtrace())
        return 1
    end
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
