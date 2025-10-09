# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: Active Inference Chronic Pain Multi-Theory Model v1
# Generated: 2025-10-09 07:55:54

using RxInfer
using Distributions
using LinearAlgebra
using Plots
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Model parameters extracted from GNN specification
const NUM_STATES = 378
const NUM_OBSERVATIONS = 72
const NUM_ACTIONS = 81
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
    
    # State sequence - use randomvar() for proper RxInfer variable creation
    s = randomvar(n_steps)
    
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
    obs = Vector{Int}(undef, n_steps)
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
    println("\n🚀 Starting Active Inference simulation...")
    
    # Generate observations
    observations_data = generate_observations(TIME_STEPS)
    println("📋 Generated observation sequence: $observations_data")
    
    # Create data for inference
    data = (observations = observations_data,)
    
    # Perform inference
    println("\n🧠 Running variational inference...")
    result = infer(
        model = active_inference_model(n_steps=TIME_STEPS, observations=observations_data),
        data = data,
        options = (
            iterations = 50,
            showprogress = true,
            free_energy = true
        )
    )
    
    # Extract results
    println("\n📊 Inference Results:")
    
    # Extract posterior marginals
    states_marginals = result.posteriors[:states]
    A_marginal = result.posteriors[:A]
    B_marginal = result.posteriors[:B]
    D_marginal = result.posteriors[:D]
    
    println("✓ Successfully computed posterior marginals")
    println("  - State posteriors: $(length(states_marginals)) time steps")
    
    # Compute free energy if available
    if haskey(result, :free_energy)
        free_energy = result.free_energy
        println("🎯 Free Energy: $free_energy")
    end
    
    # Display state beliefs over time
    println("\n📈 State beliefs over time:")
    for (t, state_belief) in enumerate(states_marginals)
        belief_mode = mode(state_belief)
        belief_prob = pdf(state_belief, belief_mode)
        println("  Step $t: Most likely state = $belief_mode (prob ≈ $(round(belief_prob, digits=3)))")
    end
    
    return result
end

# Visualization function
function plot_results(result, observations_data)
    println("\n📊 Creating visualization...")
    
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
    println("Model: Active Inference Chronic Pain Multi-Theory Model v1")
    println("="^60)
    
    try
        # Run the simulation
        result = run_active_inference_simulation()
        
        # Generate synthetic observations for plotting
        observations_data = generate_observations(TIME_STEPS)
        
        # Create visualizations
        plot_results(result, observations_data)
        
        println("\n✅ Simulation completed successfully!")
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
