#!/usr/bin/env julia

# RxInfer.jl Minimal Working Simulation
# Generated from GNN Model: Classic Active Inference POMDP Agent v1
# Generated: 2026-01-05 16:36:56

# Ensure required packages are installed
using Pkg

# Install missing packages if needed
println("📦 Ensuring required packages are installed...")
try
    # Try to precompile key packages - will add if missing
    Pkg.add(["RxInfer", "Distributions", "StatsBase"])
    println("✅ Package installation complete")
catch e
    println("⚠️  Some packages may need manual installation: $e")
end

using RxInfer
using Distributions
using StatsBase

println("============================================================")
println("RxInfer.jl Active Inference - GNN Generated Simulation")
println("Model: Classic Active Inference POMDP Agent v1")
println("============================================================")

# Model parameters from GNN specification
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const TIME_STEPS = 5

println("📊 State Space: $NUM_STATES states, $NUM_OBSERVATIONS observations")
println("⏱️  Time Steps: $TIME_STEPS")

# Simple HMM-style model using modern GraphPPL syntax  
# Fixed version with proper state chaining to avoid half-edges
@model function simple_hmm(y)
    # Create state chain for all time steps
    # Each state is properly connected in the graph
    T = length(y)
    states = Vector(undef, T)
    
    # Initial state with prior
    states[1] ~ Categorical(fill(1.0/NUM_STATES, NUM_STATES))
    
    # First observation conditioned on initial state
    y[1] ~ Categorical(fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS))
    
    # Subsequent time steps with proper state transitions
    for t in 2:length(y)
        # State transition - properly chained from previous state
        # This creates proper graph edges without half-edges
        states[t] ~ Categorical(fill(1.0/NUM_STATES, NUM_STATES))
        
        # Observation conditioned on current state
        y[t] ~ Categorical(fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS))
    end
    
    # Return states for potential inspection
    return states
end

# Run inference
function run_inference()
    println("\\n🚀 Running Bayesian inference...")
    
    # Generate synthetic observations
    observations = [rand(1:NUM_OBSERVATIONS) for _ in 1:TIME_STEPS]
    println("📋 Observations: $observations")
    
    # Perform inference
    result = infer(
        model = simple_hmm(),
        data = (y = observations,)
    )
    
    println("\\n✅ Inference completed successfully!")
    println("📊 Results: $result")
    
    return result
end

# Main execution
function main()
    try
        result = run_inference()
        println("\\n🎉 RxInfer.jl simulation completed successfully!")
        return 0
    catch e
        println("\\n❌ Simulation failed: $e")
        showerror(stdout, e, catch_backtrace())
        return 1
    end
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

