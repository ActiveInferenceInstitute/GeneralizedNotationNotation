#!/usr/bin/env julia
# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: Classic Active Inference POMDP Agent v1
# Generated: 2025-07-25 20:17:23

using RxInfer
using Distributions
using LinearAlgebra

# Model parameters
num_states = 3
num_obs = 3
num_controls = 3

# Define the model
@model function active_inference_model(num_steps)
    # State variables
    s = randomvar(num_steps)
    
    # Observation variables
    o = datavar(Vector{Float64}, num_steps)
    
    # Prior distributions
    s[1] ~ NormalMeanVariance(0.0, 1.0)
    
    # State transitions and observations
    for t in 2:num_steps
        s[t] ~ NormalMeanVariance(s[t-1], 0.1)
        o[t] ~ NormalMeanVariance(s[t], 0.5)
    end
end

# Simulation parameters
num_steps = 10

# Create model
model = active_inference_model(num_steps)

# Generate synthetic data
observations = randn(num_steps)

# Run inference
results = inference(
    model = model,
    data = (o = observations,),
    iterations = 10
)

println("RxInfer.jl simulation completed!")
println("State estimates: ", results.posteriors[:s])
