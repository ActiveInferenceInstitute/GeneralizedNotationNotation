#!/usr/bin/env julia
"""
RxInfer.jl simulation code for actinf_pomdp_agent
Generated from GNN specification
"""

using RxInfer
using Distributions
using LinearAlgebra

@model function actinf_pomdp_agent_model(n)
    # Define variables
    x = randomvar(n)
    y = datavar(Float64, n)
    
    # Define priors
    x_prior ~ NormalMeanVariance(0.0, 1.0)
    
    # Define likelihood
    for i in 1:n
        x[i] ~ NormalMeanVariance(x_prior, 1.0)
        y[i] ~ NormalMeanVariance(x[i], 0.1)
    end
end

function run_actinf_pomdp_agent_inference(data, n)
    """Run inference for actinf_pomdp_agent."""
    
    # Create model
    model = actinf_pomdp_agent_model(n)
    
    # Set up constraints
    constraints = @constraints begin
        q(x) :: NormalMeanVariance
        q(x_prior) :: NormalMeanVariance
    end
    
    # Run inference
    result = inference(
        model = model,
        data = (y = data,),
        constraints = constraints,
        initmarginals = (x_prior = NormalMeanVariance(0.0, 1.0),),
        iterations = 10
    )
    
    return result
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Generate sample data
    n = 100
    true_x = randn(n)
    data = true_x .+ 0.1 .* randn(n)
    
    # Run inference
    result = run_actinf_pomdp_agent_inference(data, n)
    
    println("Inference completed successfully")
    println("Posterior mean of x_prior: ", mean(result.posteriors[:x_prior]))
end
