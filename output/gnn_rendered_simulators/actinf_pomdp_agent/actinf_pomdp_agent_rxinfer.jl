# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: Classic Active Inference POMDP Agent v1
# Generated: 2025-07-25 09:58:28

using RxInfer
using Distributions
using LinearAlgebra

# Model parameters
num_states = 3
num_obs = 3
num_controls = 3

# Define the model
@model function active_inference_model(n_steps)
    # Priors
    s_prior ~ Dirichlet(ones(num_states))
    A ~ MatrixDirichlet(ones(num_obs, num_states))
    B ~ ArrayDirichlet(ones(num_states, num_states, num_controls))
    C ~ Normal(0, 1)
    
    # State sequence
    s = randomvar(n_steps)
    o = datavar(Vector{Float64}, n_steps)
    
    # Initial state
    s[1] ~ Categorical(s_prior)
    
    # State transitions and observations
    for t in 2:n_steps
        s[t] ~ Categorical(B[:, :, 1])  # Assuming single action for now
        o[t] ~ Normal(A * s[t], 0.1)
    end
end

# Inference
n_steps = 10
results = inference(
    model = active_inference_model(n_steps),
    data = (o = [randn(num_obs) for _ in 1:n_steps],),
    initmarginals = (s = Categorical(ones(num_states) / num_states),),
    returnvars = (s = KeepLast(),)
)

println("RxInfer.jl simulation completed!")
