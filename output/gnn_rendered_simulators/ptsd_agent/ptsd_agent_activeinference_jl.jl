# ActiveInference.jl Simulation
# Generated from GNN Model: Unknown
# Generated: 2025-07-25 22:45:19

using ActiveInference
using LinearAlgebra

# Model parameters
num_states = 3
num_obs = 3
num_controls = 3

# Initialize matrices
A = Matrix{Float64}(I, num_obs, num_states)  # Likelihood matrix
B = zeros(num_states, num_states, num_controls)  # Transition matrix
for i in 1:num_controls
    B[:, :, i] = Matrix{Float64}(I, num_states, num_states)
end
C = zeros(num_obs)  # Preference vector
D = ones(num_states) / num_states  # Prior over states

# Create agent
agent = ActiveInferenceAgent(A, B, C, D)

# Simulation parameters
T = 10

# Run simulation
for t in 1:T
    # Generate observation
    obs = rand(1:num_obs)
    
    # Agent inference
    qs = infer_states(agent, obs)
    q_pi = infer_policies(agent)
    action = sample_action(agent, q_pi)
    
    println("Step \$t: Observation=\$obs, Action=\$action")
    println("  State beliefs: \$qs")
end

println("ActiveInference.jl simulation completed!")
