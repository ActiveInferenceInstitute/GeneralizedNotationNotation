#!/usr/bin/env julia
# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: Classic Active Inference POMDP Agent v1
# Generated: 2026-01-06 13:50:31

using Pkg

println("📦 Ensuring required packages are installed...")
try
    Pkg.add(["RxInfer", "Distributions", "Plots", "LinearAlgebra", "Random", "StatsBase"])
catch e
    println("⚠️  Package install error (might be already installed): $e")
end

using RxInfer
using Distributions
using LinearAlgebra
using Plots
using Random
using StatsBase

Random.seed!(42)

# --- Model Parameters ---
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 3
const TIME_STEPS = 20

# Parameter Matrices (from GNN)
# We use raw Vector of Vectors and convert to Matrix/Tensor for RxInfer
A_raw = fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS, NUM_STATES)
B_raw = fill(1.0/NUM_STATES, NUM_STATES, NUM_STATES, NUM_ACTIONS)
D_raw = fill(1.0/NUM_STATES, NUM_STATES)

# Convert to Julia Matrices
function to_matrix(raw)
    try
        if raw isa Matrix
            return raw
        elseif raw isa Vector && raw[1] isa Vector
            return hcat(raw...)
        end
    catch
    end
    return raw
end

function to_tensor(raw)
    try
        # B is [actions][prev][next] or similar. 
        # GNN Standard: B[action][next_state][prev_state] usually?
        # Let's assume input matches expected dimensions or is list of lists of lists
        if raw isa Array{Float64, 3}
            return raw
        elseif raw isa Vector && raw[1] isa Vector && raw[1][1] isa Vector
            # Dimensions: Action x Next x Prev ?
            # RxInfer expects: Next x Prev x Action (or similar, checking dims)
            # Let's construct generic 3D array
            n_actions = length(raw)
            n_next = length(raw[1])
            n_prev = length(raw[1][1])
            
            # Create tensor
            tensor = zeros(n_next, n_prev, n_actions)
            for a in 1:n_actions
                for n in 1:n_next
                    for p in 1:n_prev
                        tensor[n, p, a] = raw[a][n][p]
                    end
                end
            end
            return tensor
        end
    catch
    end
    return raw
end

A_matrix = to_matrix(A_raw)
B_matrix = to_tensor(B_raw) # Handling GNN B format
D_vector = Vector{Float64}(D_raw)

println("A matrix size: $(size(A_matrix))")
println("B matrix size: $(size(B_matrix))")
println("D vector size: $(size(D_vector))")


# --- RxInfer Model ---
# Fixed parameters version (active inference with known model)
@model function active_inference_model(observations, n_steps, A, B, D)
    
    # State sequence
    s = Vector{Any}(undef, n_steps)
    
    # Initial state
    s_init ~ Categorical(D)
    s[1] = s_init
    
    # First observation
    observations[1] ~ Categorical(s[1], copy(A))
    
    # State transitions and observations
    for t in 2:n_steps
        # Action selection (Random policy for now)
        action_idx = rand(1:NUM_ACTIONS)
        
        # State transition 
        # B is [Next, Prev, Action]
        # We slice B by action to get a transition matrix B_a
        B_a = B[:, :, action_idx] 
        s_next ~ DiscreteTransition(s[t-1], copy(B_a)) 
        s[t] = s_next
        
        # Observation
        # Try Categorical with two arguments (aliased to DiscreteTransition or similar?)
        observations[t] ~ Categorical(s[t], copy(A))
    end
    
    return s
end

# --- Simulation & Inference ---
function run_simulation()
    
    # Generate synthetic observations using the generative model (manual)
    # We use the same A and B matrices
    real_states = Vector{Int}(undef, TIME_STEPS)
    real_obs = Vector{Int}(undef, TIME_STEPS)
    
    # Initial
    current_state = rand(Categorical(D_vector))
    real_states[1] = current_state
    # Manual categorical logic for observations (using probability vector)
    real_obs[1] = rand(Categorical(A_matrix[:, current_state]))
    
    for t in 2:TIME_STEPS
        action = rand(1:NUM_ACTIONS)
        # B_matrix is [Next, Prev, Action]
        next_probs = B_matrix[:, current_state, action]
        current_state = rand(Categorical(next_probs))
        real_states[t] = current_state
        real_obs[t] = rand(Categorical(A_matrix[:, current_state]))
    end
    
    println("Observation sequence: $real_obs")
    
    # Run Inference
    result = infer(
        model = active_inference_model(n_steps=TIME_STEPS, A=A_matrix, B=B_matrix, D=D_vector),
        data = (observations = real_obs,),
        iterations = 10 # Not needed for exact inference in discrete case usually, but good for stability if loops
    )
    
    println("Inference complete.")
    return result, real_states, real_obs
end

# --- Main ---
function main()
    try
        result, true_states, obs = run_simulation()
        
        # Visualize
        posteriors = result.posteriors[:s]
        
        # Extract belief trace for state 1 over time
        belief_trace = [pdf(posteriors[t], 1) for t in 1:TIME_STEPS]
        
        p = plot(belief_trace, label="Belief(State 1)", title="State Inference", xlabel="Time", ylabel="Probability")
        scatter!(p, true_states .== 1, label="True State 1", markershape=:star)
        
        out_path = "rxinfer_results.png"
        savefig(p, out_path)
        println("Saved plot to $out_path")
        println("✅ RxInfer simulation successful")
        return 0
    catch e
        println("❌ Simulation failed: $e")
        # print stacktrace
        showerror(stdout, e, catch_backtrace())
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
