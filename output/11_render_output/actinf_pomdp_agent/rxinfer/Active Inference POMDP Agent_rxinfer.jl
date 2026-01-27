#!/usr/bin/env julia
# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: Active Inference POMDP Agent
# Generated: 2026-01-27 08:47:31

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
using JSON

Random.seed!(42)

# --- Model Parameters ---
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const NUM_ACTIONS = 3
const TIME_STEPS = 30

# Parameter Matrices (from GNN)
# We use raw Vector of Vectors and convert to Matrix/Tensor for RxInfer
A_raw = ([0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9])
B_raw = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
D_raw = [0.33333, 0.33333, 0.33333]

# Convert to Julia Matrices
function to_matrix(raw)
    try
        if raw isa Matrix
            return raw
        end
        # Handle Tuple or Vector of Tuples/Vectors
        arr = collect(raw)
        if !isempty(arr) && (arr[1] isa Tuple || arr[1] isa Vector)
            rows = [collect(r) for r in arr]
            return hcat(rows...)'
        end
    catch e
        println("to_matrix warning: $e")
    end
    return raw
end

function to_tensor(raw)
    try
        if raw isa Array{Float64, 3}
            return raw
        end
        # Handle Tuple or Vector structure for B[action][row][col]
        arr = collect(raw)
        if !isempty(arr)
            first_action = collect(arr[1])
            if !isempty(first_action) && (first_action[1] isa Tuple || first_action[1] isa Vector)
                n_actions = length(arr)
                n_rows = length(first_action)
                first_row = collect(first_action[1])
                n_cols = length(first_row)
                
                tensor = zeros(n_rows, n_cols, n_actions)
                for a in 1:n_actions
                    action_data = collect(arr[a])
                    for r in 1:n_rows
                        row_data = collect(action_data[r])
                        for c in 1:n_cols
                            tensor[r, c, a] = row_data[c]
                        end
                    end
                end
                return tensor
            end
        end
    catch e
        println("to_tensor warning: $e")
    end
    return raw
end

A_matrix = to_matrix(A_raw)
B_matrix = to_tensor(B_raw) # Handling GNN B format
D_vector = Vector{Float64}(collect(D_raw))

# Normalize D_vector to ensure it sums exactly to 1.0 (required for Categorical)
D_vector = D_vector ./ sum(D_vector)

# Normalize A_matrix columns (each column should sum to 1)
for j in 1:size(A_matrix, 2)
    A_matrix[:, j] = A_matrix[:, j] ./ sum(A_matrix[:, j])
end

println("A matrix size: $(size(A_matrix))")
println("B matrix size: $(size(B_matrix))")
println("D vector size: $(size(D_vector))")


# --- RxInfer Model ---
# Fixed parameters version (active inference with known model)
@model function active_inference_model(observations, n_steps, A, B, D)
    
    # State sequence 
    # GraphPPL v3/v4 supports auto-collection of variables via indexing.
    # No randomvar declaration needed.
    
    # Initial state
    s[1] ~ Categorical(D)
    
    # First observation
    observations[1] ~ DiscreteTransition(s[1], A)
    
    # State transitions and observations
    for t in 2:n_steps
        # Action selection (Random policy for now)
        action_idx = rand(1:NUM_ACTIONS)
        
        # State transition 
        # B is [Next, Prev, Action]
        # We slice B by action to get a transition matrix B_a
        B_a = B[:, :, action_idx] 
        s[t] ~ DiscreteTransition(s[t-1], B_a)
        
        # Observation
        observations[t] ~ DiscreteTransition(s[t], A)
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
    
    # One-hot encode observations (required for DiscreteTransition in RxInfer v4)
    function one_hot(idx, n)
        v = zeros(n)
        v[idx] = 1.0
        return v
    end
    
    obs_one_hot = [one_hot(o, NUM_OBSERVATIONS) for o in real_obs]
    println("Observation sequence (one-hot) prepared.")
    
    # Run Inference
    result = infer(
        model = active_inference_model(n_steps=TIME_STEPS, A=A_matrix, B=B_matrix, D=D_vector),
        data = (observations = obs_one_hot,),
        iterations = 10 
    )
    
    println("Inference complete.")
    
    # Standardized result extraction
    # Convert belief traces (Categorical) to probability vectors
    all_iterations = result.posteriors[:s]
    final_iteration = all_iterations[end]
    beliefs = [probvec(final_iteration[t]) for t in 1:TIME_STEPS]
    
    results_data = Dict(
        "framework" => "rxinfer",
        "model_name" => "Active Inference POMDP Agent",
        "time_steps" => TIME_STEPS,
        "true_states" => real_states,
        "observations" => real_obs,
        "beliefs" => beliefs,
        "num_states" => NUM_STATES,
        "num_observations" => NUM_OBSERVATIONS
    )
    
    open("simulation_results.json", "w") do f
        JSON.print(f, results_data, 4)
    end
    println("✅ Standardized results saved to simulation_results.json")
    
    return result, real_states, real_obs
end

# --- Main ---
function main()
    try
        result, true_states, obs = run_simulation()
        
        # Visualize
        # result.posteriors[:s] returns a vector of iterations, 
        # each being a vector of Categorical distributions over time.
        all_iterations = result.posteriors[:s]
        final_iteration_posteriors = all_iterations[end]
        
        # Extract belief trace for state 1 over time
        belief_trace = [pdf(final_iteration_posteriors[t], 1) for t in 1:TIME_STEPS]
        
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
