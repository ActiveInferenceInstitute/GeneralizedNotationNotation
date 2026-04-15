#!/usr/bin/env julia
# RxInfer.jl Active Inference Simulation
# Generated from GNN Model: Two State Bistable POMDP
# Generated: 2026-04-15 12:26:34

using Pkg

println("📦 Ensuring required packages are installed...")
try
    Pkg.add(["RxInfer", "Distributions", "LinearAlgebra", "Random", "StatsBase"])
catch e
    println("⚠️  Package install error (might be already installed): $e")
end

using RxInfer
using Distributions
using LinearAlgebra
using Random
using StatsBase
using JSON

Random.seed!(42)

# --- Model Parameters ---
const NUM_STATES = 2
const NUM_OBSERVATIONS = 2
const NUM_ACTIONS = 2
const TIME_STEPS = 15

# Parameter Matrices (from GNN)
# We use raw Vector of Vectors and convert to Matrix/Tensor for RxInfer
A_raw = ([0.8, 0.2], [0.2, 0.8])
B_raw = ([[0.8, 0.3], [0.2, 0.7]], [[0.3, 0.8], [0.7, 0.2]])
C_raw = [0.0, 2.0]
D_raw = [0.5, 0.5]

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
        @warn "to_matrix conversion failed" exception=e
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
                # 3D case: each element is a matrix (list of rows)
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
            elseif first_action[1] isa Number
                # 2D case: each element is a flat vector (row of transition matrix)
                # This is a passive model (HMM/Markov Chain) with no action dimension
                println("ℹ️  B is 2D (passive model) — expanding to 3D with single action")
                n_rows = length(arr)
                n_cols = length(first_action)
                mat = zeros(n_rows, n_cols)
                for r in 1:n_rows
                    row_data = collect(arr[r])
                    for c in 1:n_cols
                        mat[r, c] = row_data[c]
                    end
                end
                # Normalize columns
                for c in 1:n_cols
                    cs = sum(mat[:, c])
                    if cs > 0
                        mat[:, c] ./= cs
                    end
                end
                # Expand to 3D with single action
                return reshape(mat, n_rows, n_cols, 1)
            end
        end
    catch e
        @warn "to_tensor conversion failed" exception=e
        println("to_tensor warning: $e")
    end
    return raw
end

A_matrix = to_matrix(A_raw)
B_matrix = to_tensor(B_raw) # Handling GNN B format
C_vector = Vector{Float64}(collect(C_raw))
D_vector = Vector{Float64}(collect(D_raw))

# Normalize D_vector to ensure it sums exactly to 1.0 (required for Categorical)
D_vector = D_vector ./ sum(D_vector)

# Normalize A_matrix columns (each column should sum to 1)
for j in 1:size(A_matrix, 2)
    A_matrix[:, j] = A_matrix[:, j] ./ sum(A_matrix[:, j])
end

# Softmax utility function
function softmax(x)
    ex = exp.(x .- maximum(x))
    return ex ./ sum(ex)
end

# Convert C (log-preferences) to preferred observation distribution
C_preferred = softmax(C_vector)

println("A matrix size: $(size(A_matrix))")
println("B matrix size: $(size(B_matrix))")
println("C vector (preferences): $C_vector")
println("C preferred (softmax): $C_preferred")
println("D vector size: $(size(D_vector))")


# --- RxInfer Single-Step Inference Model ---
# Used for belief updating given a single observation
@model function belief_update_model(observation, A, prior)
    # State prior from previous belief
    s ~ Categorical(prior)
    # Observation likelihood
    observation ~ DiscreteTransition(s, A)
    return s
end

# --- Expected Free Energy (EFE) Computation ---
# G(a) = ambiguity + risk
# ambiguity: expected uncertainty about observations given predicted states
# risk: KL divergence between expected observations and preferred observations
function compute_efe(belief, action_idx, A, B, C_pref)
    # Predicted next state distribution: s' = B[:,:,a] * belief
    B_a = B[:, :, action_idx]
    predicted_state = B_a * belief
    
    # Normalize (handle numerical issues)
    predicted_state = max.(predicted_state, 1e-16)
    predicted_state = predicted_state ./ sum(predicted_state)
    
    # Expected observation distribution: o' = A * s'
    predicted_obs = A * predicted_state
    predicted_obs = max.(predicted_obs, 1e-16)
    predicted_obs = predicted_obs ./ sum(predicted_obs)
    
    # Ambiguity: expected entropy of observations conditioned on states
    # H[P(o|s)] weighted by predicted state
    ambiguity = 0.0
    for j in 1:length(predicted_state)
        if predicted_state[j] > 1e-16
            # Entropy of column j of A (observation distribution for state j)
            col = A[:, j]
            col = max.(col, 1e-16)
            ambiguity -= predicted_state[j] * sum(col .* log.(col))
        end
    end
    
    # Risk: KL divergence D_KL(predicted_obs || C_preferred)
    C_safe = max.(C_pref, 1e-16)
    risk = sum(predicted_obs .* (log.(predicted_obs) .- log.(C_safe)))
    
    # EFE = ambiguity + risk (lower is better)
    return ambiguity + risk
end

# --- Active Inference Action Selection ---
function select_action(belief, A, B, C_pref; action_precision=4.0)
    n_actions = size(B, 3)
    efe_values = zeros(n_actions)
    
    for a in 1:n_actions
        efe_values[a] = compute_efe(belief, a, A, B, C_pref)
    end
    
    # Policy via softmax over negative EFE (lower EFE = higher probability)
    neg_efe = -action_precision .* efe_values
    action_probs = softmax(neg_efe)
    
    # Sample action from policy
    action = rand(Categorical(action_probs))
    
    return action, efe_values, action_probs
end

# --- One-hot encoding ---
function one_hot(idx, n)
    v = zeros(n)
    v[idx] = 1.0
    return v
end

# --- Active Inference Simulation Loop ---
function run_simulation()
    println("\n🧠 Running Active Inference simulation with EFE-based action selection...")
    
    # Storage
    true_states = Vector{Int}(undef, TIME_STEPS)
    observations = Vector{Int}(undef, TIME_STEPS)
    actions = Vector{Int}(undef, TIME_STEPS)
    beliefs = Vector{Vector{Float64}}(undef, TIME_STEPS)
    efe_history = Vector{Vector{Float64}}(undef, TIME_STEPS)
    action_probs_history = Vector{Vector{Float64}}(undef, TIME_STEPS)
    
    # Initialize environment
    current_state = rand(Categorical(D_vector))
    current_belief = copy(D_vector)
    
    for t in 1:TIME_STEPS
        # 1. Environment generates observation
        true_states[t] = current_state
        obs = rand(Categorical(A_matrix[:, current_state]))
        observations[t] = obs
        
        # 2. Infer beliefs using RxInfer (single-step Bayesian inference)
        obs_one_hot = one_hot(obs, NUM_OBSERVATIONS)
        try
            result = infer(
                model = belief_update_model(A=A_matrix, prior=current_belief),
                data = (observation = obs_one_hot,),
                iterations = 5
            )
            # Extract posterior belief
            posterior = result.posteriors[:s]
            final_posterior = posterior[end]
            current_belief = probvec(final_posterior)
        catch e
            # Recovery: manual Bayesian update if RxInfer fails
            println("  Step $t: RxInfer inference recovery - $e")
            likelihood = A_matrix[obs, :]
            unnormalized = current_belief .* likelihood
            current_belief = unnormalized ./ sum(unnormalized)
        end
        
        # Ensure belief is valid
        current_belief = max.(current_belief, 1e-16)
        current_belief = current_belief ./ sum(current_belief)
        beliefs[t] = copy(current_belief)
        
        # 3. Compute EFE and select action (Active Inference!)
        action, efe_values, action_probs = select_action(
            current_belief, A_matrix, B_matrix, C_preferred
        )
        actions[t] = action
        efe_history[t] = copy(efe_values)
        action_probs_history[t] = copy(action_probs)
        
        # 4. Environment transitions based on selected action
        next_probs = B_matrix[:, current_state, action]
        next_probs = max.(next_probs, 1e-16)
        next_probs = next_probs ./ sum(next_probs)
        current_state = rand(Categorical(next_probs))
        
        # 5. Update belief for next timestep (predictive prior)
        B_a = B_matrix[:, :, action]
        current_belief = B_a * current_belief
        current_belief = max.(current_belief, 1e-16)
        current_belief = current_belief ./ sum(current_belief)
        
        println("  Step $t: obs=$obs, action=$action, belief_max=$(round(maximum(beliefs[t]), digits=3)), EFE=$(round.(efe_values, digits=3))")
    end
    
    println("\n✅ Active Inference simulation complete")
    println("Action distribution: ", StatsBase.countmap(actions))
    
    # Compute per-step EFE of selected action
    selected_efe = [efe_history[t][actions[t]] for t in 1:TIME_STEPS]
    
    # Save results
    results_data = Dict(
        "framework" => "rxinfer",
        "model_name" => "Two State Bistable POMDP",
        "time_steps" => TIME_STEPS,
        "true_states" => true_states,
        "observations" => observations,
        "actions" => actions,
        "beliefs" => beliefs,
        "efe_history" => selected_efe,
        "efe_per_action" => efe_history,
        "action_probabilities" => action_probs_history,
        "num_states" => NUM_STATES,
        "num_observations" => NUM_OBSERVATIONS,
        "num_actions" => NUM_ACTIONS,
        "preferences" => C_vector,
        "validation" => Dict(
            "all_beliefs_valid" => all(b -> all(x -> 0.0 <= x <= 1.0, b), beliefs),
            "beliefs_sum_to_one" => all(b -> abs(sum(b) - 1.0) < 0.01, beliefs),
            "actions_in_range" => all(a -> 1 <= a <= NUM_ACTIONS, actions)
        )
    )
    
    open("simulation_results.json", "w") do f
        JSON.print(f, results_data, 4)
    end
    println("✅ Standardized results saved to simulation_results.json")
    
    return beliefs, actions, efe_history
end

# --- Main ---
function main()
    try
        beliefs, actions, efe_hist = run_simulation()
        
        println("✅ RxInfer Active Inference simulation successful")
        println("📊 Visualizations will be generated by the analysis step")
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
