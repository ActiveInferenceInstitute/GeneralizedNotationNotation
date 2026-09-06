#!/usr/bin/env julia

"""
Advanced POMDP Analysis Module for ActiveInference.jl

This module provides comprehensive analysis tools for POMDP models beyond basic statistics:
- Information-theoretic measures (entropy, mutual information, information gain)
- Convergence analysis (belief convergence, parameter convergence, policy stability)
- Policy performance metrics (regret analysis, optimality measures, efficiency metrics)
- Theoretical bounds (sample complexity, approximation error, convergence rates)
- Model comparison and validation (cross-validation, likelihood ratios, Bayes factors)
- Sensitivity analysis (parameter robustness, noise tolerance, perturbation analysis)
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates

# Ensure required packages are available
for pkg in ["StatsBase", "HypothesisTests", "Plots", "PlotlyJS"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase

# ====================================
# INFORMATION-THEORETIC MEASURES
# ====================================

"""Calculate Shannon entropy of a probability distribution."""
function shannon_entropy(prob_dist::Vector{Float64})
    # Remove zero probabilities for log calculation
    non_zero_probs = filter(p -> p > 1e-12, prob_dist)
    -sum(p * log2(p) for p in non_zero_probs)
end

"""Calculate KL divergence between two probability distributions."""
function kl_divergence(p::Vector{Float64}, q::Vector{Float64})
    # Ensure same length
    @assert length(p) == length(q) "Distributions must have same length"
    
    kl = 0.0
    for i in 1:length(p)
        if p[i] > 1e-12 && q[i] > 1e-12
            kl += p[i] * log(p[i] / q[i])
        elseif p[i] > 1e-12 && q[i] <= 1e-12
            kl = Inf
            break
        end
    end
    return kl
end

"""Calculate mutual information between states and observations."""
function mutual_information(joint_dist::Matrix{Float64})
    # Marginal distributions
    p_state = sum(joint_dist, dims=2)[:, 1]
    p_obs = sum(joint_dist, dims=1)[1, :]
    
    mi = 0.0
    for i in 1:size(joint_dist, 1)
        for j in 1:size(joint_dist, 2)
            if joint_dist[i, j] > 1e-12 && p_state[i] > 1e-12 && p_obs[j] > 1e-12
                mi += joint_dist[i, j] * log(joint_dist[i, j] / (p_state[i] * p_obs[j]))
            end
        end
    end
    return mi
end

"""Calculate information gain from belief updating."""
function information_gain(prior_beliefs::Vector{Float64}, posterior_beliefs::Vector{Float64})
    kl_divergence(posterior_beliefs, prior_beliefs)
end

"""Calculate expected information gain for a given observation model."""
function expected_information_gain(beliefs::Vector{Float64}, A_matrix::Matrix{Float64})
    expected_gain = 0.0
    n_obs = size(A_matrix, 1)
    
    for o in 1:n_obs
        # Probability of observation
        p_obs = sum(beliefs[s] * A_matrix[o, s] for s in 1:length(beliefs))
        
        if p_obs > 1e-12
            # Posterior after observing o
            posterior = zeros(length(beliefs))
            for s in 1:length(beliefs)
                posterior[s] = beliefs[s] * A_matrix[o, s] / p_obs
            end
            
            # Information gain for this observation
            gain = information_gain(beliefs, posterior)
            expected_gain += p_obs * gain
        end
    end
    
    return expected_gain
end

# ====================================
# CONVERGENCE ANALYSIS
# ====================================

"""Analyze belief convergence over time."""
function analyze_belief_convergence(beliefs_trace::Matrix{Float64})
    n_steps, n_states = size(beliefs_trace)
    convergence_metrics = Dict{String, Any}()
    
    # Calculate belief variance over time
    belief_variance = [var(beliefs_trace[t, :]) for t in 1:n_steps]
    convergence_metrics["belief_variance_trace"] = belief_variance
    convergence_metrics["final_variance"] = belief_variance[end]
    convergence_metrics["variance_reduction"] = belief_variance[1] - belief_variance[end]
    
    # Calculate convergence rate (exponential fitting)
    if n_steps > 10
        # Fit exponential decay to variance
        log_variance = log.(max.(belief_variance, 1e-12))
        x = collect(1:n_steps)
        
        # Simple linear regression on log scale
        n = length(x)
        mean_x, mean_y = mean(x), mean(log_variance)
        slope = sum((x .- mean_x) .* (log_variance .- mean_y)) / sum((x .- mean_x).^2)
        
        convergence_metrics["exponential_rate"] = -slope
        convergence_metrics["convergence_time_constant"] = -1/slope
    end
    
    # Steady-state analysis (last 20% of trajectory)
    steady_start = max(1, round(Int, 0.8 * n_steps))
    steady_beliefs = beliefs_trace[steady_start:end, :]
    convergence_metrics["steady_state_mean"] = mean(steady_beliefs, dims=1)[1, :]
    convergence_metrics["steady_state_std"] = std(steady_beliefs, dims=1)[1, :]
    
    # Autocorrelation analysis
    if n_steps > 20
        belief_autocorr = zeros(n_states)
        for s in 1:n_states
            belief_series = beliefs_trace[:, s]
            # Lag-1 autocorrelation
            if std(belief_series) > 1e-12
                belief_autocorr[s] = cor(belief_series[1:end-1], belief_series[2:end])
            end
        end
        convergence_metrics["belief_autocorrelation"] = belief_autocorr
    end
    
    return convergence_metrics
end

"""Analyze parameter learning convergence."""
function analyze_parameter_convergence(param_trace::Matrix{Float64}, true_params::Vector{Float64})
    n_episodes, n_params = size(param_trace)
    convergence_metrics = Dict{String, Any}()
    
    # Parameter error over time
    param_errors = [norm(param_trace[t, :] - true_params) for t in 1:n_episodes]
    convergence_metrics["parameter_error_trace"] = param_errors
    convergence_metrics["final_error"] = param_errors[end]
    convergence_metrics["error_reduction"] = param_errors[1] - param_errors[end]
    
    # Learning rate estimation
    if n_episodes > 10
        log_errors = log.(max.(param_errors, 1e-12))
        x = collect(1:n_episodes)
        
        # Fit exponential decay
        n = length(x)
        mean_x, mean_y = mean(x), mean(log_errors)
        slope = sum((x .- mean_x) .* (log_errors .- mean_y)) / sum((x .- mean_x).^2)
        
        convergence_metrics["learning_rate"] = -slope
        convergence_metrics["learning_time_constant"] = -1/slope
    end
    
    # Convergence quality metrics
    final_10_percent = max(1, round(Int, 0.9 * n_episodes)):n_episodes
    convergence_metrics["final_error_mean"] = mean(param_errors[final_10_percent])
    convergence_metrics["final_error_std"] = std(param_errors[final_10_percent])
    convergence_metrics["convergence_efficiency"] = param_errors[1] / param_errors[end]
    
    return convergence_metrics
end

"""Analyze policy stability and convergence."""
function analyze_policy_convergence(policy_trace::Array{Float64, 3})
    n_steps, n_policies = size(policy_trace)[1:2]
    convergence_metrics = Dict{String, Any}()
    
    # Policy entropy over time (measures exploration vs exploitation)
    policy_entropy = [shannon_entropy(policy_trace[t, :]) for t in 1:n_steps]
    convergence_metrics["policy_entropy_trace"] = policy_entropy
    convergence_metrics["initial_entropy"] = policy_entropy[1]
    convergence_metrics["final_entropy"] = policy_entropy[end]
    convergence_metrics["entropy_reduction"] = policy_entropy[1] - policy_entropy[end]
    
    # Policy stability (variance in policy across time)
    policy_stability = zeros(n_policies)
    for p in 1:n_policies
        policy_stability[p] = var(policy_trace[:, p])
    end
    convergence_metrics["policy_stability"] = policy_stability
    convergence_metrics["mean_policy_stability"] = mean(policy_stability)
    
    # Convergence to deterministic policy
    final_policy = policy_trace[end, :]
    max_prob = maximum(final_policy)
    convergence_metrics["final_policy_determinism"] = max_prob
    convergence_metrics["final_policy_concentration"] = max_prob / (1/n_policies)  # Relative to uniform
    
    return convergence_metrics
end

# ====================================
# POLICY PERFORMANCE METRICS
# ====================================

"""Calculate regret analysis for policy performance."""
function calculate_regret_analysis(rewards_trace::Vector{Float64}, optimal_reward::Float64)
    n_steps = length(rewards_trace)
    regret_metrics = Dict{String, Any}()
    
    # Instantaneous regret
    instantaneous_regret = [optimal_reward - rewards_trace[t] for t in 1:n_steps]
    regret_metrics["instantaneous_regret"] = instantaneous_regret
    
    # Cumulative regret
    cumulative_regret = cumsum(instantaneous_regret)
    regret_metrics["cumulative_regret"] = cumulative_regret
    regret_metrics["total_regret"] = cumulative_regret[end]
    
    # Average regret
    average_regret = cumulative_regret ./ (1:n_steps)
    regret_metrics["average_regret"] = average_regret
    regret_metrics["final_average_regret"] = average_regret[end]
    
    # Regret growth rate
    if n_steps > 10
        log_regret = log.(max.(cumulative_regret, 1e-12))
        log_steps = log.(1:n_steps)
        
        # Fit power law: log(regret) = α + β*log(t)
        n = length(log_steps)
        mean_x, mean_y = mean(log_steps), mean(log_regret)
        beta = sum((log_steps .- mean_x) .* (log_regret .- mean_y)) / sum((log_steps .- mean_x).^2)
        
        regret_metrics["regret_growth_exponent"] = beta
        regret_metrics["sublinear_regret"] = beta < 1.0
    end
    
    return regret_metrics
end

"""Calculate policy efficiency metrics."""
function calculate_policy_efficiency(actions_trace::Vector{Int}, rewards_trace::Vector{Float64})
    efficiency_metrics = Dict{String, Any}()
    
    # Action diversity
    unique_actions = length(unique(actions_trace))
    total_actions = length(actions_trace)
    efficiency_metrics["action_diversity"] = unique_actions / total_actions
    
    # Action efficiency (reward per action)
    efficiency_metrics["reward_per_action"] = sum(rewards_trace) / total_actions
    
    # Action consistency (how often the same action is repeated)
    action_changes = sum(actions_trace[2:end] .!= actions_trace[1:end-1])
    efficiency_metrics["action_consistency"] = 1.0 - (action_changes / (total_actions - 1))
    
    # Exploration vs exploitation ratio
    most_frequent_action_count = maximum([count(==(a), actions_trace) for a in unique(actions_trace)])
    efficiency_metrics["exploitation_ratio"] = most_frequent_action_count / total_actions
    efficiency_metrics["exploration_ratio"] = 1.0 - efficiency_metrics["exploitation_ratio"]
    
    return efficiency_metrics
end

"""Calculate optimality measures."""
function calculate_optimality_measures(observed_performance::Vector{Float64}, 
                                     optimal_performance::Vector{Float64})
    optimality_metrics = Dict{String, Any}()
    
    # Performance ratio
    performance_ratios = observed_performance ./ max.(optimal_performance, 1e-12)
    optimality_metrics["performance_ratios"] = performance_ratios
    optimality_metrics["mean_performance_ratio"] = mean(performance_ratios)
    optimality_metrics["min_performance_ratio"] = minimum(performance_ratios)
    
    # Optimality gap
    optimality_gaps = optimal_performance - observed_performance
    optimality_metrics["optimality_gaps"] = optimality_gaps
    optimality_metrics["mean_optimality_gap"] = mean(optimality_gaps)
    optimality_metrics["max_optimality_gap"] = maximum(optimality_gaps)
    
    # Relative error
    relative_errors = abs.(optimality_gaps) ./ max.(optimal_performance, 1e-12)
    optimality_metrics["relative_errors"] = relative_errors
    optimality_metrics["mean_relative_error"] = mean(relative_errors)
    
    return optimality_metrics
end

# ====================================
# THEORETICAL BOUNDS AND COMPLEXITY
# ====================================

"""Calculate sample complexity bounds for learning."""
function calculate_sample_complexity_bounds(n_states::Int, n_obs::Int, n_actions::Int, 
                                          confidence::Float64 = 0.95, 
                                          accuracy::Float64 = 0.01)
    bounds = Dict{String, Any}()
    
    # PAC learning bounds for finite MDPs
    delta = 1.0 - confidence
    
    # Sample complexity for learning observation model (A matrix)
    # Using Hoeffding's inequality: O(|S||O|/ε²)log(1/δ)
    A_complexity = ceil(Int, (n_states * n_obs * log(2/delta)) / (2 * accuracy^2))
    bounds["A_matrix_sample_complexity"] = A_complexity
    
    # Sample complexity for learning transition model (B matrix) 
    # O(|S|²|A|/ε²)log(1/δ)
    B_complexity = ceil(Int, (n_states^2 * n_actions * log(2/delta)) / (2 * accuracy^2))
    bounds["B_matrix_sample_complexity"] = B_complexity
    
    # Total sample complexity
    bounds["total_sample_complexity"] = A_complexity + B_complexity
    
    # Regret bounds (simplified)
    # Regret grows as O(√T) for finite MDPs with known structure
    bounds["regret_bound_coefficient"] = sqrt(n_states * n_actions)
    
    return bounds
end

"""Calculate approximation error bounds."""
function calculate_approximation_bounds(belief_precision::Float64, 
                                      observation_noise::Float64,
                                      transition_noise::Float64)
    bounds = Dict{String, Any}()
    
    # Belief approximation error
    bounds["max_belief_error"] = belief_precision
    
    # Observation model approximation error  
    bounds["observation_approximation_error"] = observation_noise
    
    # Transition model approximation error
    bounds["transition_approximation_error"] = transition_noise
    
    # Propagated error bounds (simplified analysis)
    # Error propagates multiplicatively through time
    bounds["error_propagation_rate"] = max(observation_noise, transition_noise)
    
    # Stability condition (error must be < 1 for convergence)
    bounds["stability_condition"] = bounds["error_propagation_rate"] < 1.0
    
    return bounds
end

"""Analyze computational complexity."""
function analyze_computational_complexity(n_states::Int, n_obs::Int, n_actions::Int, 
                                        time_horizon::Int)
    complexity = Dict{String, Any}()
    
    # State inference complexity: O(|S||O|)
    complexity["state_inference_complexity"] = n_states * n_obs
    
    # Policy inference complexity: O(|S||A|T) for planning horizon T
    complexity["policy_inference_complexity"] = n_states * n_actions * time_horizon
    
    # Learning complexity: O(|S|²|A| + |S||O|) per update
    complexity["learning_complexity"] = n_states^2 * n_actions + n_states * n_obs
    
    # Memory complexity: O(|S|²|A| + |S||O|)
    complexity["memory_complexity"] = n_states^2 * n_actions + n_states * n_obs
    
    # Space complexity class
    if n_states <= 10 && n_actions <= 10
        complexity["complexity_class"] = "Small"
    elseif n_states <= 100 && n_actions <= 100
        complexity["complexity_class"] = "Medium"
    else
        complexity["complexity_class"] = "Large"
    end
    
    return complexity
end

# ====================================
# MODEL COMPARISON AND VALIDATION
# ====================================

"""Perform cross-validation analysis."""
function cross_validation_analysis(data::Matrix{Float64}, k_folds::Int = 5)
    n_samples = size(data, 1)
    fold_size = ceil(Int, n_samples / k_folds)
    
    cv_results = Dict{String, Any}()
    fold_errors = Float64[]
    
    for k in 1:k_folds
        # Define test set
        test_start = (k-1) * fold_size + 1
        test_end = min(k * fold_size, n_samples)
        test_indices = test_start:test_end
        
        # Training set is everything else
        train_indices = [1:test_start-1; test_end+1:n_samples]
        
        if length(train_indices) > 0
            train_data = data[train_indices, :]
            test_data = data[test_indices, :]
            
            # Simple validation: mean squared error
            train_mean = mean(train_data, dims=1)[1, :]
            test_error = mean((test_data .- train_mean').^2)
            push!(fold_errors, test_error)
        end
    end
    
    cv_results["fold_errors"] = fold_errors
    cv_results["mean_cv_error"] = mean(fold_errors)
    cv_results["std_cv_error"] = std(fold_errors)
    cv_results["cv_confidence_interval"] = (mean(fold_errors) - 1.96*std(fold_errors)/sqrt(k_folds),
                                           mean(fold_errors) + 1.96*std(fold_errors)/sqrt(k_folds))
    
    return cv_results
end

"""Calculate model likelihood and Bayes factors."""
function calculate_model_likelihood(observations::Vector{Int}, 
                                  beliefs_trace::Matrix{Float64},
                                  A_matrix::Matrix{Float64})
    n_steps = length(observations)
    log_likelihood = 0.0
    
    for t in 1:n_steps
        obs = observations[t]
        beliefs = beliefs_trace[t, :]
        
        # Predicted probability of observation
        pred_prob = sum(beliefs[s] * A_matrix[obs, s] for s in 1:length(beliefs))
        
        if pred_prob > 1e-12
            log_likelihood += log(pred_prob)
        else
            log_likelihood += log(1e-12)  # Avoid -Inf
        end
    end
    
    return log_likelihood
end

"""Calculate Bayes factor for model comparison."""
function calculate_bayes_factor(log_likelihood1::Float64, log_likelihood2::Float64,
                              model1_complexity::Int, model2_complexity::Int)
    # Simplified BIC approximation
    bic1 = -2 * log_likelihood1 + model1_complexity * log(100)  # Assume 100 data points
    bic2 = -2 * log_likelihood2 + model2_complexity * log(100)
    
    # Bayes factor approximation
    log_bayes_factor = 0.5 * (bic2 - bic1)
    bayes_factor = exp(log_bayes_factor)
    
    return bayes_factor, log_bayes_factor
end

# ====================================
# SENSITIVITY ANALYSIS
# ====================================

"""Perform parameter sensitivity analysis."""
function parameter_sensitivity_analysis(base_params::Dict{String, Float64},
                                       param_ranges::Dict{String, Tuple{Float64, Float64}},
                                       performance_function::Function,
                                       n_samples::Int = 100)
    sensitivity_results = Dict{String, Any}()
    
    for (param_name, (min_val, max_val)) in param_ranges
        param_values = range(min_val, max_val, length=n_samples)
        performance_values = Float64[]
        
        for val in param_values
            # Create modified parameters
            modified_params = copy(base_params)
            modified_params[param_name] = val
            
            # Evaluate performance
            perf = performance_function(modified_params)
            push!(performance_values, perf)
        end
        
        # Calculate sensitivity metrics
        sensitivity_results[param_name] = Dict(
            "parameter_values" => collect(param_values),
            "performance_values" => performance_values,
            "sensitivity" => std(performance_values) / std(param_values),
            "min_performance" => minimum(performance_values),
            "max_performance" => maximum(performance_values),
            "performance_range" => maximum(performance_values) - minimum(performance_values)
        )
    end
    
    return sensitivity_results
end

"""Analyze noise tolerance."""
function noise_tolerance_analysis(clean_data::Matrix{Float64},
                                noise_levels::Vector{Float64},
                                performance_function::Function)
    tolerance_results = Dict{String, Any}()
    
    for noise_level in noise_levels
        # Add Gaussian noise
        noisy_data = clean_data + noise_level * randn(size(clean_data))
        
        # Evaluate performance
        performance = performance_function(noisy_data)
        
        tolerance_results[noise_level] = performance
    end
    
    noise_performance = [tolerance_results[noise] for noise in noise_levels]
    
    return Dict(
        "noise_levels" => noise_levels,
        "performance_values" => noise_performance,
        "noise_sensitivity" => -gradient(noise_performance, noise_levels),
        "tolerance_threshold" => find_tolerance_threshold(noise_levels, noise_performance)
    )
end

"""Find tolerance threshold (where performance drops below acceptable level)."""
function find_tolerance_threshold(noise_levels::Vector{Float64}, 
                                performance_values::Vector{Float64},
                                acceptable_ratio::Float64 = 0.9)
    baseline_performance = performance_values[1]  # Assume first is noise-free
    threshold_performance = acceptable_ratio * baseline_performance
    
    for (i, perf) in enumerate(performance_values)
        if perf < threshold_performance
            return noise_levels[i]
        end
    end
    
    return noise_levels[end]  # No threshold found within tested range
end

"""Simple gradient calculation."""
function gradient(y::Vector{Float64}, x::Vector{Float64})
    n = length(y)
    grad = zeros(n)
    
    # Forward difference for first point
    grad[1] = (y[2] - y[1]) / (x[2] - x[1])
    
    # Central difference for middle points
    for i in 2:n-1
        grad[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    end
    
    # Backward difference for last point
    grad[n] = (y[n] - y[n-1]) / (x[n] - x[n-1])
    
    return grad
end

# ====================================
# MAIN ANALYSIS FUNCTIONS
# ====================================

"""Comprehensive POMDP analysis of simulation data."""
function comprehensive_pomdp_analysis(output_dir::String)
    analysis_dir = joinpath(output_dir, "advanced_analysis")
    mkpath(analysis_dir)
    
    println("\n🔬 Advanced POMDP Analysis")
    println("="^50)
    
    # Load data traces
    data_traces_dir = joinpath(output_dir, "data_traces")
    
    try
        # Load belief traces
        beliefs_data = readdlm(joinpath(data_traces_dir, "beliefs_over_time.csv"), ',', skipstart=6)
        beliefs_trace = beliefs_data[:, 2:end]  # Skip step column
        
        # Load actions and observations
        actions_data = readdlm(joinpath(data_traces_dir, "actions_over_time.csv"), ',', skipstart=6)
        obs_data = readdlm(joinpath(data_traces_dir, "observations_over_time.csv"), ',', skipstart=6)
        
        actions_trace = Int.(actions_data[:, 2])
        obs_trace = Int.(obs_data[:, 2])
        
        # Load learning data if available
        learning_data = nothing
        try
            learning_data = readdlm(joinpath(data_traces_dir, "learning_curve.csv"), ',', skipstart=6)
        catch
            @warn "Learning data not found, skipping learning analysis"
        end
        
        # ===== INFORMATION-THEORETIC ANALYSIS =====
        println("📊 Computing information-theoretic measures...")
        
        info_analysis = Dict{String, Any}()
        
        # Belief entropy over time
        belief_entropies = [shannon_entropy(beliefs_trace[t, :]) for t in 1:size(beliefs_trace, 1)]
        info_analysis["belief_entropy_trace"] = belief_entropies
        info_analysis["initial_entropy"] = belief_entropies[1]
        info_analysis["final_entropy"] = belief_entropies[end]
        info_analysis["entropy_reduction"] = belief_entropies[1] - belief_entropies[end]
        
        # Information gain analysis
        info_gains = Float64[]
        for t in 2:size(beliefs_trace, 1)
            gain = information_gain(beliefs_trace[t-1, :], beliefs_trace[t, :])
            push!(info_gains, gain)
        end
        info_analysis["information_gains"] = info_gains
        info_analysis["mean_information_gain"] = mean(info_gains)
        info_analysis["total_information_gain"] = sum(info_gains)
        
        # Save information analysis
        open(joinpath(analysis_dir, "information_theoretic_analysis.txt"), "w") do f
            println(f, "Information-Theoretic Analysis")
            println(f, "="^40)
            println(f, "Initial belief entropy: $(round(info_analysis["initial_entropy"], digits=4))")
            println(f, "Final belief entropy: $(round(info_analysis["final_entropy"], digits=4))")
            println(f, "Entropy reduction: $(round(info_analysis["entropy_reduction"], digits=4))")
            println(f, "Mean information gain per step: $(round(info_analysis["mean_information_gain"], digits=6))")
            println(f, "Total information gain: $(round(info_analysis["total_information_gain"], digits=4))")
        end
        
        # ===== CONVERGENCE ANALYSIS =====
        println("📈 Analyzing convergence properties...")
        
        convergence_analysis = analyze_belief_convergence(beliefs_trace)
        
        # Save convergence analysis
        open(joinpath(analysis_dir, "convergence_analysis.txt"), "w") do f
            println(f, "Convergence Analysis")
            println(f, "="^40)
            println(f, "Final belief variance: $(round(convergence_analysis["final_variance"], digits=6))")
            println(f, "Variance reduction: $(round(convergence_analysis["variance_reduction"], digits=6))")
            
            if haskey(convergence_analysis, "exponential_rate")
                println(f, "Exponential convergence rate: $(round(convergence_analysis["exponential_rate"], digits=4))")
                println(f, "Time constant: $(round(convergence_analysis["convergence_time_constant"], digits=2))")
            end
            
            println(f, "Steady-state belief mean: $(round.(convergence_analysis["steady_state_mean"], digits=4))")
            println(f, "Steady-state belief std: $(round.(convergence_analysis["steady_state_std"], digits=6))")
        end
        
        # ===== POLICY PERFORMANCE ANALYSIS =====
        println("🎯 Analyzing policy performance...")
        
        # Simple reward calculation (preference satisfaction)
        rewards = Float64[]
        for t in 1:length(obs_trace)
            # Simple reward: higher preference for observation 2
            reward = obs_trace[t] == 2 ? 1.0 : 0.0
            push!(rewards, reward)
        end
        
        # Policy efficiency
        efficiency_metrics = calculate_policy_efficiency(actions_trace, rewards)
        
        # Regret analysis (assume optimal reward is 1.0)
        regret_analysis = calculate_regret_analysis(rewards, 1.0)
        
        # Save performance analysis
        open(joinpath(analysis_dir, "policy_performance_analysis.txt"), "w") do f
            println(f, "Policy Performance Analysis")
            println(f, "="^40)
            println(f, "Action diversity: $(round(efficiency_metrics["action_diversity"], digits=3))")
            println(f, "Reward per action: $(round(efficiency_metrics["reward_per_action"], digits=3))")
            println(f, "Action consistency: $(round(efficiency_metrics["action_consistency"], digits=3))")
            println(f, "Exploration ratio: $(round(efficiency_metrics["exploration_ratio"], digits=3))")
            println(f, "Exploitation ratio: $(round(efficiency_metrics["exploitation_ratio"], digits=3))")
            println(f, "")
            println(f, "Regret Analysis:")
            println(f, "Total regret: $(round(regret_analysis["total_regret"], digits=3))")
            println(f, "Final average regret: $(round(regret_analysis["final_average_regret"], digits=4))")
            
            if haskey(regret_analysis, "regret_growth_exponent")
                println(f, "Regret growth exponent: $(round(regret_analysis["regret_growth_exponent"], digits=3))")
                println(f, "Sublinear regret: $(regret_analysis["sublinear_regret"])")
            end
        end
        
        # ===== THEORETICAL BOUNDS =====
        println("📐 Computing theoretical bounds...")
        
        n_states = size(beliefs_trace, 2)
        n_obs = length(unique(obs_trace))
        n_actions = length(unique(actions_trace))
        
        sample_complexity = calculate_sample_complexity_bounds(n_states, n_obs, n_actions)
        computational_complexity = analyze_computational_complexity(n_states, n_obs, n_actions, 1)
        
        # Save theoretical analysis
        open(joinpath(analysis_dir, "theoretical_bounds.txt"), "w") do f
            println(f, "Theoretical Bounds and Complexity")
            println(f, "="^40)
            println(f, "Problem Dimensions:")
            println(f, "  States: $n_states")
            println(f, "  Observations: $n_obs") 
            println(f, "  Actions: $n_actions")
            println(f, "")
            println(f, "Sample Complexity Bounds:")
            println(f, "  A matrix learning: $(sample_complexity["A_matrix_sample_complexity"]) samples")
            println(f, "  B matrix learning: $(sample_complexity["B_matrix_sample_complexity"]) samples")
            println(f, "  Total complexity: $(sample_complexity["total_sample_complexity"]) samples")
            println(f, "")
            println(f, "Computational Complexity:")
            println(f, "  State inference: O($(computational_complexity["state_inference_complexity"]))")
            println(f, "  Policy inference: O($(computational_complexity["policy_inference_complexity"]))")
            println(f, "  Learning: O($(computational_complexity["learning_complexity"]))")
            println(f, "  Complexity class: $(computational_complexity["complexity_class"])")
        end
        
        # ===== PARAMETER LEARNING ANALYSIS (if available) =====
        if learning_data !== nothing
            println("🎓 Analyzing parameter learning...")
            
            # Assume simple 2-state, 2-obs problem for now
            true_params = [0.8, 0.2, 0.2, 0.8]  # Simple A matrix flattened
            param_trace = learning_data[:, 2:end]  # Skip episode column
            
            if size(param_trace, 2) == length(true_params)
                learning_convergence = analyze_parameter_convergence(param_trace, true_params)
                
                open(joinpath(analysis_dir, "parameter_learning_analysis.txt"), "w") do f
                    println(f, "Parameter Learning Analysis")
                    println(f, "="^40)
                    println(f, "Final parameter error: $(round(learning_convergence["final_error"], digits=6))")
                    println(f, "Error reduction: $(round(learning_convergence["error_reduction"], digits=6))")
                    println(f, "Convergence efficiency: $(round(learning_convergence["convergence_efficiency"], digits=2))")
                    
                    if haskey(learning_convergence, "learning_rate")
                        println(f, "Learning rate: $(round(learning_convergence["learning_rate"], digits=4))")
                        println(f, "Learning time constant: $(round(learning_convergence["learning_time_constant"], digits=2))")
                    end
                    
                    println(f, "Final error statistics:")
                    println(f, "  Mean: $(round(learning_convergence["final_error_mean"], digits=6))")
                    println(f, "  Std: $(round(learning_convergence["final_error_std"], digits=6))")
                end
            end
        end
        
        # ===== SAVE ANALYSIS DATA =====
        println("💾 Saving detailed analysis data...")
        
        # Save as CSV for further analysis
        writedlm(joinpath(analysis_dir, "belief_entropy_trace.csv"), 
                 ["step" "entropy"; (1:length(belief_entropies)) belief_entropies], ',')
        
        writedlm(joinpath(analysis_dir, "information_gain_trace.csv"),
                 ["step" "information_gain"; (2:length(info_gains)+1) info_gains], ',')
        
        writedlm(joinpath(analysis_dir, "convergence_variance_trace.csv"),
                 ["step" "variance"; (1:length(convergence_analysis["belief_variance_trace"])) convergence_analysis["belief_variance_trace"]], ',')
        
        writedlm(joinpath(analysis_dir, "regret_analysis.csv"),
                 ["step" "instantaneous_regret" "cumulative_regret" "average_regret"; 
                  (1:length(regret_analysis["instantaneous_regret"])) regret_analysis["instantaneous_regret"] regret_analysis["cumulative_regret"] regret_analysis["average_regret"]], ',')
        
        println("✅ Advanced POMDP analysis completed successfully!")
        println("📁 Results saved to: $analysis_dir")
        
    catch e
        println("❌ Error in advanced analysis: $e")
        @warn "Advanced analysis failed" exception=e
    end
end

"""Main entry point for advanced analysis."""
function main()
    if length(ARGS) > 0
        output_dir = ARGS[1]
        if isdir(output_dir)
            comprehensive_pomdp_analysis(output_dir)
        else
            error("❌ Directory not found: $output_dir")
        end
    else
        println("Usage: julia advanced_pomdp_analysis.jl <output_directory>")
        println("Example: julia advanced_pomdp_analysis.jl activeinference_outputs_YYYY-MM-DD_HH-MM-SS")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 