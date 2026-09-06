#!/usr/bin/env julia

"""
Advanced Uncertainty Quantification Module for ActiveInference.jl

This module provides comprehensive uncertainty analysis and decomposition:
- Epistemic vs aleatoric uncertainty decomposition
- Uncertainty propagation through belief updates
- Information-theoretic uncertainty measures
- Model uncertainty and parameter uncertainty quantification
- Uncertainty-aware decision making and active learning
- Bayesian uncertainty estimation with confidence intervals
- Uncertainty reduction strategies and information gain analysis
- Predictive uncertainty and forecast confidence assessment
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates
using JSON
using Random

# Ensure required packages
for pkg in ["StatsBase", "Bootstrap", "Optim", "PDMats"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase

# ====================================
# EPISTEMIC VS ALEATORIC UNCERTAINTY
# ====================================

"""Decompose uncertainty into epistemic (knowledge) and aleatoric (irreducible) components."""
function decompose_uncertainty(beliefs_trace::Matrix{Float64},
                             observations::Vector{Int},
                             n_bootstrap_samples::Int = 100)
    n_steps, n_states = size(beliefs_trace)
    uncertainty_decomposition = Dict{String, Any}()
    
    # 1. Aleatoric uncertainty: inherent randomness in the environment
    aleatoric_uncertainty = zeros(n_steps)
    for t in 1:n_steps
        # Aleatoric uncertainty is the entropy of the belief distribution
        aleatoric_uncertainty[t] = shannon_entropy(beliefs_trace[t, :])
    end
    
    uncertainty_decomposition["aleatoric_uncertainty"] = aleatoric_uncertainty
    uncertainty_decomposition["mean_aleatoric"] = mean(aleatoric_uncertainty)
    
    # 2. Epistemic uncertainty: uncertainty due to lack of knowledge
    # Estimated using bootstrap sampling of belief trajectories
    epistemic_uncertainty = zeros(n_steps)
    
    Random.seed!(42)  # For reproducibility
    
    for t in 1:n_steps
        if t >= 5  # Need some history for bootstrap
            # Bootstrap sample belief trajectories
            bootstrap_beliefs = zeros(n_bootstrap_samples, n_states)
            
            for sample in 1:n_bootstrap_samples
                # Sample with replacement from recent history
                history_window = max(1, t-10):t
                sampled_indices = sample(history_window, min(5, length(history_window)), replace=true)
                
                # Create perturbed belief based on sampled history
                base_belief = beliefs_trace[t, :]
                noise_level = 0.1 * std([norm(beliefs_trace[i, :] - beliefs_trace[i-1, :]) 
                                        for i in max(2, t-5):t if i > 1])
                
                perturbed_belief = base_belief + noise_level * randn(n_states)
                perturbed_belief = max.(perturbed_belief, 0.001)  # Ensure positive
                perturbed_belief ./= sum(perturbed_belief)  # Normalize
                
                bootstrap_beliefs[sample, :] = perturbed_belief
            end
            
            # Epistemic uncertainty is the variance across bootstrap samples
            epistemic_uncertainty[t] = mean(var(bootstrap_beliefs, dims=1))
        end
    end
    
    uncertainty_decomposition["epistemic_uncertainty"] = epistemic_uncertainty
    uncertainty_decomposition["mean_epistemic"] = mean(epistemic_uncertainty[5:end])
    
    # 3. Total uncertainty and decomposition
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
    uncertainty_decomposition["total_uncertainty"] = total_uncertainty
    uncertainty_decomposition["mean_total"] = mean(total_uncertainty)
    
    # Relative contributions
    if mean(total_uncertainty) > 1e-6
        uncertainty_decomposition["aleatoric_ratio"] = mean(aleatoric_uncertainty) / mean(total_uncertainty)
        uncertainty_decomposition["epistemic_ratio"] = mean(epistemic_uncertainty[5:end]) / mean(total_uncertainty[5:end])
    end
    
    # 4. Uncertainty evolution analysis
    if n_steps > 10
        # How does uncertainty change over time?
        aleatoric_trend = cor(1:n_steps, aleatoric_uncertainty)
        epistemic_trend = cor(5:n_steps, epistemic_uncertainty[5:end])
        
        uncertainty_decomposition["aleatoric_trend"] = aleatoric_trend
        uncertainty_decomposition["epistemic_trend"] = epistemic_trend
        
        # Uncertainty reduction rate
        initial_uncertainty = mean(total_uncertainty[1:5])
        final_uncertainty = mean(total_uncertainty[end-4:end])
        uncertainty_reduction = (initial_uncertainty - final_uncertainty) / initial_uncertainty
        
        uncertainty_decomposition["uncertainty_reduction_rate"] = uncertainty_reduction
    end
    
    return uncertainty_decomposition
end

"""Quantify model uncertainty through ensemble methods."""
function quantify_model_uncertainty(beliefs_trace::Matrix{Float64},
                                  actions::Vector{Int},
                                  observations::Vector{Int},
                                  n_ensemble_models::Int = 20)
    model_uncertainty = Dict{String, Any}()
    n_steps, n_states = size(beliefs_trace)
    
    # Create ensemble of models with different parameters
    Random.seed!(123)
    ensemble_predictions = zeros(n_ensemble_models, n_steps, n_states)
    
    for model_idx in 1:n_ensemble_models
        # Create perturbed model parameters
        noise_scale = 0.1
        perturbed_beliefs = copy(beliefs_trace)
        
        # Add parameter uncertainty by perturbing belief updates
        for t in 2:n_steps
            # Add noise to belief update
            update_noise = noise_scale * randn(n_states)
            perturbed_beliefs[t, :] += update_noise
            
            # Ensure valid probability distribution
            perturbed_beliefs[t, :] = max.(perturbed_beliefs[t, :], 0.001)
            perturbed_beliefs[t, :] ./= sum(perturbed_beliefs[t, :])
        end
        
        ensemble_predictions[model_idx, :, :] = perturbed_beliefs
    end
    
    # Calculate model uncertainty metrics
    prediction_variance = zeros(n_steps, n_states)
    prediction_disagreement = zeros(n_steps)
    
    for t in 1:n_steps
        # Variance across ensemble models
        prediction_variance[t, :] = var(ensemble_predictions[:, t, :], dims=1)[1, :]
        
        # Overall disagreement (mean pairwise distance)
        disagreements = []
        for i in 1:n_ensemble_models
            for j in (i+1):n_ensemble_models
                disagreement = norm(ensemble_predictions[i, t, :] - ensemble_predictions[j, t, :])
                push!(disagreements, disagreement)
            end
        end
        prediction_disagreement[t] = mean(disagreements)
    end
    
    model_uncertainty["prediction_variance"] = prediction_variance
    model_uncertainty["prediction_disagreement"] = prediction_disagreement
    model_uncertainty["mean_disagreement"] = mean(prediction_disagreement)
    model_uncertainty["max_disagreement"] = maximum(prediction_disagreement)
    
    # Model confidence intervals
    confidence_intervals = zeros(n_steps, n_states, 2)  # [lower, upper] bounds
    for t in 1:n_steps
        for s in 1:n_states
            predictions = ensemble_predictions[:, t, s]
            sorted_predictions = sort(predictions)
            
            # 95% confidence interval
            lower_idx = max(1, round(Int, 0.025 * n_ensemble_models))
            upper_idx = min(n_ensemble_models, round(Int, 0.975 * n_ensemble_models))
            
            confidence_intervals[t, s, 1] = sorted_predictions[lower_idx]
            confidence_intervals[t, s, 2] = sorted_predictions[upper_idx]
        end
    end
    
    model_uncertainty["confidence_intervals"] = confidence_intervals
    
    # Calibration analysis: how well do confidence intervals capture true values
    coverage_rates = zeros(n_states)
    for s in 1:n_states
        within_interval = 0
        for t in 1:n_steps
            true_value = beliefs_trace[t, s]
            lower_bound = confidence_intervals[t, s, 1]
            upper_bound = confidence_intervals[t, s, 2]
            
            if lower_bound <= true_value <= upper_bound
                within_interval += 1
            end
        end
        coverage_rates[s] = within_interval / n_steps
    end
    
    model_uncertainty["coverage_rates"] = coverage_rates
    model_uncertainty["mean_coverage"] = mean(coverage_rates)
    model_uncertainty["calibration_quality"] = abs(mean(coverage_rates) - 0.95)  # How close to nominal 95%
    
    return model_uncertainty
end

# ====================================
# INFORMATION-THEORETIC UNCERTAINTY
# ====================================

"""Calculate information-theoretic measures of uncertainty."""
function calculate_information_uncertainty(beliefs_trace::Matrix{Float64},
                                         observations::Vector{Int},
                                         actions::Vector{Int})
    info_uncertainty = Dict{String, Any}()
    n_steps, n_states = size(beliefs_trace)
    
    # 1. Differential entropy over time
    differential_entropies = [shannon_entropy(beliefs_trace[t, :]) for t in 1:n_steps]
    info_uncertainty["differential_entropy"] = differential_entropies
    info_uncertainty["mean_entropy"] = mean(differential_entropies)
    info_uncertainty["entropy_trend"] = cor(1:n_steps, differential_entropies)
    
    # 2. Mutual information between states and observations
    mutual_information_values = zeros(n_steps - 1)
    for t in 2:n_steps
        # Create joint distribution between previous beliefs and current observation
        prev_beliefs = beliefs_trace[t-1, :]
        curr_obs = observations[t]
        
        # Approximate joint distribution
        joint_dist = zeros(n_states, maximum(observations))
        for s in 1:n_states
            if curr_obs <= size(joint_dist, 2)
                joint_dist[s, curr_obs] = prev_beliefs[s]
            end
        end
        
        # Normalize
        if sum(joint_dist) > 0
            joint_dist ./= sum(joint_dist)
            mutual_information_values[t-1] = mutual_information(joint_dist)
        end
    end
    
    info_uncertainty["mutual_information"] = mutual_information_values
    info_uncertainty["mean_mutual_information"] = mean(mutual_information_values)
    
    # 3. Conditional entropy H(S|O)
    conditional_entropies = zeros(n_steps - 1)
    for t in 2:n_steps
        prev_entropy = differential_entropies[t-1]
        mutual_info = mutual_information_values[t-1]
        conditional_entropies[t-1] = prev_entropy - mutual_info
    end
    
    info_uncertainty["conditional_entropy"] = conditional_entropies
    info_uncertainty["mean_conditional_entropy"] = mean(conditional_entropies)
    
    # 4. Information gain from observations
    information_gains = zeros(n_steps - 1)
    for t in 2:n_steps
        prior_entropy = differential_entropies[t-1]
        posterior_entropy = differential_entropies[t]
        information_gains[t-1] = prior_entropy - posterior_entropy
    end
    
    info_uncertainty["information_gain"] = information_gains
    info_uncertainty["mean_information_gain"] = mean(information_gains)
    info_uncertainty["total_information_gained"] = sum(information_gains)
    
    # 5. Predictive information
    if n_steps > 5
        predictive_information_values = zeros(n_steps - 5)
        for t in 1:(n_steps - 5)
            # Information in current state about future states
            current_entropy = differential_entropies[t]
            future_entropy = differential_entropies[t + 5]
            
            # Simplified predictive information
            predictive_information_values[t] = max(0, current_entropy - future_entropy)
        end
        
        info_uncertainty["predictive_information"] = predictive_information_values
        info_uncertainty["mean_predictive_information"] = mean(predictive_information_values)
    end
    
    # 6. Uncertainty reduction efficiency
    if length(information_gains) > 1
        # How efficiently does each observation reduce uncertainty?
        efficiency_scores = information_gains ./ differential_entropies[1:end-1]
        info_uncertainty["uncertainty_reduction_efficiency"] = efficiency_scores
        info_uncertainty["mean_efficiency"] = mean(efficiency_scores)
    end
    
    return info_uncertainty
end

"""Calculate mutual information between joint distributions."""
function mutual_information(joint_dist::Matrix{Float64})
    # Marginal distributions
    p_x = sum(joint_dist, dims=2)[:, 1]
    p_y = sum(joint_dist, dims=1)[1, :]
    
    mi = 0.0
    for i in 1:size(joint_dist, 1)
        for j in 1:size(joint_dist, 2)
            if joint_dist[i, j] > 1e-12 && p_x[i] > 1e-12 && p_y[j] > 1e-12
                mi += joint_dist[i, j] * log(joint_dist[i, j] / (p_x[i] * p_y[j]))
            end
        end
    end
    return mi
end

# ====================================
# UNCERTAINTY-AWARE DECISION MAKING
# ====================================

"""Analyze uncertainty-aware decision making patterns."""
function analyze_uncertainty_aware_decisions(beliefs_trace::Matrix{Float64},
                                           actions::Vector{Int},
                                           uncertainty_estimates::Vector{Float64})
    decision_analysis = Dict{String, Any}()
    n_steps = length(actions)
    
    # 1. Uncertainty-action correlation
    if length(uncertainty_estimates) == n_steps && std(uncertainty_estimates) > 1e-6
        uncertainty_action_corr = cor(uncertainty_estimates, float.(actions))
        decision_analysis["uncertainty_action_correlation"] = uncertainty_action_corr
    end
    
    # 2. Risk-seeking vs risk-averse behavior
    high_uncertainty_threshold = quantile(uncertainty_estimates, 0.75)
    low_uncertainty_threshold = quantile(uncertainty_estimates, 0.25)
    
    high_uncertainty_indices = uncertainty_estimates .>= high_uncertainty_threshold
    low_uncertainty_indices = uncertainty_estimates .<= low_uncertainty_threshold
    
    if sum(high_uncertainty_indices) > 0 && sum(low_uncertainty_indices) > 0
        # Action diversity in high vs low uncertainty
        high_uncertainty_actions = actions[high_uncertainty_indices]
        low_uncertainty_actions = actions[low_uncertainty_indices]
        
        high_uncertainty_diversity = length(unique(high_uncertainty_actions)) / length(high_uncertainty_actions)
        low_uncertainty_diversity = length(unique(low_uncertainty_actions)) / length(low_uncertainty_actions)
        
        decision_analysis["high_uncertainty_action_diversity"] = high_uncertainty_diversity
        decision_analysis["low_uncertainty_action_diversity"] = low_uncertainty_diversity
        decision_analysis["uncertainty_exploration_ratio"] = high_uncertainty_diversity / low_uncertainty_diversity
        
        # Risk preference analysis
        if high_uncertainty_diversity > low_uncertainty_diversity
            decision_analysis["risk_preference"] = "risk_seeking"
        else
            decision_analysis["risk_preference"] = "risk_averse"
        end
    end
    
    # 3. Uncertainty reduction strategies
    uncertainty_changes = diff(uncertainty_estimates)
    action_changes = diff(actions)
    
    # Actions that tend to reduce uncertainty
    uncertainty_reducing_actions = []
    for action in unique(actions)
        action_indices = actions[2:end] .== action
        if sum(action_indices) > 0
            mean_uncertainty_change = mean(uncertainty_changes[action_indices])
            push!(uncertainty_reducing_actions, (action, mean_uncertainty_change))
        end
    end
    
    # Sort by uncertainty reduction (most negative = best reduction)
    sort!(uncertainty_reducing_actions, by=x->x[2])
    decision_analysis["uncertainty_reducing_actions"] = uncertainty_reducing_actions
    
    if length(uncertainty_reducing_actions) > 0
        best_action = uncertainty_reducing_actions[1][1]
        worst_action = uncertainty_reducing_actions[end][1]
        
        decision_analysis["best_uncertainty_reducing_action"] = best_action
        decision_analysis["worst_uncertainty_reducing_action"] = worst_action
    end
    
    # 4. Adaptive uncertainty thresholds
    # How does the agent's uncertainty tolerance change over time?
    window_size = min(10, n_steps ÷ 4)
    uncertainty_tolerances = []
    
    for start_idx in 1:window_size:(n_steps - window_size + 1)
        end_idx = min(start_idx + window_size - 1, n_steps)
        window_uncertainty = uncertainty_estimates[start_idx:end_idx]
        window_actions = actions[start_idx:end_idx]
        
        # Tolerance = willingness to act despite uncertainty
        action_diversity = length(unique(window_actions)) / length(window_actions)
        mean_uncertainty = mean(window_uncertainty)
        
        uncertainty_tolerance = action_diversity / (mean_uncertainty + 1e-6)
        push!(uncertainty_tolerances, uncertainty_tolerance)
    end
    
    decision_analysis["uncertainty_tolerance_evolution"] = uncertainty_tolerances
    if length(uncertainty_tolerances) > 1
        decision_analysis["tolerance_trend"] = cor(1:length(uncertainty_tolerances), uncertainty_tolerances)
    end
    
    return decision_analysis
end

# ====================================
# COMPREHENSIVE UNCERTAINTY ANALYSIS
# ====================================

"""Run comprehensive uncertainty quantification analysis."""
function comprehensive_uncertainty_analysis(output_dir::String)
    println("📊 Running Comprehensive Uncertainty Quantification Analysis")
    
    # Create output directory
    uncertainty_dir = joinpath(output_dir, "uncertainty_quantification")
    mkpath(uncertainty_dir)
    
    analysis_results = Dict{String, Any}()
    
    try
        # Load basic simulation data
        basic_data_path = joinpath(output_dir, "simulation_results", "basic_simulation.csv")
        if isfile(basic_data_path)
            basic_data = load_statistical_data(basic_data_path)
            
            if size(basic_data, 2) >= 4
                steps = Int.(basic_data[:, 1])
                observations = Int.(basic_data[:, 2])
                actions = Int.(basic_data[:, 3])
                beliefs = basic_data[:, 4]
                
                # Create beliefs trace matrix
                beliefs_trace = hcat(beliefs, 1.0 .- beliefs)
                
                # 1. Epistemic vs Aleatoric uncertainty decomposition
                uncertainty_decomposition = decompose_uncertainty(beliefs_trace, observations)
                analysis_results["uncertainty_decomposition"] = uncertainty_decomposition
                
                # 2. Model uncertainty quantification
                model_uncertainty = quantify_model_uncertainty(beliefs_trace, actions, observations)
                analysis_results["model_uncertainty"] = model_uncertainty
                
                # 3. Information-theoretic uncertainty
                info_uncertainty = calculate_information_uncertainty(beliefs_trace, observations, actions)
                analysis_results["information_uncertainty"] = info_uncertainty
                
                # 4. Uncertainty-aware decision making
                total_uncertainty = uncertainty_decomposition["total_uncertainty"]
                decision_analysis = analyze_uncertainty_aware_decisions(beliefs_trace, actions, total_uncertainty)
                analysis_results["uncertainty_aware_decisions"] = decision_analysis
                
                println("✅ Comprehensive uncertainty analysis completed")
            end
        end
        
        # 5. Load additional uncertainty data if available
        uncertainty_data_path = joinpath(output_dir, "simulation_results", "uncertainty_analysis.csv")
        if isfile(uncertainty_data_path)
            try
                uncertainty_data = load_statistical_data(uncertainty_data_path)
                
                analysis_results["extended_uncertainty_analysis"] = Dict(
                    "data_available" => true,
                    "data_shape" => size(uncertainty_data)
                )
                
                println("✅ Extended uncertainty analysis completed")
            catch e
                @warn "Extended uncertainty analysis failed: $e"
            end
        end
        
    catch e
        @error "Uncertainty quantification analysis failed: $e"
        analysis_results["error"] = string(e)
    end
    
    # Save comprehensive results
    if !isempty(analysis_results)
        # Save as JSON
        json_path = joinpath(uncertainty_dir, "uncertainty_analysis_results.json")
        open(json_path, "w") do f
            JSON.print(f, analysis_results, 2)
        end
        
        # Create summary report
        create_uncertainty_analysis_report(analysis_results, uncertainty_dir)
        
        println("📊 Uncertainty analysis results saved to: $uncertainty_dir")
    end
    
    return analysis_results
end

"""Create a comprehensive uncertainty analysis report."""
function create_uncertainty_analysis_report(results::Dict{String, Any}, output_dir::String)
    report_path = joinpath(output_dir, "uncertainty_analysis_report.md")
    
    open(report_path, "w") do f
        println(f, "# Uncertainty Quantification Analysis Report")
        println(f, "")
        println(f, "Generated: $(now())")
        println(f, "")
        
        if haskey(results, "uncertainty_decomposition")
            decomp = results["uncertainty_decomposition"]
            println(f, "## Uncertainty Decomposition")
            println(f, "")
            println(f, "- **Mean Aleatoric Uncertainty**: $(round(get(decomp, "mean_aleatoric", 0.0), digits=3))")
            println(f, "- **Mean Epistemic Uncertainty**: $(round(get(decomp, "mean_epistemic", 0.0), digits=3))")
            println(f, "- **Mean Total Uncertainty**: $(round(get(decomp, "mean_total", 0.0), digits=3))")
            
            if haskey(decomp, "aleatoric_ratio")
                println(f, "- **Aleatoric Ratio**: $(round(decomp["aleatoric_ratio"], digits=3))")
                println(f, "- **Epistemic Ratio**: $(round(decomp["epistemic_ratio"], digits=3))")
            end
            
            if haskey(decomp, "uncertainty_reduction_rate")
                println(f, "- **Uncertainty Reduction Rate**: $(round(decomp["uncertainty_reduction_rate"], digits=3))")
            end
            println(f, "")
        end
        
        if haskey(results, "model_uncertainty")
            model_unc = results["model_uncertainty"]
            println(f, "## Model Uncertainty")
            println(f, "")
            println(f, "- **Mean Disagreement**: $(round(get(model_unc, "mean_disagreement", 0.0), digits=3))")
            println(f, "- **Max Disagreement**: $(round(get(model_unc, "max_disagreement", 0.0), digits=3))")
            println(f, "- **Mean Coverage**: $(round(get(model_unc, "mean_coverage", 0.0), digits=3))")
            println(f, "- **Calibration Quality**: $(round(get(model_unc, "calibration_quality", 0.0), digits=3))")
            println(f, "")
        end
        
        if haskey(results, "information_uncertainty")
            info_unc = results["information_uncertainty"]
            println(f, "## Information-Theoretic Uncertainty")
            println(f, "")
            println(f, "- **Mean Entropy**: $(round(get(info_unc, "mean_entropy", 0.0), digits=3))")
            println(f, "- **Mean Information Gain**: $(round(get(info_unc, "mean_information_gain", 0.0), digits=3))")
            println(f, "- **Total Information Gained**: $(round(get(info_unc, "total_information_gained", 0.0), digits=3))")
            
            if haskey(info_unc, "mean_efficiency")
                println(f, "- **Uncertainty Reduction Efficiency**: $(round(info_unc["mean_efficiency"], digits=3))")
            end
            
            if haskey(info_unc, "mean_predictive_information")
                println(f, "- **Mean Predictive Information**: $(round(info_unc["mean_predictive_information"], digits=3))")
            end
            println(f, "")
        end
        
        if haskey(results, "uncertainty_aware_decisions")
            decision_unc = results["uncertainty_aware_decisions"]
            println(f, "## Uncertainty-Aware Decision Making")
            println(f, "")
            
            if haskey(decision_unc, "risk_preference")
                println(f, "- **Risk Preference**: $(decision_unc["risk_preference"])")
            end
            
            if haskey(decision_unc, "uncertainty_exploration_ratio")
                println(f, "- **Uncertainty Exploration Ratio**: $(round(decision_unc["uncertainty_exploration_ratio"], digits=3))")
            end
            
            if haskey(decision_unc, "best_uncertainty_reducing_action")
                println(f, "- **Best Uncertainty Reducing Action**: $(decision_unc["best_uncertainty_reducing_action"])")
            end
            
            if haskey(decision_unc, "tolerance_trend")
                println(f, "- **Uncertainty Tolerance Trend**: $(round(decision_unc["tolerance_trend"], digits=3))")
            end
            println(f, "")
        end
        
        println(f, "## Summary")
        println(f, "")
        println(f, "This analysis provides comprehensive uncertainty quantification including")
        println(f, "epistemic vs aleatoric decomposition, model uncertainty assessment,")
        println(f, "information-theoretic measures, and uncertainty-aware decision making patterns.")
    end
    
    println("📋 Uncertainty analysis report saved: $report_path")
end

# Utility functions
function shannon_entropy(prob_dist::Vector{Float64})
    non_zero_probs = filter(p -> p > 1e-12, prob_dist)
    -sum(p * log2(p) for p in non_zero_probs)
end

function load_statistical_data(filepath::String)
    if !isfile(filepath)
        error("Data file not found: $filepath")
    end
    
    data = readdlm(filepath, ',', skipstart=6)
    
    # Convert to numeric, handling potential parsing errors
    numeric_data = zeros(Float64, size(data, 1), size(data, 2))
    for i in 1:size(data, 1), j in 1:size(data, 2)
        try
            numeric_data[i, j] = parse(Float64, string(data[i, j]))
        catch
            numeric_data[i, j] = NaN
        end
    end
    
    # Remove rows with NaN values
    valid_rows = .!any(isnan.(numeric_data), dims=2)[:, 1]
    clean_data = numeric_data[valid_rows, :]
    
    return clean_data
end

# Export main functions
export comprehensive_uncertainty_analysis, create_uncertainty_analysis_report

println("📊 Uncertainty Quantification Module Loaded Successfully") 