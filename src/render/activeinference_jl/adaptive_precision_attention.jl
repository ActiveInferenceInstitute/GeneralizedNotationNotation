#!/usr/bin/env julia

"""
Adaptive Precision and Attention Mechanisms Module for ActiveInference.jl

This module provides advanced adaptive precision and attention capabilities:
- Dynamic precision modulation based on context and uncertainty
- Attention allocation and resource distribution mechanisms
- Salience-driven focus and selective processing
- Cognitive load monitoring and adaptation
- Multi-modal attention coordination
- Precision learning and adaptation over time
- Contextual modulation of precision parameters
- Hierarchical attention control across multiple levels
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates
using JSON

# Ensure required packages
for pkg in ["StatsBase", "Optim", "Clustering"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase

# ====================================
# ADAPTIVE PRECISION MECHANISMS
# ====================================

"""Calculate dynamic precision parameters based on uncertainty and context."""
function calculate_adaptive_precision(beliefs_trace::Matrix{Float64},
                                    prediction_errors::Vector{Float64},
                                    context_factors::Vector{Float64},
                                    base_precision::Float64 = 1.0)
    n_steps = size(beliefs_trace, 1)
    adaptive_precision = Dict{String, Any}()
    
    # 1. Uncertainty-based precision modulation
    belief_uncertainties = zeros(n_steps)
    for t in 1:n_steps
        belief_uncertainties[t] = shannon_entropy(beliefs_trace[t, :])
    end
    
    # Higher uncertainty → lower precision
    uncertainty_modulation = exp.(-belief_uncertainties)
    adaptive_precision["uncertainty_modulation"] = uncertainty_modulation
    
    # 2. Prediction error-based precision adaptation
    # Higher prediction errors → adjust precision
    error_magnitudes = abs.(prediction_errors)
    error_based_precision = zeros(length(error_magnitudes))
    
    for t in 1:length(error_magnitudes)
        if t == 1
            error_based_precision[t] = base_precision
        else
            # Adaptive learning rule: increase precision for consistent errors, decrease for noise
            error_consistency = cor(error_magnitudes[max(1, t-5):t], 1:min(6, t))
            if abs(error_consistency) > 0.5  # Consistent error pattern
                error_based_precision[t] = error_based_precision[t-1] * 1.1
            else  # Random noise
                error_based_precision[t] = error_based_precision[t-1] * 0.95
            end
        end
    end
    
    adaptive_precision["error_based_precision"] = error_based_precision
    
    # 3. Context-sensitive precision modulation
    if length(context_factors) == n_steps
        # Normalize context factors
        normalized_context = (context_factors .- minimum(context_factors)) ./ 
                           (maximum(context_factors) - minimum(context_factors) + 1e-6)
        
        # High-context situations may require higher precision
        context_precision = base_precision .* (1.0 .+ normalized_context)
        adaptive_precision["context_precision"] = context_precision
    end
    
    # 4. Combined adaptive precision
    combined_precision = zeros(n_steps)
    for t in 1:n_steps
        uncertainty_factor = uncertainty_modulation[t]
        error_factor = t <= length(error_based_precision) ? error_based_precision[t] : base_precision
        context_factor = haskey(adaptive_precision, "context_precision") ? 
                        adaptive_precision["context_precision"][t] : base_precision
        
        # Geometric mean of factors
        combined_precision[t] = (uncertainty_factor * error_factor * context_factor)^(1/3)
    end
    
    adaptive_precision["combined_precision"] = combined_precision
    adaptive_precision["precision_statistics"] = Dict(
        "mean_precision" => mean(combined_precision),
        "precision_variance" => var(combined_precision),
        "precision_range" => (minimum(combined_precision), maximum(combined_precision)),
        "adaptation_magnitude" => maximum(combined_precision) - minimum(combined_precision)
    )
    
    return adaptive_precision
end

"""Learn precision parameters over time using gradient-based optimization."""
function learn_precision_parameters(observations::Vector{Int},
                                  beliefs_trace::Matrix{Float64},
                                  actions::Vector{Int},
                                  initial_precision::Float64 = 1.0,
                                  learning_rate::Float64 = 0.01)
    n_steps = length(observations)
    learned_precision = Dict{String, Any}()
    
    # Initialize precision parameters
    precision_params = Dict(
        "observation_precision" => initial_precision,
        "action_precision" => initial_precision,
        "temporal_precision" => initial_precision
    )
    
    precision_trace = zeros(n_steps, 3)  # Track 3 precision parameters
    log_likelihood_trace = zeros(n_steps)
    
    for t in 1:n_steps
        # Calculate prediction errors for different modalities
        if t > 1
            # Observation prediction error
            predicted_obs_prob = beliefs_trace[t, observations[t]]
            obs_prediction_error = -log(predicted_obs_prob + 1e-12)
            
            # Action prediction error (consistency with beliefs)
            action_belief_consistency = beliefs_trace[t, actions[t]]
            action_prediction_error = -log(action_belief_consistency + 1e-12)
            
            # Temporal prediction error (belief consistency over time)
            temporal_consistency = 1.0 - norm(beliefs_trace[t, :] - beliefs_trace[t-1, :])
            temporal_prediction_error = -log(temporal_consistency + 1e-12)
            
            # Update precision parameters using gradient descent
            # Simplified gradient: precision should increase when errors are low
            precision_params["observation_precision"] += learning_rate * 
                (1.0 / (1.0 + obs_prediction_error) - 0.5)
            precision_params["action_precision"] += learning_rate * 
                (1.0 / (1.0 + action_prediction_error) - 0.5)
            precision_params["temporal_precision"] += learning_rate * 
                (1.0 / (1.0 + temporal_prediction_error) - 0.5)
            
            # Ensure precision parameters stay positive
            for key in keys(precision_params)
                precision_params[key] = max(precision_params[key], 0.01)
            end
            
            # Calculate log likelihood with current precision parameters
            log_likelihood = -(precision_params["observation_precision"] * obs_prediction_error +
                             precision_params["action_precision"] * action_prediction_error +
                             precision_params["temporal_precision"] * temporal_prediction_error)
            log_likelihood_trace[t] = log_likelihood
        end
        
        # Store current precision parameters
        precision_trace[t, 1] = precision_params["observation_precision"]
        precision_trace[t, 2] = precision_params["action_precision"]
        precision_trace[t, 3] = precision_params["temporal_precision"]
    end
    
    learned_precision["precision_trace"] = precision_trace
    learned_precision["log_likelihood_trace"] = log_likelihood_trace
    learned_precision["final_parameters"] = precision_params
    learned_precision["learning_efficiency"] = cor(1:n_steps, log_likelihood_trace)
    
    return learned_precision
end

# ====================================
# ATTENTION ALLOCATION MECHANISMS
# ====================================

"""Calculate attention weights based on salience and relevance."""
function calculate_attention_weights(stimuli_features::Matrix{Float64},
                                   current_goals::Vector{Float64},
                                   surprise_values::Vector{Float64},
                                   attention_capacity::Float64 = 1.0)
    n_stimuli, n_features = size(stimuli_features)
    attention_weights = zeros(n_stimuli)
    attention_metrics = Dict{String, Any}()
    
    # 1. Goal-relevance attention
    goal_relevance = zeros(n_stimuli)
    for i in 1:n_stimuli
        # Dot product between stimulus features and goals
        goal_relevance[i] = dot(stimuli_features[i, :], current_goals) / 
                          (norm(stimuli_features[i, :]) * norm(current_goals) + 1e-6)
    end
    
    # 2. Surprise-driven attention
    normalized_surprise = surprise_values ./ (maximum(surprise_values) + 1e-6)
    
    # 3. Feature salience (variance-based)
    feature_salience = zeros(n_stimuli)
    for i in 1:n_stimuli
        # Higher variance features get more attention
        feature_salience[i] = var(stimuli_features[i, :])
    end
    normalized_salience = feature_salience ./ (maximum(feature_salience) + 1e-6)
    
    # 4. Combine attention factors
    raw_attention = 0.4 * goal_relevance .+ 0.4 * normalized_surprise .+ 0.2 * normalized_salience
    
    # 5. Apply attention capacity constraint (softmax normalization)
    exp_attention = exp.(raw_attention .* attention_capacity)
    attention_weights = exp_attention ./ sum(exp_attention)
    
    attention_metrics["attention_weights"] = attention_weights
    attention_metrics["goal_relevance"] = goal_relevance
    attention_metrics["surprise_contribution"] = normalized_surprise
    attention_metrics["salience_contribution"] = normalized_salience
    attention_metrics["attention_entropy"] = shannon_entropy(attention_weights)
    attention_metrics["attention_focus"] = maximum(attention_weights)  # Peak attention
    attention_metrics["attention_distribution"] = var(attention_weights)  # Spread of attention
    
    return attention_metrics
end

"""Model dynamic attention allocation over time."""
function dynamic_attention_allocation(beliefs_trace::Matrix{Float64},
                                    observations::Vector{Int},
                                    actions::Vector{Int},
                                    attention_window::Int = 5)
    n_steps, n_states = size(beliefs_trace)
    attention_dynamics = Dict{String, Any}()
    
    attention_focus_trace = zeros(n_steps)
    attention_shifts = zeros(n_steps - 1)
    cognitive_load_trace = zeros(n_steps)
    
    for t in 1:n_steps
        # 1. Calculate current attention focus
        # Focus on the most likely state
        attention_focus_trace[t] = maximum(beliefs_trace[t, :])
        
        # 2. Calculate cognitive load
        # Higher entropy = higher cognitive load
        cognitive_load_trace[t] = shannon_entropy(beliefs_trace[t, :])
        
        # 3. Track attention shifts
        if t > 1
            prev_focus_state = argmax(beliefs_trace[t-1, :])
            curr_focus_state = argmax(beliefs_trace[t, :])
            attention_shifts[t-1] = prev_focus_state != curr_focus_state ? 1.0 : 0.0
        end
    end
    
    # 4. Windowed attention analysis
    windowed_metrics = []
    for t in attention_window:n_steps
        window_start = t - attention_window + 1
        window_beliefs = beliefs_trace[window_start:t, :]
        window_observations = observations[window_start:t]
        window_actions = actions[window_start:t]
        
        # Attention consistency within window
        focus_states = [argmax(window_beliefs[i, :]) for i in 1:size(window_beliefs, 1)]
        focus_consistency = length(unique(focus_states)) == 1 ? 1.0 : 
                          1.0 / length(unique(focus_states))
        
        # Attention-action alignment
        action_focus_alignment = sum(actions[window_start:t] .== focus_states) / attention_window
        
        # Surprise-driven attention changes
        surprises = zeros(attention_window - 1)
        for i in 2:attention_window
            expected_obs = argmax(window_beliefs[i-1, :])
            actual_obs = window_observations[i]
            surprises[i-1] = expected_obs != actual_obs ? 1.0 : 0.0
        end
        surprise_driven_shifts = cor(surprises, attention_shifts[window_start:t-1])
        
        push!(windowed_metrics, Dict(
            "time_window" => (window_start, t),
            "focus_consistency" => focus_consistency,
            "action_alignment" => action_focus_alignment,
            "surprise_correlation" => isnan(surprise_driven_shifts) ? 0.0 : surprise_driven_shifts
        ))
    end
    
    attention_dynamics["attention_focus_trace"] = attention_focus_trace
    attention_dynamics["cognitive_load_trace"] = cognitive_load_trace
    attention_dynamics["attention_shifts"] = attention_shifts
    attention_dynamics["windowed_metrics"] = windowed_metrics
    
    # Overall metrics
    attention_dynamics["mean_focus"] = mean(attention_focus_trace)
    attention_dynamics["mean_cognitive_load"] = mean(cognitive_load_trace)
    attention_dynamics["shift_frequency"] = sum(attention_shifts) / length(attention_shifts)
    attention_dynamics["load_variability"] = var(cognitive_load_trace)
    
    return attention_dynamics
end

# ====================================
# MULTI-MODAL ATTENTION COORDINATION
# ====================================

"""Coordinate attention across multiple modalities."""
function multimodal_attention_coordination(modality_beliefs::Vector{Matrix{Float64}},
                                         modality_precisions::Vector{Float64},
                                         cross_modal_weights::Matrix{Float64})
    n_modalities = length(modality_beliefs)
    coordination_metrics = Dict{String, Any}()
    
    if n_modalities < 2
        @warn "Need at least 2 modalities for coordination analysis"
        return coordination_metrics
    end
    
    # Ensure all modalities have same time dimension
    n_steps = minimum([size(beliefs, 1) for beliefs in modality_beliefs])
    
    # 1. Cross-modal attention weights
    attention_allocation = zeros(n_steps, n_modalities)
    
    for t in 1:n_steps
        # Calculate precision-weighted attention for each modality
        raw_weights = zeros(n_modalities)
        for m in 1:n_modalities
            if t <= size(modality_beliefs[m], 1)
                # Higher precision and lower entropy get more attention
                entropy_m = shannon_entropy(modality_beliefs[m][t, :])
                raw_weights[m] = modality_precisions[m] / (1.0 + entropy_m)
            end
        end
        
        # Apply cross-modal coordination
        coordinated_weights = cross_modal_weights * raw_weights
        
        # Normalize to sum to 1
        attention_allocation[t, :] = coordinated_weights ./ sum(coordinated_weights)
    end
    
    coordination_metrics["attention_allocation"] = attention_allocation
    
    # 2. Coordination stability
    allocation_changes = zeros(n_steps - 1, n_modalities)
    for t in 2:n_steps
        allocation_changes[t-1, :] = attention_allocation[t, :] - attention_allocation[t-1, :]
    end
    
    coordination_metrics["allocation_stability"] = 1.0 / (1.0 + mean(var(allocation_changes, dims=1)))
    
    # 3. Cross-modal synchronization
    synchronization_scores = zeros(n_modalities, n_modalities)
    for m1 in 1:n_modalities
        for m2 in 1:n_modalities
            if m1 != m2
                attention_m1 = attention_allocation[:, m1]
                attention_m2 = attention_allocation[:, m2]
                if std(attention_m1) > 1e-6 && std(attention_m2) > 1e-6
                    synchronization_scores[m1, m2] = abs(cor(attention_m1, attention_m2))
                end
            end
        end
    end
    
    coordination_metrics["synchronization_matrix"] = synchronization_scores
    coordination_metrics["mean_synchronization"] = mean(synchronization_scores[synchronization_scores .> 0])
    
    # 4. Dominant modality analysis
    dominant_modality_trace = [argmax(attention_allocation[t, :]) for t in 1:n_steps]
    modality_dominance = zeros(n_modalities)
    for m in 1:n_modalities
        modality_dominance[m] = sum(dominant_modality_trace .== m) / n_steps
    end
    
    coordination_metrics["modality_dominance"] = modality_dominance
    coordination_metrics["dominance_entropy"] = shannon_entropy(modality_dominance)
    
    return coordination_metrics
end

# ====================================
# COMPREHENSIVE ADAPTIVE PRECISION ANALYSIS
# ====================================

"""Run comprehensive adaptive precision and attention analysis."""
function comprehensive_precision_attention_analysis(output_dir::String)
    println("🎯 Running Comprehensive Adaptive Precision and Attention Analysis")
    
    # Create output directory
    precision_dir = joinpath(output_dir, "adaptive_precision_attention")
    mkpath(precision_dir)
    
    analysis_results = Dict{String, Any}()
    
    try
        # Load basic simulation data
        basic_data_path = joinpath(output_dir, "simulation_results", "basic_simulation.csv")
        if isfile(basic_data_path)
            basic_data = load_statistical_data(basic_data_path)
            
            if size(basic_data, 2) >= 4
                steps = basic_data[:, 1]
                observations = Int.(basic_data[:, 2])
                actions = Int.(basic_data[:, 3])
                beliefs = basic_data[:, 4]
                
                # Create beliefs trace matrix (binary case for demonstration)
                beliefs_trace = hcat(beliefs, 1.0 .- beliefs)
                
                # Generate prediction errors and context factors
                prediction_errors = zeros(length(observations))
                for t in 2:length(observations)
                    predicted_obs = argmax(beliefs_trace[t-1, :])
                    actual_obs = observations[t]
                    prediction_errors[t] = abs(predicted_obs - actual_obs)
                end
                
                context_factors = sin.(steps .* 0.1) .+ rand(length(steps)) * 0.2  # Simulated context
                
                # 1. Adaptive precision analysis
                precision_results = calculate_adaptive_precision(
                    beliefs_trace, prediction_errors, context_factors
                )
                analysis_results["adaptive_precision"] = precision_results
                
                # 2. Precision learning analysis
                precision_learning = learn_precision_parameters(
                    observations, beliefs_trace, actions
                )
                analysis_results["precision_learning"] = precision_learning
                
                # 3. Attention dynamics analysis
                attention_dynamics = dynamic_attention_allocation(
                    beliefs_trace, observations, actions
                )
                analysis_results["attention_dynamics"] = attention_dynamics
                
                println("✅ Basic precision and attention analysis completed")
            end
        end
        
        # 4. Multi-modal analysis (if multi-modal data available)
        multimodal_path = joinpath(output_dir, "simulation_results", "multimodal_simulation.csv")
        if isfile(multimodal_path)
            try
                multimodal_data = load_statistical_data(multimodal_path)
                
                # Create mock multi-modal beliefs for demonstration
                n_steps = min(50, size(multimodal_data, 1))
                modality1_beliefs = rand(n_steps, 3)  # 3-state visual modality
                modality2_beliefs = rand(n_steps, 2)  # 2-state auditory modality
                
                # Normalize to proper probability distributions
                for t in 1:n_steps
                    modality1_beliefs[t, :] ./= sum(modality1_beliefs[t, :])
                    modality2_beliefs[t, :] ./= sum(modality2_beliefs[t, :])
                end
                
                modality_beliefs = [modality1_beliefs, modality2_beliefs]
                modality_precisions = [1.5, 1.0]  # Visual more precise than auditory
                cross_modal_weights = [1.0 0.3; 0.3 1.0]  # Some cross-modal influence
                
                multimodal_results = multimodal_attention_coordination(
                    modality_beliefs, modality_precisions, cross_modal_weights
                )
                analysis_results["multimodal_coordination"] = multimodal_results
                
                println("✅ Multi-modal attention coordination analysis completed")
            catch e
                @warn "Multi-modal analysis failed: $e"
            end
        end
        
    catch e
        @error "Precision and attention analysis failed: $e"
        analysis_results["error"] = string(e)
    end
    
    # Save comprehensive results
    if !isempty(analysis_results)
        # Save as JSON
        json_path = joinpath(precision_dir, "precision_attention_results.json")
        open(json_path, "w") do f
            JSON.print(f, analysis_results, 2)
        end
        
        # Create summary report
        create_precision_attention_report(analysis_results, precision_dir)
        
        println("🎯 Precision and attention analysis results saved to: $precision_dir")
    end
    
    return analysis_results
end

"""Create a comprehensive precision and attention analysis report."""
function create_precision_attention_report(results::Dict{String, Any}, output_dir::String)
    report_path = joinpath(output_dir, "precision_attention_report.md")
    
    open(report_path, "w") do f
        println(f, "# Adaptive Precision and Attention Analysis Report")
        println(f, "")
        println(f, "Generated: $(now())")
        println(f, "")
        
        if haskey(results, "adaptive_precision")
            precision = results["adaptive_precision"]
            stats = get(precision, "precision_statistics", Dict())
            
            println(f, "## Adaptive Precision Analysis")
            println(f, "")
            println(f, "- **Mean Precision**: $(get(stats, "mean_precision", "N/A"))")
            println(f, "- **Precision Variance**: $(get(stats, "precision_variance", "N/A"))")
            println(f, "- **Adaptation Magnitude**: $(get(stats, "adaptation_magnitude", "N/A"))")
            if haskey(stats, "precision_range")
                range_val = stats["precision_range"]
                println(f, "- **Precision Range**: $(range_val[1]) to $(range_val[2])")
            end
            println(f, "")
        end
        
        if haskey(results, "precision_learning")
            learning = results["precision_learning"]
            println(f, "## Precision Learning")
            println(f, "")
            println(f, "- **Learning Efficiency**: $(get(learning, "learning_efficiency", "N/A"))")
            if haskey(learning, "final_parameters")
                params = learning["final_parameters"]
                println(f, "- **Final Observation Precision**: $(get(params, "observation_precision", "N/A"))")
                println(f, "- **Final Action Precision**: $(get(params, "action_precision", "N/A"))")
                println(f, "- **Final Temporal Precision**: $(get(params, "temporal_precision", "N/A"))")
            end
            println(f, "")
        end
        
        if haskey(results, "attention_dynamics")
            attention = results["attention_dynamics"]
            println(f, "## Attention Dynamics")
            println(f, "")
            println(f, "- **Mean Focus**: $(get(attention, "mean_focus", "N/A"))")
            println(f, "- **Mean Cognitive Load**: $(get(attention, "mean_cognitive_load", "N/A"))")
            println(f, "- **Attention Shift Frequency**: $(get(attention, "shift_frequency", "N/A"))")
            println(f, "- **Load Variability**: $(get(attention, "load_variability", "N/A"))")
            println(f, "")
        end
        
        if haskey(results, "multimodal_coordination")
            multimodal = results["multimodal_coordination"]
            println(f, "## Multi-Modal Attention Coordination")
            println(f, "")
            println(f, "- **Allocation Stability**: $(get(multimodal, "allocation_stability", "N/A"))")
            println(f, "- **Mean Synchronization**: $(get(multimodal, "mean_synchronization", "N/A"))")
            println(f, "- **Dominance Entropy**: $(get(multimodal, "dominance_entropy", "N/A"))")
            if haskey(multimodal, "modality_dominance")
                dominance = multimodal["modality_dominance"]
                for (i, dom) in enumerate(dominance)
                    println(f, "- **Modality $i Dominance**: $(round(dom, digits=3))")
                end
            end
            println(f, "")
        end
        
        println(f, "## Summary")
        println(f, "")
        println(f, "This analysis examines adaptive precision mechanisms and attention allocation")
        println(f, "in POMDP reasoning, including dynamic precision modulation, attention dynamics,")
        println(f, "and multi-modal coordination capabilities.")
    end
    
    println("📋 Precision and attention report saved: $report_path")
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
export comprehensive_precision_attention_analysis, create_precision_attention_report

println("🎯 Adaptive Precision and Attention Module Loaded Successfully") 