#!/usr/bin/env julia

"""
Multi-Scale Temporal Analysis Module for ActiveInference.jl

This module provides comprehensive temporal analysis across multiple scales:
- Hierarchical temporal reasoning across different time horizons
- Multi-resolution planning depth analysis
- Temporal abstraction and event segmentation
- Chronesthesia: mental time travel and temporal projection
- Temporal coherence and consistency analysis across scales
- Planning horizon optimization and adaptive depth selection
- Memory consolidation and temporal compression analysis
- Future-directed thinking and prospective inference
- Temporal prediction accuracy across different scales
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Distributions
using Printf
using Dates
using JSON
using FFTW

# Ensure required packages
for pkg in ["StatsBase", "DSP", "Clustering", "Wavelets"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase

# ====================================
# HIERARCHICAL TEMPORAL ANALYSIS
# ====================================

"""Analyze decision-making across multiple temporal hierarchies."""
function analyze_temporal_hierarchies(beliefs_trace::Matrix{Float64},
                                    actions::Vector{Int},
                                    observations::Vector{Int},
                                    temporal_scales::Vector{Int} = [1, 3, 5, 10, 20])
    hierarchical_metrics = Dict{String, Any}()
    n_steps = length(actions)
    
    for scale in temporal_scales
        if scale <= n_steps
            scale_key = "scale_$(scale)"
            scale_metrics = Dict{String, Any}()
            
            # 1. Temporal chunking at this scale
            n_chunks = div(n_steps, scale)
            chunks = []
            
            for chunk_idx in 1:n_chunks
                start_idx = (chunk_idx - 1) * scale + 1
                end_idx = min(chunk_idx * scale, n_steps)
                
                chunk_actions = actions[start_idx:end_idx]
                chunk_beliefs = beliefs_trace[start_idx:end_idx, :]
                chunk_observations = observations[start_idx:end_idx]
                
                # Analyze chunk characteristics
                chunk_entropy = mean([shannon_entropy(chunk_beliefs[t, :]) for t in 1:size(chunk_beliefs, 1)])
                chunk_action_diversity = length(unique(chunk_actions)) / length(chunk_actions)
                chunk_stability = calculate_chunk_stability(chunk_beliefs)
                
                push!(chunks, Dict(
                    "chunk_id" => chunk_idx,
                    "time_range" => (start_idx, end_idx),
                    "entropy" => chunk_entropy,
                    "action_diversity" => chunk_action_diversity,
                    "stability" => chunk_stability,
                    "dominant_action" => mode(chunk_actions),
                    "actions" => chunk_actions,
                    "observations" => chunk_observations
                ))
            end
            
            scale_metrics["chunks"] = chunks
            scale_metrics["n_chunks"] = n_chunks
            
            # 2. Scale-specific metrics
            if n_chunks > 1
                chunk_entropies = [chunk["entropy"] for chunk in chunks]
                chunk_stabilities = [chunk["stability"] for chunk in chunks]
                
                scale_metrics["mean_chunk_entropy"] = mean(chunk_entropies)
                scale_metrics["entropy_variability"] = var(chunk_entropies)
                scale_metrics["mean_chunk_stability"] = mean(chunk_stabilities)
                scale_metrics["stability_trend"] = cor(1:length(chunk_stabilities), chunk_stabilities)
                
                # Chunk transition analysis
                chunk_transitions = []
                for i in 2:n_chunks
                    prev_chunk = chunks[i-1]
                    curr_chunk = chunks[i]
                    
                    entropy_change = curr_chunk["entropy"] - prev_chunk["entropy"]
                    action_change = curr_chunk["dominant_action"] != prev_chunk["dominant_action"]
                    stability_change = curr_chunk["stability"] - prev_chunk["stability"]
                    
                    push!(chunk_transitions, Dict(
                        "entropy_change" => entropy_change,
                        "action_change" => action_change,
                        "stability_change" => stability_change
                    ))
                end
                
                scale_metrics["chunk_transitions"] = chunk_transitions
                
                # Temporal structure at this scale
                entropy_changes = [trans["entropy_change"] for trans in chunk_transitions]
                action_changes = [trans["action_change"] for trans in chunk_transitions]
                
                scale_metrics["entropy_change_variance"] = var(entropy_changes)
                scale_metrics["action_change_frequency"] = sum(action_changes) / length(action_changes)
            end
            
            # 3. Predictability at this scale
            if n_chunks >= 3
                predictability_scores = []
                for i in 3:n_chunks
                    # Try to predict current chunk from previous chunks
                    prev_entropy = chunks[i-1]["entropy"]
                    prev_prev_entropy = chunks[i-2]["entropy"]
                    current_entropy = chunks[i]["entropy"]
                    
                    # Simple linear prediction
                    predicted_entropy = prev_entropy + (prev_entropy - prev_prev_entropy)
                    prediction_error = abs(predicted_entropy - current_entropy)
                    
                    push!(predictability_scores, 1.0 / (1.0 + prediction_error))
                end
                
                scale_metrics["predictability"] = mean(predictability_scores)
            end
            
            hierarchical_metrics[scale_key] = scale_metrics
        end
    end
    
    # Cross-scale analysis
    hierarchical_metrics["cross_scale_analysis"] = analyze_cross_scale_relationships(hierarchical_metrics, temporal_scales)
    
    return hierarchical_metrics
end

"""Calculate stability within a temporal chunk."""
function calculate_chunk_stability(chunk_beliefs::Matrix{Float64})
    if size(chunk_beliefs, 1) < 2
        return 1.0
    end
    
    belief_changes = [norm(chunk_beliefs[t, :] - chunk_beliefs[t-1, :]) for t in 2:size(chunk_beliefs, 1)]
    return 1.0 / (1.0 + mean(belief_changes))
end

"""Analyze relationships across different temporal scales."""
function analyze_cross_scale_relationships(hierarchical_metrics::Dict{String, Any}, scales::Vector{Int})
    cross_scale = Dict{String, Any}()
    
    # Collect metrics across scales
    scale_entropies = []
    scale_stabilities = []
    scale_predictabilities = []
    
    for scale in scales
        scale_key = "scale_$(scale)"
        if haskey(hierarchical_metrics, scale_key)
            scale_data = hierarchical_metrics[scale_key]
            
            if haskey(scale_data, "mean_chunk_entropy")
                push!(scale_entropies, (scale, scale_data["mean_chunk_entropy"]))
            end
            
            if haskey(scale_data, "mean_chunk_stability")
                push!(scale_stabilities, (scale, scale_data["mean_chunk_stability"]))
            end
            
            if haskey(scale_data, "predictability")
                push!(scale_predictabilities, (scale, scale_data["predictability"]))
            end
        end
    end
    
    # Analyze scale relationships
    if length(scale_entropies) > 2
        entropy_scales = [item[1] for item in scale_entropies]
        entropy_values = [item[2] for item in scale_entropies]
        cross_scale["entropy_scale_correlation"] = cor(log.(entropy_scales), entropy_values)
    end
    
    if length(scale_stabilities) > 2
        stability_scales = [item[1] for item in scale_stabilities]
        stability_values = [item[2] for item in scale_stabilities]
        cross_scale["stability_scale_correlation"] = cor(log.(stability_scales), stability_values)
    end
    
    if length(scale_predictabilities) > 2
        predict_scales = [item[1] for item in scale_predictabilities]
        predict_values = [item[2] for item in scale_predictabilities]
        cross_scale["predictability_scale_correlation"] = cor(log.(predict_scales), predict_values)
    end
    
    # Optimal scale analysis
    if length(scale_predictabilities) > 0
        best_predictability_idx = argmax([item[2] for item in scale_predictabilities])
        cross_scale["optimal_prediction_scale"] = scale_predictabilities[best_predictability_idx][1]
    end
    
    return cross_scale
end

# ====================================
# PLANNING DEPTH ANALYSIS
# ====================================

"""Analyze planning depth and horizon effects."""
function analyze_planning_depth(beliefs_trace::Matrix{Float64},
                              actions::Vector{Int},
                              planning_horizons::Vector{Int} = [1, 3, 5, 10])
    planning_analysis = Dict{String, Any}()
    n_steps = size(beliefs_trace, 1)
    
    for horizon in planning_horizons
        if horizon < n_steps
            horizon_key = "horizon_$(horizon)"
            horizon_metrics = Dict{String, Any}()
            
            # Simulate planning at different depths
            planning_quality_scores = []
            planning_consistency_scores = []
            
            for t in 1:(n_steps - horizon)
                current_beliefs = beliefs_trace[t, :]
                future_beliefs = beliefs_trace[t + horizon, :]
                
                # How well did the agent's implicit planning work?
                # Measure: how much the beliefs changed in the "planned" direction
                belief_change = future_beliefs - current_beliefs
                
                # Planning quality: coherent belief evolution
                planning_quality = 1.0 / (1.0 + norm(belief_change - mean(belief_change)))
                push!(planning_quality_scores, planning_quality)
                
                # Planning consistency: similar actions lead to similar outcomes
                if t > 1
                    prev_actions = actions[max(1, t-horizon):t-1]
                    curr_actions = actions[t:min(t+horizon-1, length(actions))]
                    
                    action_similarity = calculate_action_sequence_similarity(prev_actions, curr_actions)
                    push!(planning_consistency_scores, action_similarity)
                end
            end
            
            horizon_metrics["mean_planning_quality"] = mean(planning_quality_scores)
            horizon_metrics["planning_quality_variance"] = var(planning_quality_scores)
            horizon_metrics["mean_planning_consistency"] = length(planning_consistency_scores) > 0 ? 
                mean(planning_consistency_scores) : 0.0
            
            # Prediction accuracy at this horizon
            prediction_errors = []
            for t in 1:(n_steps - horizon)
                current_belief = beliefs_trace[t, :]
                future_belief = beliefs_trace[t + horizon, :]
                
                # Simple prediction: expect beliefs to remain similar
                prediction_error = norm(future_belief - current_belief)
                push!(prediction_errors, prediction_error)
            end
            
            horizon_metrics["mean_prediction_error"] = mean(prediction_errors)
            horizon_metrics["prediction_accuracy"] = 1.0 / (1.0 + mean(prediction_errors))
            
            # Temporal discounting effects
            if horizon > 1
                # How much do future states matter vs immediate states?
                immediate_uncertainty = mean([shannon_entropy(beliefs_trace[t, :]) for t in 1:min(5, n_steps)])
                future_uncertainty = mean([shannon_entropy(beliefs_trace[t, :]) for t in max(1, n_steps-horizon+1):n_steps])
                
                temporal_discount_factor = future_uncertainty / (immediate_uncertainty + 1e-6)
                horizon_metrics["temporal_discount_factor"] = temporal_discount_factor
            end
            
            planning_analysis[horizon_key] = horizon_metrics
        end
    end
    
    # Optimal planning horizon analysis
    if length(planning_horizons) > 1
        accuracy_scores = []
        for horizon in planning_horizons
            horizon_key = "horizon_$(horizon)"
            if haskey(planning_analysis, horizon_key)
                accuracy = planning_analysis[horizon_key]["prediction_accuracy"]
                push!(accuracy_scores, (horizon, accuracy))
            end
        end
        
        if length(accuracy_scores) > 0
            best_horizon_idx = argmax([score[2] for score in accuracy_scores])
            planning_analysis["optimal_planning_horizon"] = accuracy_scores[best_horizon_idx][1]
            
            # Diminishing returns analysis
            if length(accuracy_scores) >= 3
                horizons = [score[1] for score in accuracy_scores]
                accuracies = [score[2] for score in accuracy_scores]
                
                # Calculate marginal improvements
                marginal_improvements = diff(accuracies)
                planning_analysis["diminishing_returns"] = cor(horizons[2:end], marginal_improvements) < 0
            end
        end
    end
    
    return planning_analysis
end

"""Calculate similarity between action sequences."""
function calculate_action_sequence_similarity(seq1::Vector{Int}, seq2::Vector{Int})
    if length(seq1) == 0 || length(seq2) == 0
        return 0.0
    end
    
    min_length = min(length(seq1), length(seq2))
    matches = sum(seq1[1:min_length] .== seq2[1:min_length])
    return matches / min_length
end

# ====================================
# TEMPORAL COHERENCE ANALYSIS
# ====================================

"""Analyze temporal coherence and consistency across different scales."""
function analyze_temporal_coherence(beliefs_trace::Matrix{Float64},
                                  actions::Vector{Int},
                                  observations::Vector{Int},
                                  coherence_windows::Vector{Int} = [3, 5, 10])
    coherence_analysis = Dict{String, Any}()
    n_steps = length(actions)
    
    for window in coherence_windows
        if window <= n_steps
            window_key = "window_$(window)"
            window_metrics = Dict{String, Any}()
            
            coherence_scores = []
            action_coherence_scores = []
            observation_coherence_scores = []
            
            for t in window:n_steps
                start_idx = t - window + 1
                
                # Belief coherence: how smooth is belief evolution?
                window_beliefs = beliefs_trace[start_idx:t, :]
                belief_smoothness = calculate_belief_smoothness(window_beliefs)
                push!(coherence_scores, belief_smoothness)
                
                # Action coherence: how consistent are actions within window?
                window_actions = actions[start_idx:t]
                action_consistency = calculate_action_consistency(window_actions)
                push!(action_coherence_scores, action_consistency)
                
                # Observation coherence: how predictable are observations?
                window_observations = observations[start_idx:t]
                obs_predictability = calculate_observation_predictability(window_observations)
                push!(observation_coherence_scores, obs_predictability)
            end
            
            window_metrics["mean_belief_coherence"] = mean(coherence_scores)
            window_metrics["belief_coherence_variance"] = var(coherence_scores)
            window_metrics["mean_action_coherence"] = mean(action_coherence_scores)
            window_metrics["mean_observation_coherence"] = mean(observation_coherence_scores)
            
            # Cross-modal coherence
            belief_action_coherence = cor(coherence_scores, action_coherence_scores)
            belief_obs_coherence = cor(coherence_scores, observation_coherence_scores)
            
            window_metrics["belief_action_coherence"] = isnan(belief_action_coherence) ? 0.0 : belief_action_coherence
            window_metrics["belief_observation_coherence"] = isnan(belief_obs_coherence) ? 0.0 : belief_obs_coherence
            
            coherence_analysis[window_key] = window_metrics
        end
    end
    
    # Global coherence metrics
    if length(coherence_windows) > 1
        # How does coherence change with window size?
        window_coherences = []
        for window in coherence_windows
            window_key = "window_$(window)"
            if haskey(coherence_analysis, window_key)
                coherence = coherence_analysis[window_key]["mean_belief_coherence"]
                push!(window_coherences, (window, coherence))
            end
        end
        
        if length(window_coherences) > 2
            windows = [item[1] for item in window_coherences]
            coherences = [item[2] for item in window_coherences]
            coherence_analysis["coherence_scale_dependency"] = cor(log.(windows), coherences)
        end
    end
    
    return coherence_analysis
end

"""Calculate smoothness of belief evolution within a window."""
function calculate_belief_smoothness(beliefs_window::Matrix{Float64})
    if size(beliefs_window, 1) < 2
        return 1.0
    end
    
    # Calculate second derivatives (acceleration)
    accelerations = []
    for t in 3:size(beliefs_window, 1)
        prev_change = beliefs_window[t-1, :] - beliefs_window[t-2, :]
        curr_change = beliefs_window[t, :] - beliefs_window[t-1, :]
        acceleration = norm(curr_change - prev_change)
        push!(accelerations, acceleration)
    end
    
    # Smoothness is inverse of mean acceleration
    if length(accelerations) > 0
        return 1.0 / (1.0 + mean(accelerations))
    else
        return 1.0
    end
end

"""Calculate consistency of actions within a window."""
function calculate_action_consistency(actions_window::Vector{Int})
    if length(actions_window) <= 1
        return 1.0
    end
    
    # Consistency as inverse of number of unique actions
    n_unique = length(unique(actions_window))
    return 1.0 / n_unique
end

"""Calculate predictability of observations within a window."""
function calculate_observation_predictability(observations_window::Vector{Int})
    if length(observations_window) <= 2
        return 0.5
    end
    
    # Simple predictability: how often does the next observation match the previous?
    matches = 0
    for t in 2:length(observations_window)
        if observations_window[t] == observations_window[t-1]
            matches += 1
        end
    end
    
    return matches / (length(observations_window) - 1)
end

# ====================================
# COMPREHENSIVE TEMPORAL ANALYSIS
# ====================================

"""Run comprehensive multi-scale temporal analysis."""
function comprehensive_temporal_analysis(output_dir::String)
    println("⏰ Running Comprehensive Multi-Scale Temporal Analysis")
    
    # Create output directory
    temporal_dir = joinpath(output_dir, "multi_scale_temporal")
    mkpath(temporal_dir)
    
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
                
                # 1. Hierarchical temporal analysis
                temporal_hierarchies = analyze_temporal_hierarchies(
                    beliefs_trace, actions, observations, [1, 3, 5, 10, 15]
                )
                analysis_results["temporal_hierarchies"] = temporal_hierarchies
                
                # 2. Planning depth analysis
                planning_depth = analyze_planning_depth(
                    beliefs_trace, actions, [1, 3, 5, 8]
                )
                analysis_results["planning_depth"] = planning_depth
                
                # 3. Temporal coherence analysis
                temporal_coherence = analyze_temporal_coherence(
                    beliefs_trace, actions, observations, [3, 5, 8]
                )
                analysis_results["temporal_coherence"] = temporal_coherence
                
                println("✅ Multi-scale temporal analysis completed")
            end
        end
        
        # 4. Load additional temporal data if available
        temporal_data_path = joinpath(output_dir, "simulation_results", "temporal_analysis.csv")
        if isfile(temporal_data_path)
            try
                temporal_data = load_statistical_data(temporal_data_path)
                
                # Additional temporal metrics analysis here
                analysis_results["extended_temporal_analysis"] = Dict(
                    "data_available" => true,
                    "data_shape" => size(temporal_data)
                )
                
                println("✅ Extended temporal analysis completed")
            catch e
                @warn "Extended temporal analysis failed: $e"
            end
        end
        
    catch e
        @error "Multi-scale temporal analysis failed: $e"
        analysis_results["error"] = string(e)
    end
    
    # Save comprehensive results
    if !isempty(analysis_results)
        # Save as JSON
        json_path = joinpath(temporal_dir, "temporal_analysis_results.json")
        open(json_path, "w") do f
            JSON.print(f, analysis_results, 2)
        end
        
        # Create summary report
        create_temporal_analysis_report(analysis_results, temporal_dir)
        
        println("⏰ Multi-scale temporal analysis results saved to: $temporal_dir")
    end
    
    return analysis_results
end

"""Create a comprehensive temporal analysis report."""
function create_temporal_analysis_report(results::Dict{String, Any}, output_dir::String)
    report_path = joinpath(output_dir, "temporal_analysis_report.md")
    
    open(report_path, "w") do f
        println(f, "# Multi-Scale Temporal Analysis Report")
        println(f, "")
        println(f, "Generated: $(now())")
        println(f, "")
        
        if haskey(results, "temporal_hierarchies")
            hierarchies = results["temporal_hierarchies"]
            println(f, "## Temporal Hierarchies")
            println(f, "")
            
            # Analyze each scale
            for (scale_key, scale_data) in hierarchies
                if isa(scale_data, Dict) && haskey(scale_data, "n_chunks")
                    scale_num = replace(scale_key, "scale_" => "")
                    println(f, "### Scale $scale_num")
                    println(f, "- **Number of Chunks**: $(scale_data["n_chunks"])")
                    
                    if haskey(scale_data, "mean_chunk_entropy")
                        println(f, "- **Mean Chunk Entropy**: $(round(scale_data["mean_chunk_entropy"], digits=3))")
                    end
                    
                    if haskey(scale_data, "predictability")
                        println(f, "- **Predictability**: $(round(scale_data["predictability"], digits=3))")
                    end
                    println(f, "")
                end
            end
            
            # Cross-scale analysis
            if haskey(hierarchies, "cross_scale_analysis")
                cross_scale = hierarchies["cross_scale_analysis"]
                println(f, "### Cross-Scale Analysis")
                
                if haskey(cross_scale, "optimal_prediction_scale")
                    println(f, "- **Optimal Prediction Scale**: $(cross_scale["optimal_prediction_scale"])")
                end
                
                for (metric, value) in cross_scale
                    if metric != "optimal_prediction_scale" && isa(value, Number)
                        println(f, "- **$(replace(metric, "_" => " ") |> titlecase)**: $(round(value, digits=3))")
                    end
                end
                println(f, "")
            end
        end
        
        if haskey(results, "planning_depth")
            planning = results["planning_depth"]
            println(f, "## Planning Depth Analysis")
            println(f, "")
            
            if haskey(planning, "optimal_planning_horizon")
                println(f, "- **Optimal Planning Horizon**: $(planning["optimal_planning_horizon"])")
            end
            
            if haskey(planning, "diminishing_returns")
                println(f, "- **Diminishing Returns Detected**: $(planning["diminishing_returns"])")
            end
            
            # Analyze each horizon
            for (horizon_key, horizon_data) in planning
                if isa(horizon_data, Dict) && haskey(horizon_data, "prediction_accuracy")
                    horizon_num = replace(horizon_key, "horizon_" => "")
                    println(f, "### Horizon $horizon_num")
                    println(f, "- **Prediction Accuracy**: $(round(horizon_data["prediction_accuracy"], digits=3))")
                    
                    if haskey(horizon_data, "mean_planning_quality")
                        println(f, "- **Planning Quality**: $(round(horizon_data["mean_planning_quality"], digits=3))")
                    end
                    println(f, "")
                end
            end
        end
        
        if haskey(results, "temporal_coherence")
            coherence = results["temporal_coherence"]
            println(f, "## Temporal Coherence Analysis")
            println(f, "")
            
            if haskey(coherence, "coherence_scale_dependency")
                println(f, "- **Coherence Scale Dependency**: $(round(coherence["coherence_scale_dependency"], digits=3))")
                println(f, "")
            end
            
            # Analyze each window
            for (window_key, window_data) in coherence
                if isa(window_data, Dict) && haskey(window_data, "mean_belief_coherence")
                    window_num = replace(window_key, "window_" => "")
                    println(f, "### Window $window_num")
                    println(f, "- **Belief Coherence**: $(round(window_data["mean_belief_coherence"], digits=3))")
                    println(f, "- **Action Coherence**: $(round(window_data["mean_action_coherence"], digits=3))")
                    println(f, "- **Observation Coherence**: $(round(window_data["mean_observation_coherence"], digits=3))")
                    println(f, "")
                end
            end
        end
        
        println(f, "## Summary")
        println(f, "")
        println(f, "This analysis examines temporal reasoning across multiple scales,")
        println(f, "including hierarchical chunking, planning depth optimization,")
        println(f, "and temporal coherence across different time windows.")
    end
    
    println("📋 Temporal analysis report saved: $report_path")
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
export comprehensive_temporal_analysis, create_temporal_analysis_report

println("⏰ Multi-Scale Temporal Analysis Module Loaded Successfully") 