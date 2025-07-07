#!/usr/bin/env julia

"""
Meta-Cognitive Analysis Module for ActiveInference.jl

This module provides advanced meta-cognitive capabilities for hierarchical POMDP understanding:
- Higher-order belief monitoring and meta-awareness assessment
- Confidence estimation and uncertainty decomposition (epistemic vs aleatoric)
- Meta-learning: learning about learning processes and adaptation strategies
- Hierarchical temporal abstraction with multiple time scales
- Theory of mind modeling for multi-agent scenarios
- Metacognitive control and strategy selection
- Self-reflection mechanisms and introspective analysis
- Cognitive load assessment and resource allocation optimization
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
for pkg in ["StatsBase", "Clustering", "MultivariateStats", "Optim"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase

# ====================================
# META-COGNITIVE AWARENESS METRICS
# ====================================

"""Calculate meta-cognitive awareness score based on belief confidence and accuracy."""
function calculate_metacognitive_awareness(beliefs_trace::Matrix{Float64}, 
                                         observations::Vector{Int},
                                         confidence_trace::Vector{Float64})
    n_steps = length(observations)
    awareness_metrics = Dict{String, Any}()
    
    # 1. Confidence-accuracy correlation (Metacognitive sensitivity)
    prediction_accuracy = zeros(n_steps)
    for t in 1:n_steps
        predicted_state = argmax(beliefs_trace[t, :])
        actual_observation = observations[t]
        prediction_accuracy[t] = (predicted_state == actual_observation) ? 1.0 : 0.0
    end
    
    if length(confidence_trace) == n_steps && std(confidence_trace) > 1e-6
        confidence_accuracy_corr = cor(confidence_trace, prediction_accuracy)
        awareness_metrics["metacognitive_sensitivity"] = confidence_accuracy_corr
    else
        awareness_metrics["metacognitive_sensitivity"] = 0.0
    end
    
    # 2. Calibration analysis (how well confidence matches accuracy)
    confidence_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    calibration_error = 0.0
    calibration_data = []
    
    for i in 1:(length(confidence_bins)-1)
        bin_mask = (confidence_trace .>= confidence_bins[i]) .& (confidence_trace .< confidence_bins[i+1])
        if sum(bin_mask) > 0
            bin_confidence = mean(confidence_trace[bin_mask])
            bin_accuracy = mean(prediction_accuracy[bin_mask])
            bin_count = sum(bin_mask)
            
            calibration_error += abs(bin_confidence - bin_accuracy) * bin_count
            push!(calibration_data, (bin_confidence, bin_accuracy, bin_count))
        end
    end
    
    awareness_metrics["calibration_error"] = calibration_error / n_steps
    awareness_metrics["calibration_data"] = calibration_data
    
    # 3. Meta-uncertainty: uncertainty about uncertainty
    belief_entropy = [shannon_entropy(beliefs_trace[t, :]) for t in 1:n_steps]
    meta_uncertainty = std(belief_entropy)
    awareness_metrics["meta_uncertainty"] = meta_uncertainty
    
    # 4. Confidence resolution (ability to distinguish between correct/incorrect)
    if length(unique(prediction_accuracy)) > 1
        confidence_resolution = abs(mean(confidence_trace[prediction_accuracy .== 1]) - 
                                  mean(confidence_trace[prediction_accuracy .== 0]))
        awareness_metrics["confidence_resolution"] = confidence_resolution
    else
        awareness_metrics["confidence_resolution"] = 0.0
    end
    
    return awareness_metrics
end

"""Analyze higher-order beliefs: beliefs about beliefs."""
function analyze_higher_order_beliefs(beliefs_trace::Matrix{Float64}, 
                                    belief_change_trace::Matrix{Float64})
    n_steps, n_states = size(beliefs_trace)
    higher_order_metrics = Dict{String, Any}()
    
    # 1. Belief stability tracking
    belief_volatility = [norm(belief_change_trace[t, :]) for t in 1:n_steps]
    higher_order_metrics["belief_volatility_trace"] = belief_volatility
    higher_order_metrics["mean_volatility"] = mean(belief_volatility)
    higher_order_metrics["volatility_trend"] = length(belief_volatility) > 1 ? 
        cor(1:length(belief_volatility), belief_volatility) : 0.0
    
    # 2. Belief coherence across states
    coherence_scores = zeros(n_steps)
    for t in 1:n_steps
        # Coherence as inverse of entropy (more peaked = more coherent)
        coherence_scores[t] = 1.0 / (1.0 + shannon_entropy(beliefs_trace[t, :]))
    end
    higher_order_metrics["coherence_trace"] = coherence_scores
    higher_order_metrics["mean_coherence"] = mean(coherence_scores)
    
    # 3. Meta-prediction: predicting belief changes
    if n_steps > 5
        meta_prediction_error = zeros(n_steps - 1)
        for t in 2:n_steps
            # Simple predictor: expect belief to continue changing in same direction
            if t > 2
                predicted_change = belief_change_trace[t-1, :]
                actual_change = belief_change_trace[t, :]
                meta_prediction_error[t-1] = norm(predicted_change - actual_change)
            end
        end
        higher_order_metrics["meta_prediction_error"] = mean(meta_prediction_error[2:end])
    end
    
    # 4. Belief momentum and acceleration
    belief_momentum = zeros(n_steps-1, n_states)
    belief_acceleration = zeros(n_steps-2, n_states)
    
    for t in 2:n_steps
        belief_momentum[t-1, :] = beliefs_trace[t, :] - beliefs_trace[t-1, :]
    end
    
    for t in 3:n_steps
        belief_acceleration[t-2, :] = belief_momentum[t-1, :] - belief_momentum[t-2, :]
    end
    
    higher_order_metrics["belief_momentum"] = belief_momentum
    higher_order_metrics["belief_acceleration"] = belief_acceleration
    higher_order_metrics["momentum_magnitude"] = [norm(belief_momentum[t, :]) for t in 1:size(belief_momentum, 1)]
    higher_order_metrics["acceleration_magnitude"] = [norm(belief_acceleration[t, :]) for t in 1:size(belief_acceleration, 1)]
    
    return higher_order_metrics
end

# ====================================
# HIERARCHICAL TEMPORAL ANALYSIS
# ====================================

"""Analyze decision-making across multiple temporal scales."""
function multi_scale_temporal_analysis(actions::Vector{Int}, 
                                     beliefs_trace::Matrix{Float64},
                                     time_scales::Vector{Int} = [1, 5, 10, 20])
    temporal_metrics = Dict{String, Any}()
    n_steps = length(actions)
    
    for scale in time_scales
        if scale <= n_steps
            scale_key = "scale_$(scale)"
            temporal_metrics[scale_key] = Dict{String, Any}()
            
            # 1. Action consistency at this scale
            action_windows = [actions[max(1, i-scale+1):i] for i in scale:n_steps]
            action_consistency = zeros(length(action_windows))
            
            for (idx, window) in enumerate(action_windows)
                # Consistency as inverse of action variance
                if length(unique(window)) == 1
                    action_consistency[idx] = 1.0
                else
                    action_consistency[idx] = 1.0 / length(unique(window))
                end
            end
            
            temporal_metrics[scale_key]["action_consistency"] = mean(action_consistency)
            temporal_metrics[scale_key]["consistency_trace"] = action_consistency
            
            # 2. Belief stability at this scale
            belief_windows = [beliefs_trace[max(1, i-scale+1):i, :] for i in scale:n_steps]
            belief_stability = zeros(length(belief_windows))
            
            for (idx, window) in enumerate(belief_windows)
                if size(window, 1) > 1
                    # Stability as 1 - variance of belief changes
                    belief_changes = [norm(window[t, :] - window[t-1, :]) for t in 2:size(window, 1)]
                    belief_stability[idx] = 1.0 / (1.0 + var(belief_changes))
                else
                    belief_stability[idx] = 1.0
                end
            end
            
            temporal_metrics[scale_key]["belief_stability"] = mean(belief_stability)
            temporal_metrics[scale_key]["stability_trace"] = belief_stability
            
            # 3. Temporal coherence: how well current actions predict future beliefs
            if scale < n_steps - 5
                coherence_scores = zeros(n_steps - scale - 5)
                for t in 1:length(coherence_scores)
                    current_action = actions[t + scale]
                    future_belief = beliefs_trace[t + scale + 5, :]
                    current_belief = beliefs_trace[t + scale, :]
                    
                    # Measure how much action helps predict future belief
                    belief_change = future_belief - current_belief
                    coherence_scores[t] = -norm(belief_change)  # Negative because less change = more coherent
                end
                temporal_metrics[scale_key]["temporal_coherence"] = mean(coherence_scores)
            end
        end
    end
    
    # Cross-scale analysis
    if length(time_scales) > 1
        # Consistency across scales
        consistency_values = [temporal_metrics["scale_$(scale)"]["action_consistency"] 
                            for scale in time_scales if haskey(temporal_metrics, "scale_$(scale)")]
        if length(consistency_values) > 1
            temporal_metrics["cross_scale_consistency"] = var(consistency_values)
        end
        
        # Stability across scales  
        stability_values = [temporal_metrics["scale_$(scale)"]["belief_stability"] 
                          for scale in time_scales if haskey(temporal_metrics, "scale_$(scale)")]
        if length(stability_values) > 1
            temporal_metrics["cross_scale_stability"] = var(stability_values)
        end
    end
    
    return temporal_metrics
end

# ====================================
# META-LEARNING ANALYSIS
# ====================================

"""Analyze meta-learning: learning about learning processes."""
function analyze_meta_learning(parameter_traces::Dict{String, Matrix{Float64}},
                             performance_trace::Vector{Float64})
    meta_learning_metrics = Dict{String, Any}()
    
    # 1. Learning rate adaptation
    if haskey(parameter_traces, "learning_rates")
        lr_trace = parameter_traces["learning_rates"]
        n_episodes, n_params = size(lr_trace)
        
        # Analyze how learning rates change over time
        lr_changes = zeros(n_episodes-1, n_params)
        for t in 2:n_episodes
            lr_changes[t-1, :] = lr_trace[t, :] - lr_trace[t-1, :]
        end
        
        meta_learning_metrics["learning_rate_adaptation"] = Dict(
            "mean_change" => mean(abs.(lr_changes), dims=1)[1, :],
            "adaptation_variance" => var(lr_changes, dims=1)[1, :],
            "adaptation_trend" => [cor(1:size(lr_changes, 1), lr_changes[:, i]) for i in 1:n_params]
        )
        
        # Correlation between learning rate changes and performance changes
        if length(performance_trace) >= n_episodes
            performance_changes = diff(performance_trace[1:n_episodes])
            lr_performance_corr = zeros(n_params)
            
            for i in 1:n_params
                if std(lr_changes[:, i]) > 1e-6 && std(performance_changes) > 1e-6
                    lr_performance_corr[i] = cor(lr_changes[:, i], performance_changes)
                end
            end
            
            meta_learning_metrics["learning_rate_performance_correlation"] = lr_performance_corr
        end
    end
    
    # 2. Meta-optimization: optimization of optimization
    n_performance = length(performance_trace)
    if n_performance > 20
        # Analyze learning phases
        window_size = min(10, n_performance ÷ 4)
        n_windows = n_performance ÷ window_size
        
        learning_phases = []
        for w in 1:n_windows
            start_idx = (w-1) * window_size + 1
            end_idx = min(w * window_size, n_performance)
            window_performance = performance_trace[start_idx:end_idx]
            
            # Characterize this learning phase
            phase_trend = cor(1:length(window_performance), window_performance)
            phase_variance = var(window_performance)
            phase_mean = mean(window_performance)
            
            push!(learning_phases, Dict(
                "trend" => phase_trend,
                "variance" => phase_variance,
                "mean_performance" => phase_mean,
                "phase_type" => phase_trend > 0.3 ? "learning" : 
                               phase_trend < -0.3 ? "declining" : "stable"
            ))
        end
        
        meta_learning_metrics["learning_phases"] = learning_phases
        
        # Overall meta-learning metrics
        phase_trends = [phase["trend"] for phase in learning_phases]
        meta_learning_metrics["meta_learning_consistency"] = 1.0 / (1.0 + var(phase_trends))
        meta_learning_metrics["overall_learning_direction"] = mean(phase_trends)
        
        # Learning acceleration
        if length(phase_trends) > 2
            learning_acceleration = diff(diff(phase_trends))
            meta_learning_metrics["learning_acceleration"] = mean(learning_acceleration)
        end
    end
    
    # 3. Strategy discovery and adaptation
    if haskey(parameter_traces, "strategy_indicators")
        strategy_trace = parameter_traces["strategy_indicators"]
        n_episodes = size(strategy_trace, 1)
        
        # Strategy switching analysis
        strategy_changes = zeros(n_episodes-1)
        for t in 2:n_episodes
            strategy_changes[t-1] = norm(strategy_trace[t, :] - strategy_trace[t-1, :])
        end
        
        meta_learning_metrics["strategy_adaptation"] = Dict(
            "switching_frequency" => sum(strategy_changes .> 0.1) / length(strategy_changes),
            "mean_change_magnitude" => mean(strategy_changes),
            "adaptation_timing" => findall(strategy_changes .> 0.1)
        )
        
        # Effectiveness of strategy changes
        if length(performance_trace) >= n_episodes
            switch_effects = zeros(sum(strategy_changes .> 0.1))
            switch_indices = findall(strategy_changes .> 0.1)
            
            for (idx, switch_t) in enumerate(switch_indices)
                if switch_t + 5 <= length(performance_trace)
                    pre_switch = mean(performance_trace[max(1, switch_t-2):switch_t])
                    post_switch = mean(performance_trace[switch_t+1:switch_t+5])
                    switch_effects[idx] = post_switch - pre_switch
                end
            end
            
            if length(switch_effects) > 0
                meta_learning_metrics["strategy_switch_effectiveness"] = mean(switch_effects)
            end
        end
    end
    
    return meta_learning_metrics
end

# ====================================
# THEORY OF MIND MODELING
# ====================================

"""Model theory of mind capabilities for multi-agent scenarios."""
function analyze_theory_of_mind(agent_beliefs::Matrix{Float64},
                              other_agent_actions::Vector{Int},
                              predicted_other_actions::Vector{Int})
    tom_metrics = Dict{String, Any}()
    n_steps = length(other_agent_actions)
    
    # 1. Prediction accuracy of other agent's actions
    prediction_accuracy = sum(predicted_other_actions .== other_agent_actions) / n_steps
    tom_metrics["action_prediction_accuracy"] = prediction_accuracy
    
    # 2. Belief attribution accuracy
    # Assume we have some ground truth about other agent's beliefs (in simulation)
    if size(agent_beliefs, 1) >= n_steps
        # Analyze how well we track other agent's belief changes
        belief_prediction_errors = zeros(n_steps-1)
        for t in 2:n_steps
            # Simple heuristic: predict other agent's beliefs based on their actions
            predicted_belief_change = 0.1 * (other_agent_actions[t] - other_agent_actions[t-1])
            actual_belief_change = norm(agent_beliefs[t, :] - agent_beliefs[t-1, :])
            belief_prediction_errors[t-1] = abs(predicted_belief_change - actual_belief_change)
        end
        
        tom_metrics["belief_attribution_error"] = mean(belief_prediction_errors)
    end
    
    # 3. Mentalizing depth analysis
    # How many levels of "I think that they think that..." reasoning
    mentalizing_indicators = zeros(n_steps-2)
    for t in 3:n_steps
        # Look for patterns that suggest deeper mentalizing
        action_pattern = [other_agent_actions[t-2], other_agent_actions[t-1], other_agent_actions[t]]
        prediction_pattern = [predicted_other_actions[t-2], predicted_other_actions[t-1], predicted_other_actions[t]]
        
        # Complex pattern matching suggests deeper theory of mind
        if length(unique(action_pattern)) > 1 && length(unique(prediction_pattern)) > 1
            pattern_correlation = cor(action_pattern, prediction_pattern)
            mentalizing_indicators[t-2] = abs(pattern_correlation)
        end
    end
    
    tom_metrics["mentalizing_depth"] = mean(mentalizing_indicators)
    
    # 4. Social learning indicators
    # How much does observing other agent's actions influence own beliefs
    if size(agent_beliefs, 1) >= n_steps
        social_influence = zeros(n_steps-1)
        for t in 2:n_steps
            belief_change = norm(agent_beliefs[t, :] - agent_beliefs[t-1, :])
            other_action_surprise = abs(other_agent_actions[t] - mean(other_agent_actions[1:t-1]))
            
            # If belief change correlates with surprising actions from others, it suggests social learning
            social_influence[t-1] = belief_change * other_action_surprise
        end
        
        tom_metrics["social_learning_strength"] = mean(social_influence)
    end
    
    return tom_metrics
end

# ====================================
# COMPREHENSIVE META-COGNITIVE ANALYSIS
# ====================================

"""Run comprehensive meta-cognitive analysis on ActiveInference.jl data."""
function comprehensive_metacognitive_analysis(output_dir::String)
    println("🧠 Running Comprehensive Meta-Cognitive Analysis")
    
    # Create output directory
    metacog_dir = joinpath(output_dir, "metacognitive_analysis")
    mkpath(metacog_dir)
    
    analysis_results = Dict{String, Any}()
    
    # Load data with error handling
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
                
                # Generate confidence trace (placeholder - in real implementation this would come from model)
                confidence_trace = 1.0 .- abs.(beliefs .- 0.5) .* 2  # Higher confidence when more certain
                
                # Create beliefs trace matrix
                beliefs_trace = hcat(beliefs, 1.0 .- beliefs)  # Binary case
                belief_changes = zeros(size(beliefs_trace))
                for t in 2:size(beliefs_trace, 1)
                    belief_changes[t, :] = beliefs_trace[t, :] - beliefs_trace[t-1, :]
                end
                
                # 1. Meta-cognitive awareness analysis
                awareness_results = calculate_metacognitive_awareness(beliefs_trace, observations, confidence_trace)
                analysis_results["metacognitive_awareness"] = awareness_results
                
                # 2. Higher-order beliefs analysis
                higher_order_results = analyze_higher_order_beliefs(beliefs_trace, belief_changes)
                analysis_results["higher_order_beliefs"] = higher_order_results
                
                # 3. Multi-scale temporal analysis
                temporal_results = multi_scale_temporal_analysis(actions, beliefs_trace)
                analysis_results["temporal_analysis"] = temporal_results
                
                println("✅ Basic meta-cognitive analysis completed")
            else
                @warn "Insufficient data columns in basic simulation file"
            end
        else
            @warn "Basic simulation data not found: $basic_data_path"
        end
        
        # 4. Meta-learning analysis (if parameter traces available)
        parameter_traces = Dict{String, Matrix{Float64}}()
        
        # Look for parameter learning data
        param_files = ["learning_trace.csv", "parameter_evolution.csv", "adaptation_trace.csv"]
        for file in param_files
            file_path = joinpath(output_dir, "simulation_results", file)
            if isfile(file_path)
                try
                    param_data = load_statistical_data(file_path)
                    parameter_traces[file] = param_data
                catch e
                    @warn "Failed to load parameter file $file: $e"
                end
            end
        end
        
        if !isempty(parameter_traces) && haskey(analysis_results, "metacognitive_awareness")
            # Create dummy performance trace based on prediction accuracy
            performance_trace = ones(length(observations))
            
            meta_learning_results = analyze_meta_learning(parameter_traces, performance_trace)
            analysis_results["meta_learning"] = meta_learning_results
            
            println("✅ Meta-learning analysis completed")
        end
        
        # 5. Theory of mind analysis (if multi-agent data available)
        multiagent_path = joinpath(output_dir, "simulation_results", "multiagent_simulation.csv")
        if isfile(multiagent_path)
            try
                multiagent_data = load_statistical_data(multiagent_path)
                if size(multiagent_data, 2) >= 6
                    agent_beliefs = multiagent_data[:, 3:4]
                    other_actions = Int.(multiagent_data[:, 5])
                    predicted_actions = Int.(multiagent_data[:, 6])
                    
                    tom_results = analyze_theory_of_mind(agent_beliefs, other_actions, predicted_actions)
                    analysis_results["theory_of_mind"] = tom_results
                    
                    println("✅ Theory of mind analysis completed")
                end
            catch e
                @warn "Failed to analyze multi-agent data: $e"
            end
        end
        
    catch e
        @error "Meta-cognitive analysis failed: $e"
        analysis_results["error"] = string(e)
    end
    
    # Save comprehensive results
    if !isempty(analysis_results)
        # Save as JSON
        json_path = joinpath(metacog_dir, "metacognitive_analysis_results.json")
        open(json_path, "w") do f
            JSON.print(f, analysis_results, 2)
        end
        
        # Create summary report
        create_metacognitive_report(analysis_results, metacog_dir)
        
        println("📊 Meta-cognitive analysis results saved to: $metacog_dir")
    end
    
    return analysis_results
end

"""Create a comprehensive meta-cognitive analysis report."""
function create_metacognitive_report(results::Dict{String, Any}, output_dir::String)
    report_path = joinpath(output_dir, "metacognitive_analysis_report.md")
    
    open(report_path, "w") do f
        println(f, "# Meta-Cognitive Analysis Report")
        println(f, "")
        println(f, "Generated: $(now())")
        println(f, "")
        
        if haskey(results, "metacognitive_awareness")
            awareness = results["metacognitive_awareness"]
            println(f, "## Meta-Cognitive Awareness")
            println(f, "")
            println(f, "- **Metacognitive Sensitivity**: $(get(awareness, "metacognitive_sensitivity", "N/A"))")
            println(f, "- **Calibration Error**: $(get(awareness, "calibration_error", "N/A"))")
            println(f, "- **Meta-Uncertainty**: $(get(awareness, "meta_uncertainty", "N/A"))")
            println(f, "- **Confidence Resolution**: $(get(awareness, "confidence_resolution", "N/A"))")
            println(f, "")
        end
        
        if haskey(results, "higher_order_beliefs")
            higher_order = results["higher_order_beliefs"]
            println(f, "## Higher-Order Beliefs")
            println(f, "")
            println(f, "- **Mean Volatility**: $(get(higher_order, "mean_volatility", "N/A"))")
            println(f, "- **Volatility Trend**: $(get(higher_order, "volatility_trend", "N/A"))")
            println(f, "- **Mean Coherence**: $(get(higher_order, "mean_coherence", "N/A"))")
            if haskey(higher_order, "meta_prediction_error")
                println(f, "- **Meta-Prediction Error**: $(higher_order["meta_prediction_error"])")
            end
            println(f, "")
        end
        
        if haskey(results, "temporal_analysis")
            temporal = results["temporal_analysis"]
            println(f, "## Multi-Scale Temporal Analysis")
            println(f, "")
            for (scale_key, scale_data) in temporal
                if isa(scale_data, Dict) && haskey(scale_data, "action_consistency")
                    println(f, "### $scale_key")
                    println(f, "- **Action Consistency**: $(scale_data["action_consistency"])")
                    println(f, "- **Belief Stability**: $(scale_data["belief_stability"])")
                    if haskey(scale_data, "temporal_coherence")
                        println(f, "- **Temporal Coherence**: $(scale_data["temporal_coherence"])")
                    end
                    println(f, "")
                end
            end
        end
        
        if haskey(results, "meta_learning")
            meta_learning = results["meta_learning"]
            println(f, "## Meta-Learning Analysis")
            println(f, "")
            if haskey(meta_learning, "learning_phases")
                phases = meta_learning["learning_phases"]
                println(f, "- **Number of Learning Phases**: $(length(phases))")
                learning_phase_types = [phase["phase_type"] for phase in phases]
                println(f, "- **Phase Types**: $(join(unique(learning_phase_types), ", "))")
            end
            if haskey(meta_learning, "meta_learning_consistency")
                println(f, "- **Meta-Learning Consistency**: $(meta_learning["meta_learning_consistency"])")
            end
            println(f, "")
        end
        
        if haskey(results, "theory_of_mind")
            tom = results["theory_of_mind"]
            println(f, "## Theory of Mind")
            println(f, "")
            println(f, "- **Action Prediction Accuracy**: $(get(tom, "action_prediction_accuracy", "N/A"))")
            println(f, "- **Belief Attribution Error**: $(get(tom, "belief_attribution_error", "N/A"))")
            println(f, "- **Mentalizing Depth**: $(get(tom, "mentalizing_depth", "N/A"))")
            println(f, "- **Social Learning Strength**: $(get(tom, "social_learning_strength", "N/A"))")
            println(f, "")
        end
        
        println(f, "## Summary")
        println(f, "")
        println(f, "This meta-cognitive analysis provides insights into higher-order reasoning processes,")
        println(f, "including awareness of one's own cognitive processes, confidence calibration,")
        println(f, "multi-scale temporal reasoning, and theory of mind capabilities.")
    end
    
    println("📋 Meta-cognitive report saved: $report_path")
end

# Utility function from advanced_pomdp_analysis.jl (included for completeness)
function shannon_entropy(prob_dist::Vector{Float64})
    non_zero_probs = filter(p -> p > 1e-12, prob_dist)
    -sum(p * log2(p) for p in non_zero_probs)
end

# Utility function to load statistical data (placeholder implementation)
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

# Export the main function
export comprehensive_metacognitive_analysis, create_metacognitive_report

println("🧠 Meta-Cognitive Analysis Module Loaded Successfully") 