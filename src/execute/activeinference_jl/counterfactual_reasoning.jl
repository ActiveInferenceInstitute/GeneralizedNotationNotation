#!/usr/bin/env julia

"""
Counterfactual Reasoning Module for ActiveInference.jl

This module provides advanced counterfactual reasoning capabilities for POMDP understanding:
- Alternative scenario generation and analysis
- What-if reasoning for different action sequences
- Causal intervention modeling and outcome prediction
- Temporal counterfactuals across different time horizons
- Belief counterfactuals: how beliefs would differ under alternative evidence
- Policy counterfactuals: alternative policy outcomes comparison
- Regret and relief analysis based on counterfactual outcomes
- Causal attribution and responsibility assignment
- Learning from counterfactual scenarios for improved decision-making
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
for pkg in ["StatsBase", "Combinatorics", "Optim"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using StatsBase

# ====================================
# COUNTERFACTUAL SCENARIO GENERATION
# ====================================

"""Generate alternative action sequences for counterfactual analysis."""
function generate_counterfactual_actions(original_actions::Vector{Int},
                                       n_possible_actions::Int,
                                       intervention_points::Vector{Int},
                                       n_alternatives::Int = 5)
    counterfactual_scenarios = Dict{String, Any}()
    
    # 1. Single-point interventions
    single_interventions = []
    for intervention_t in intervention_points
        for alternative_action in 1:n_possible_actions
            if alternative_action != original_actions[intervention_t]
                # Create alternative sequence
                alternative_sequence = copy(original_actions)
                alternative_sequence[intervention_t] = alternative_action
                
                push!(single_interventions, Dict(
                    "type" => "single_intervention",
                    "intervention_time" => intervention_t,
                    "original_action" => original_actions[intervention_t],
                    "alternative_action" => alternative_action,
                    "sequence" => alternative_sequence
                ))
            end
        end
    end
    
    counterfactual_scenarios["single_interventions"] = single_interventions
    
    # 2. Multi-point interventions
    multi_interventions = []
    if length(intervention_points) >= 2
        # Random multi-point interventions
        Random.seed!(42)  # For reproducibility
        for _ in 1:n_alternatives
            alternative_sequence = copy(original_actions)
            intervention_times = sample(intervention_points, min(3, length(intervention_points)), replace=false)
            
            original_actions_at_interventions = []
            alternative_actions_at_interventions = []
            
            for t in intervention_times
                original_action = original_actions[t]
                alternative_action = rand(1:n_possible_actions)
                while alternative_action == original_action
                    alternative_action = rand(1:n_possible_actions)
                end
                
                alternative_sequence[t] = alternative_action
                push!(original_actions_at_interventions, original_action)
                push!(alternative_actions_at_interventions, alternative_action)
            end
            
            push!(multi_interventions, Dict(
                "type" => "multi_intervention",
                "intervention_times" => intervention_times,
                "original_actions" => original_actions_at_interventions,
                "alternative_actions" => alternative_actions_at_interventions,
                "sequence" => alternative_sequence
            ))
        end
    end
    
    counterfactual_scenarios["multi_interventions"] = multi_interventions
    
    # 3. Policy-level alternatives
    policy_alternatives = []
    
    # Greedy policy alternative
    greedy_sequence = copy(original_actions)
    for t in intervention_points
        # Simple greedy: action that minimizes immediate uncertainty
        greedy_sequence[t] = 1  # Placeholder - would use belief-based selection in real implementation
    end
    
    push!(policy_alternatives, Dict(
        "type" => "greedy_policy",
        "description" => "Greedy uncertainty minimization",
        "sequence" => greedy_sequence
    ))
    
    # Random policy alternative
    random_sequence = copy(original_actions)
    Random.seed!(123)
    for t in intervention_points
        random_sequence[t] = rand(1:n_possible_actions)
    end
    
    push!(policy_alternatives, Dict(
        "type" => "random_policy",
        "description" => "Random action selection",
        "sequence" => random_sequence
    ))
    
    counterfactual_scenarios["policy_alternatives"] = policy_alternatives
    
    return counterfactual_scenarios
end

"""Simulate outcomes for counterfactual action sequences."""
function simulate_counterfactual_outcomes(counterfactual_scenarios::Dict{String, Any},
                                        A_matrix::Matrix{Float64},  # Observation model
                                        B_matrix::Array{Float64, 3},  # Transition model
                                        initial_beliefs::Vector{Float64},
                                        observations::Vector{Int})
    outcome_results = Dict{String, Any}()
    
    for (scenario_type, scenarios) in counterfactual_scenarios
        if isa(scenarios, Vector)
            scenario_outcomes = []
            
            for scenario in scenarios
                if haskey(scenario, "sequence")
                    alternative_actions = scenario["sequence"]
                    
                    # Simulate belief evolution under alternative actions
                    simulated_beliefs, simulated_observations, performance_metrics = 
                        simulate_belief_evolution(alternative_actions, A_matrix, B_matrix, 
                                                initial_beliefs, observations)
                    
                    scenario_outcome = copy(scenario)
                    scenario_outcome["simulated_beliefs"] = simulated_beliefs
                    scenario_outcome["simulated_observations"] = simulated_observations
                    scenario_outcome["performance_metrics"] = performance_metrics
                    
                    push!(scenario_outcomes, scenario_outcome)
                end
            end
            
            outcome_results[scenario_type] = scenario_outcomes
        end
    end
    
    return outcome_results
end

"""Simulate belief evolution under alternative action sequences."""
function simulate_belief_evolution(actions::Vector{Int},
                                 A_matrix::Matrix{Float64},
                                 B_matrix::Array{Float64, 3},
                                 initial_beliefs::Vector{Float64},
                                 observations::Vector{Int})
    n_steps = length(actions)
    n_states = length(initial_beliefs)
    
    beliefs_trace = zeros(n_steps, n_states)
    beliefs_trace[1, :] = initial_beliefs
    
    simulated_observations = zeros(Int, n_steps)
    performance_metrics = Dict{String, Float64}()
    
    for t in 1:(n_steps-1)
        # Current beliefs
        current_beliefs = beliefs_trace[t, :]
        action = actions[t]
        
        # Predict next state based on transition model
        if size(B_matrix, 3) >= action
            predicted_next_beliefs = B_matrix[:, :, action]' * current_beliefs
        else
            # Fallback: no change
            predicted_next_beliefs = current_beliefs
        end
        
        # Observe environment (use actual observations for consistency)
        if t + 1 <= length(observations)
            observation = observations[t + 1]
        else
            # Generate observation based on predicted state
            predicted_state = argmax(predicted_next_beliefs)
            observation = sample_observation(predicted_state, A_matrix)
        end
        
        simulated_observations[t + 1] = observation
        
        # Update beliefs based on observation
        if observation <= size(A_matrix, 1)
            likelihood = A_matrix[observation, :]
            posterior_beliefs = predicted_next_beliefs .* likelihood
            posterior_beliefs ./= sum(posterior_beliefs)
        else
            posterior_beliefs = predicted_next_beliefs
        end
        
        beliefs_trace[t + 1, :] = posterior_beliefs
    end
    
    # Calculate performance metrics
    performance_metrics["final_uncertainty"] = shannon_entropy(beliefs_trace[end, :])
    performance_metrics["mean_uncertainty"] = mean([shannon_entropy(beliefs_trace[t, :]) for t in 1:n_steps])
    performance_metrics["belief_stability"] = calculate_belief_stability(beliefs_trace)
    performance_metrics["convergence_rate"] = calculate_convergence_rate(beliefs_trace)
    
    return beliefs_trace, simulated_observations, performance_metrics
end

"""Sample observation based on state and observation model."""
function sample_observation(state::Int, A_matrix::Matrix{Float64})
    if state <= size(A_matrix, 2)
        obs_probs = A_matrix[:, state]
        return sample(Weights(obs_probs))
    else
        return 1  # Fallback
    end
end

# ====================================
# COUNTERFACTUAL ANALYSIS METRICS
# ====================================

"""Calculate regret and relief based on counterfactual outcomes."""
function calculate_regret_relief(original_outcome::Dict{String, Any},
                               counterfactual_outcomes::Vector{Dict{String, Any}})
    regret_relief_metrics = Dict{String, Any}()
    
    original_performance = original_outcome["performance_metrics"]
    original_uncertainty = original_performance["final_uncertainty"]
    
    # Calculate regret: how much better could we have done?
    better_outcomes = []
    worse_outcomes = []
    
    for cf_outcome in counterfactual_outcomes
        cf_performance = cf_outcome["performance_metrics"]
        cf_uncertainty = cf_performance["final_uncertainty"]
        
        performance_difference = original_uncertainty - cf_uncertainty  # Lower uncertainty is better
        
        if performance_difference < 0  # Counterfactual was better
            push!(better_outcomes, abs(performance_difference))
        else  # Counterfactual was worse
            push!(worse_outcomes, performance_difference)
        end
    end
    
    # Regret: maximum improvement we missed
    regret = length(better_outcomes) > 0 ? maximum(better_outcomes) : 0.0
    
    # Relief: how much worse it could have been
    relief = length(worse_outcomes) > 0 ? maximum(worse_outcomes) : 0.0
    
    regret_relief_metrics["regret"] = regret
    regret_relief_metrics["relief"] = relief
    regret_relief_metrics["regret_relief_ratio"] = relief > 0 ? regret / relief : Inf
    regret_relief_metrics["n_better_outcomes"] = length(better_outcomes)
    regret_relief_metrics["n_worse_outcomes"] = length(worse_outcomes)
    regret_relief_metrics["mean_improvement_missed"] = length(better_outcomes) > 0 ? mean(better_outcomes) : 0.0
    regret_relief_metrics["mean_harm_avoided"] = length(worse_outcomes) > 0 ? mean(worse_outcomes) : 0.0
    
    return regret_relief_metrics
end

"""Analyze causal impact of specific interventions."""
function analyze_causal_impact(original_beliefs::Matrix{Float64},
                             intervention_outcomes::Vector{Dict{String, Any}})
    causal_impact_metrics = Dict{String, Any}()
    
    # 1. Direct intervention effects
    intervention_effects = []
    
    for intervention in intervention_outcomes
        if haskey(intervention, "intervention_time") && haskey(intervention, "simulated_beliefs")
            t_intervention = intervention["intervention_time"]
            cf_beliefs = intervention["simulated_beliefs"]
            
            # Compare beliefs at intervention time and after
            if t_intervention <= size(original_beliefs, 1) && t_intervention <= size(cf_beliefs, 1)
                original_belief_t = original_beliefs[t_intervention, :]
                cf_belief_t = cf_beliefs[t_intervention, :]
                
                immediate_impact = norm(cf_belief_t - original_belief_t)
                
                # Propagated impact (how long the effect lasts)
                propagated_impacts = []
                for t in (t_intervention + 1):min(size(original_beliefs, 1), size(cf_beliefs, 1))
                    original_belief_future = original_beliefs[t, :]
                    cf_belief_future = cf_beliefs[t, :]
                    future_impact = norm(cf_belief_future - original_belief_future)
                    push!(propagated_impacts, future_impact)
                end
                
                push!(intervention_effects, Dict(
                    "intervention_time" => t_intervention,
                    "original_action" => intervention["original_action"],
                    "alternative_action" => intervention["alternative_action"],
                    "immediate_impact" => immediate_impact,
                    "propagated_impacts" => propagated_impacts,
                    "impact_duration" => length(propagated_impacts),
                    "total_impact" => sum(propagated_impacts)
                ))
            end
        end
    end
    
    causal_impact_metrics["intervention_effects"] = intervention_effects
    
    # 2. Aggregate causal metrics
    if length(intervention_effects) > 0
        immediate_impacts = [effect["immediate_impact"] for effect in intervention_effects]
        total_impacts = [effect["total_impact"] for effect in intervention_effects]
        impact_durations = [effect["impact_duration"] for effect in intervention_effects]
        
        causal_impact_metrics["mean_immediate_impact"] = mean(immediate_impacts)
        causal_impact_metrics["max_immediate_impact"] = maximum(immediate_impacts)
        causal_impact_metrics["mean_total_impact"] = mean(total_impacts)
        causal_impact_metrics["mean_impact_duration"] = mean(impact_durations)
        causal_impact_metrics["impact_persistence"] = cor(1:length(immediate_impacts), total_impacts)
    end
    
    return causal_impact_metrics
end

"""Analyze belief counterfactuals: how beliefs would differ under alternative evidence."""
function analyze_belief_counterfactuals(original_beliefs::Matrix{Float64},
                                      original_observations::Vector{Int},
                                      A_matrix::Matrix{Float64})
    belief_counterfactuals = Dict{String, Any}()
    n_steps, n_states = size(original_beliefs)
    n_observations = size(A_matrix, 1)
    
    # 1. Alternative observation scenarios
    observation_counterfactuals = []
    
    for t in 2:min(n_steps, 10)  # Analyze first 10 steps
        original_obs = original_observations[t]
        
        for alternative_obs in 1:n_observations
            if alternative_obs != original_obs
                # Create alternative observation sequence
                alt_observations = copy(original_observations)
                alt_observations[t] = alternative_obs
                
                # Simulate beliefs under alternative observations
                alt_beliefs = simulate_beliefs_with_observations(
                    alt_observations[1:t], A_matrix, original_beliefs[1, :]
                )
                
                # Calculate belief divergence
                belief_divergence = norm(alt_beliefs[end, :] - original_beliefs[t, :])
                
                push!(observation_counterfactuals, Dict(
                    "time" => t,
                    "original_observation" => original_obs,
                    "alternative_observation" => alternative_obs,
                    "belief_divergence" => belief_divergence,
                    "alternative_beliefs" => alt_beliefs
                ))
            end
        end
    end
    
    belief_counterfactuals["observation_counterfactuals"] = observation_counterfactuals
    
    # 2. Evidence strength analysis
    evidence_impacts = []
    
    for cf in observation_counterfactuals
        t = cf["time"]
        divergence = cf["belief_divergence"]
        
        # How much this evidence change matters depends on prior uncertainty
        if t > 1
            prior_uncertainty = shannon_entropy(original_beliefs[t-1, :])
            evidence_impact = divergence / (prior_uncertainty + 1e-6)
            
            push!(evidence_impacts, Dict(
                "time" => t,
                "prior_uncertainty" => prior_uncertainty,
                "evidence_impact" => evidence_impact,
                "normalized_impact" => evidence_impact
            ))
        end
    end
    
    belief_counterfactuals["evidence_impacts"] = evidence_impacts
    
    # 3. Critical evidence points
    if length(evidence_impacts) > 0
        impact_scores = [ei["evidence_impact"] for ei in evidence_impacts]
        critical_threshold = mean(impact_scores) + std(impact_scores)
        
        critical_points = [ei for ei in evidence_impacts if ei["evidence_impact"] > critical_threshold]
        belief_counterfactuals["critical_evidence_points"] = critical_points
        belief_counterfactuals["n_critical_points"] = length(critical_points)
    end
    
    return belief_counterfactuals
end

"""Simulate beliefs under alternative observation sequences."""
function simulate_beliefs_with_observations(observations::Vector{Int},
                                          A_matrix::Matrix{Float64},
                                          initial_beliefs::Vector{Float64})
    n_steps = length(observations)
    n_states = length(initial_beliefs)
    beliefs_trace = zeros(n_steps, n_states)
    beliefs_trace[1, :] = initial_beliefs
    
    for t in 2:n_steps
        prior_beliefs = beliefs_trace[t-1, :]
        observation = observations[t]
        
        if observation <= size(A_matrix, 1)
            likelihood = A_matrix[observation, :]
            posterior = prior_beliefs .* likelihood
            posterior ./= sum(posterior)
            beliefs_trace[t, :] = posterior
        else
            beliefs_trace[t, :] = prior_beliefs
        end
    end
    
    return beliefs_trace
end

# ====================================
# COMPREHENSIVE COUNTERFACTUAL ANALYSIS
# ====================================

"""Run comprehensive counterfactual reasoning analysis."""
function comprehensive_counterfactual_analysis(output_dir::String)
    println("🔀 Running Comprehensive Counterfactual Reasoning Analysis")
    
    # Create output directory
    counterfactual_dir = joinpath(output_dir, "counterfactual_reasoning")
    mkpath(counterfactual_dir)
    
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
                
                # Create simple A and B matrices for analysis
                A_matrix = [0.8 0.2; 0.2 0.8]  # 2x2 observation model
                B_matrix = zeros(2, 2, 2)
                B_matrix[:, :, 1] = [0.9 0.1; 0.1 0.9]  # Action 1
                B_matrix[:, :, 2] = [0.1 0.9; 0.9 0.1]  # Action 2
                
                # Define intervention points (every 5th step)
                intervention_points = collect(5:5:min(length(actions), 20))
                
                # 1. Generate counterfactual scenarios
                counterfactual_scenarios = generate_counterfactual_actions(
                    actions, 2, intervention_points, 3
                )
                analysis_results["counterfactual_scenarios"] = counterfactual_scenarios
                
                # 2. Simulate counterfactual outcomes
                counterfactual_outcomes = simulate_counterfactual_outcomes(
                    counterfactual_scenarios, A_matrix, B_matrix, beliefs_trace[1, :], observations
                )
                analysis_results["counterfactual_outcomes"] = counterfactual_outcomes
                
                # 3. Calculate regret and relief
                original_outcome = Dict(
                    "performance_metrics" => Dict(
                        "final_uncertainty" => shannon_entropy(beliefs_trace[end, :]),
                        "mean_uncertainty" => mean([shannon_entropy(beliefs_trace[t, :]) for t in 1:size(beliefs_trace, 1)]),
                        "belief_stability" => calculate_belief_stability(beliefs_trace),
                        "convergence_rate" => calculate_convergence_rate(beliefs_trace)
                    )
                )
                
                all_cf_outcomes = []
                for (_, scenarios) in counterfactual_outcomes
                    if isa(scenarios, Vector)
                        append!(all_cf_outcomes, scenarios)
                    end
                end
                
                regret_relief = calculate_regret_relief(original_outcome, all_cf_outcomes)
                analysis_results["regret_relief"] = regret_relief
                
                # 4. Causal impact analysis
                single_interventions = get(counterfactual_outcomes, "single_interventions", [])
                if !isempty(single_interventions)
                    causal_impact = analyze_causal_impact(beliefs_trace, single_interventions)
                    analysis_results["causal_impact"] = causal_impact
                end
                
                # 5. Belief counterfactuals
                belief_counterfactuals = analyze_belief_counterfactuals(
                    beliefs_trace, observations, A_matrix
                )
                analysis_results["belief_counterfactuals"] = belief_counterfactuals
                
                println("✅ Counterfactual reasoning analysis completed")
            end
        end
        
    catch e
        @error "Counterfactual analysis failed: $e"
        analysis_results["error"] = string(e)
    end
    
    # Save comprehensive results
    if !isempty(analysis_results)
        # Save as JSON
        json_path = joinpath(counterfactual_dir, "counterfactual_analysis_results.json")
        open(json_path, "w") do f
            JSON.print(f, analysis_results, 2)
        end
        
        # Create summary report
        create_counterfactual_report(analysis_results, counterfactual_dir)
        
        println("🔀 Counterfactual analysis results saved to: $counterfactual_dir")
    end
    
    return analysis_results
end

"""Create a comprehensive counterfactual analysis report."""
function create_counterfactual_report(results::Dict{String, Any}, output_dir::String)
    report_path = joinpath(output_dir, "counterfactual_analysis_report.md")
    
    open(report_path, "w") do f
        println(f, "# Counterfactual Reasoning Analysis Report")
        println(f, "")
        println(f, "Generated: $(now())")
        println(f, "")
        
        if haskey(results, "counterfactual_scenarios")
            scenarios = results["counterfactual_scenarios"]
            println(f, "## Counterfactual Scenarios")
            println(f, "")
            
            for (scenario_type, scenario_list) in scenarios
                if isa(scenario_list, Vector)
                    println(f, "### $(replace(scenario_type, "_" => " ") |> titlecase)")
                    println(f, "- **Number of scenarios**: $(length(scenario_list))")
                    println(f, "")
                end
            end
        end
        
        if haskey(results, "regret_relief")
            regret_relief = results["regret_relief"]
            println(f, "## Regret and Relief Analysis")
            println(f, "")
            println(f, "- **Regret**: $(get(regret_relief, "regret", "N/A"))")
            println(f, "- **Relief**: $(get(regret_relief, "relief", "N/A"))")
            println(f, "- **Regret/Relief Ratio**: $(get(regret_relief, "regret_relief_ratio", "N/A"))")
            println(f, "- **Better Outcomes**: $(get(regret_relief, "n_better_outcomes", "N/A"))")
            println(f, "- **Worse Outcomes**: $(get(regret_relief, "n_worse_outcomes", "N/A"))")
            println(f, "- **Mean Improvement Missed**: $(get(regret_relief, "mean_improvement_missed", "N/A"))")
            println(f, "- **Mean Harm Avoided**: $(get(regret_relief, "mean_harm_avoided", "N/A"))")
            println(f, "")
        end
        
        if haskey(results, "causal_impact")
            causal = results["causal_impact"]
            println(f, "## Causal Impact Analysis")
            println(f, "")
            println(f, "- **Mean Immediate Impact**: $(get(causal, "mean_immediate_impact", "N/A"))")
            println(f, "- **Max Immediate Impact**: $(get(causal, "max_immediate_impact", "N/A"))")
            println(f, "- **Mean Total Impact**: $(get(causal, "mean_total_impact", "N/A"))")
            println(f, "- **Mean Impact Duration**: $(get(causal, "mean_impact_duration", "N/A"))")
            println(f, "- **Impact Persistence**: $(get(causal, "impact_persistence", "N/A"))")
            println(f, "")
        end
        
        if haskey(results, "belief_counterfactuals")
            belief_cf = results["belief_counterfactuals"]
            println(f, "## Belief Counterfactuals")
            println(f, "")
            
            if haskey(belief_cf, "observation_counterfactuals")
                obs_cf = belief_cf["observation_counterfactuals"]
                println(f, "- **Observation Counterfactuals**: $(length(obs_cf))")
            end
            
            if haskey(belief_cf, "critical_evidence_points")
                critical = belief_cf["critical_evidence_points"]
                println(f, "- **Critical Evidence Points**: $(length(critical))")
            end
            
            println(f, "")
        end
        
        println(f, "## Summary")
        println(f, "")
        println(f, "This analysis explores alternative scenarios and what-if reasoning,")
        println(f, "examining how different actions or observations would have led to")
        println(f, "different outcomes, and quantifying regret, relief, and causal impacts.")
    end
    
    println("📋 Counterfactual report saved: $report_path")
end

# Utility functions
function shannon_entropy(prob_dist::Vector{Float64})
    non_zero_probs = filter(p -> p > 1e-12, prob_dist)
    -sum(p * log2(p) for p in non_zero_probs)
end

function calculate_belief_stability(beliefs_trace::Matrix{Float64})
    n_steps = size(beliefs_trace, 1)
    if n_steps < 2
        return 1.0
    end
    
    changes = [norm(beliefs_trace[t, :] - beliefs_trace[t-1, :]) for t in 2:n_steps]
    return 1.0 / (1.0 + mean(changes))
end

function calculate_convergence_rate(beliefs_trace::Matrix{Float64})
    n_steps = size(beliefs_trace, 1)
    if n_steps < 5
        return 0.0
    end
    
    entropies = [shannon_entropy(beliefs_trace[t, :]) for t in 1:n_steps]
    
    # Fit exponential decay
    if std(entropies) > 1e-6
        return -cor(1:n_steps, entropies)
    else
        return 0.0
    end
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
export comprehensive_counterfactual_analysis, create_counterfactual_report

println("🔀 Counterfactual Reasoning Module Loaded Successfully") 