#!/usr/bin/env julia
"""
Enhanced ActiveInference.jl simulation for actinf_pomdp_agent
Generated from GNN specification: unknown.md
Features comprehensive visualizations and data export
"""

using Distributions
using LinearAlgebra
using Random
using JSON
using Plots

# Enhanced utilities and logging
function log_success(name, message)
    println("✅ $name: $message")
end

function log_step(name, step, data)
    println("📊 $name Step $step: $data")
end

# Custom softmax for numerical stability
function enhanced_softmax(x::AbstractVector)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

# Enhanced POMDP agent structure
mutable struct EnhancedActinf_pomdp_agentAgent
    A::Array{Float64,2}
    B::Array{Float64,3}
    C::Vector{Float64}
    D::Vector{Float64}
    belief::Vector{Float64}
    num_states::Int
    num_obs::Int
    num_actions::Int
    performance_metrics::Dict{String,Any}
    simulation_history::Vector{Dict{String,Any}}
end

function create_enhanced_agent()
    log_success("Agent Creation", "Creating Enhanced ActiveInference.jl agent")
    
    num_states = 3
    num_obs = 4
    num_actions = 2
    
    # Real POMDP matrices from GNN specification
    A = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.33, 0.33, 0.33]]
    
    # Normalize A matrix columns
    for s in 1:num_states
        A[:, s] = A[:, s] ./ sum(A[:, s])
    end
    
    # B tensor from GNN
    B_data = [[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]]
    B = zeros(num_states, num_states, num_actions)
    
    for a in 1:num_actions
        for s in 1:num_states
            B[:, s, a] = B_data[a][s]
        end
    end
    
    # Normalize B tensor
    for a in 1:num_actions
        for s in 1:num_states
            B[:, s, a] = B[:, s, a] ./ sum(B[:, s, a])
        end
    end
    
    C = [0.1, 0.1, 1.0, 0.0]
    D = [0.333, 0.333, 0.333]
    D = D ./ sum(D)
    
    agent = EnhancedActinf_pomdp_agentAgent(
        A, B, C, D, copy(D), 
        num_states, num_obs, num_actions,
        Dict{String,Any}(), Vector{Dict{String,Any}}()
    )
    
    log_success("Agent Creation", "Enhanced agent created with comprehensive tracking")
    println("  A column sums: ", [sum(A[:, s]) for s in 1:num_states])
    
    return agent
end

function run_enhanced_simulation(agent::EnhancedActinf_pomdp_agentAgent, num_steps::Int = 15)
    log_success("Simulation", "Running Enhanced ActiveInference.jl simulation ($num_steps steps)")
    
    # Enhanced data collection
    belief_history = Vector{Float64}[]
    action_history = Int[]
    observation_history = Int[]
    reward_history = Float64[]
    free_energy_history = Float64[]
    utility_history = Vector{Float64}[]
    policy_history = Vector{Float64}[]
    entropy_history = Float64[]
    precision_history = Float64[]
    
    # Initialize true state
    true_state = rand(1:agent.num_states)
    
    for step in 1:num_steps
        step_start = time()
        
        # Record belief
        push!(belief_history, copy(agent.belief))
        
        # Calculate entropy
        entropy = -sum(agent.belief .* log.(agent.belief .+ 1e-12))
        push!(entropy_history, entropy)
        
        # Calculate free energy
        free_energy = 0.0
        for s in 1:agent.num_states
            if agent.belief[s] > 1e-12
                free_energy += agent.belief[s] * log(agent.belief[s] + 1e-12)
            end
        end
        push!(free_energy_history, free_energy)
        
        # Policy evaluation with expected utilities
        expected_utilities = Float64[]
        for action in 1:agent.num_actions
            predicted_state = agent.B[:, :, action]' * agent.belief
            predicted_obs = agent.A * predicted_state
            utility = sum(predicted_obs .* agent.C)
            push!(expected_utilities, utility)
        end
        push!(utility_history, copy(expected_utilities))
        
        # Action selection with precision
        precision = 16.0 + 2.0 * randn()
        precision = max(1.0, precision)
        push!(precision_history, precision)
        
        action_probs = enhanced_softmax(expected_utilities .* precision)
        action = rand(Categorical(action_probs))
        push!(action_history, action)
        push!(policy_history, copy(action_probs))
        
        # Environment dynamics
        transition_probs = agent.B[:, true_state, action]
        transition_probs = transition_probs ./ sum(transition_probs)
        true_state = rand(Categorical(transition_probs))
        
        # Observation generation
        obs_probs = agent.A[:, true_state]
        obs_probs = obs_probs ./ sum(obs_probs)
        observation = rand(Categorical(obs_probs))
        push!(observation_history, observation)
        
        # Reward calculation
        reward = agent.C[observation]
        push!(reward_history, reward)
        
        # Belief update
        likelihood = agent.A[observation, :]
        prior = agent.B[:, :, action]' * agent.belief
        posterior = likelihood .* prior
        posterior = posterior ./ sum(posterior)
        agent.belief = posterior
        
        # Record step data
        step_data = Dict{String,Any}(
            "step" => step,
            "true_state" => true_state,
            "action" => action,
            "observation" => observation,
            "reward" => reward,
            "free_energy" => free_energy,
            "entropy" => entropy,
            "precision" => precision,
            "belief_state" => copy(agent.belief),
            "duration_ms" => (time() - step_start) * 1000
        )
        push!(agent.simulation_history, step_data)
        
        log_step("Enhanced ActiveInference", step, Dict(
            "FE" => round(free_energy, digits=3),
            "action" => action,
            "obs" => observation,
            "reward" => round(reward, digits=3),
            "entropy" => round(entropy, digits=3)
        ))
    end
    
    # Calculate performance metrics
    total_reward = sum(reward_history)
    avg_reward = mean(reward_history)
    final_fe = free_energy_history[end]
    final_entropy = entropy_history[end]
    avg_precision = mean(precision_history)
    belief_stability = mean([var(b) for b in belief_history])
    action_diversity = length(unique(action_history)) / agent.num_actions
    
    agent.performance_metrics = Dict{String,Any}(
        "total_reward" => total_reward,
        "average_reward" => avg_reward,
        "final_free_energy" => final_fe,
        "average_free_energy" => mean(free_energy_history),
        "final_entropy" => final_entropy,
        "average_entropy" => mean(entropy_history),
        "average_precision" => avg_precision,
        "belief_stability" => belief_stability,
        "action_diversity" => action_diversity,
        "steps_completed" => num_steps,
        "simulation_duration" => sum([s["duration_ms"] for s in agent.simulation_history])
    )
    
    # Compile results
    results = Dict(
        "metadata" => Dict(
            "model_name" => "actinf_pomdp_agent",
            "framework" => "activeinference_jl_enhanced",
            "gnn_source" => "unknown.md",
            "num_steps" => num_steps
        ),
        "traces" => Dict(
            "belief_states" => belief_history,
            "actions" => action_history,
            "observations" => observation_history,
            "rewards" => reward_history,
            "free_energy" => free_energy_history,
            "entropy" => entropy_history,
            "precision" => precision_history,
            "expected_utilities" => utility_history,
            "policy_probabilities" => policy_history
        ),
        "summary" => agent.performance_metrics,
        "simulation_history" => agent.simulation_history
    )
    
    log_success("Simulation Complete", "Enhanced simulation completed successfully")
    println("  📊 Total reward: $(round(total_reward, digits=3))")
    println("  🧠 Final entropy: $(round(final_entropy, digits=3))")
    println("  ⚡ Final free energy: $(round(final_fe, digits=3))")
    println("  🎯 Action diversity: $(round(action_diversity, digits=3))")
    
    return results
end

function create_enhanced_visualizations(results, output_dir)
    log_success("Visualization", "Generating Enhanced ActiveInference.jl visualizations")
    
    viz_dir = joinpath(output_dir, "visualizations")
    mkpath(viz_dir)
    
    traces = results["traces"]
    belief_states = reduce(hcat, traces["belief_states"])'
    free_energy = traces["free_energy"]
    entropy = traces["entropy"]
    rewards = traces["rewards"]
    
    viz_files = String[]
    
    # 1. Free Energy Evolution
    p1 = plot(free_energy, 
             title="ENHANCED Free Energy Evolution - ActiveInference.jl",
             xlabel="Time Step", 
             ylabel="Free Energy",
             linewidth=3,
             alpha=0.8,
             color=:blue,
             grid=true,
             marker=:circle,
             markersize=4)
    
    fe_file = joinpath(viz_dir, "ENHANCED_free_energy_evolution.png")
    savefig(p1, fe_file)
    push!(viz_files, fe_file)
    
    # 2. Belief Evolution
    p2 = plot(title="ENHANCED Belief Evolution - ActiveInference.jl",
             xlabel="Time Step",
             ylabel="Belief Probability", 
             grid=true,
             linewidth=3)
    
    for i in 1:size(belief_states, 2)
        plot!(p2, belief_states[:, i], 
              label="State $i", 
              alpha=0.8,
              marker=:circle,
              markersize=3)
    end
    
    belief_file = joinpath(viz_dir, "ENHANCED_belief_evolution.png")
    savefig(p2, belief_file)
    push!(viz_files, belief_file)
    
    # 3. Comprehensive Dashboard
    p3 = plot(free_energy, title="Free Energy", xlabel="Step", ylabel="FE",
             linewidth=2, color=:purple, grid=true)
    p4 = plot(entropy, title="Entropy", xlabel="Step", ylabel="Entropy (nats)",
             linewidth=2, color=:red, grid=true)
    p5 = plot(cumsum(rewards), title="Cumulative Reward", xlabel="Step", ylabel="Reward",
             linewidth=2, color=:green, fill=(0, :green, 0.3), grid=true)
    
    # Final belief pie chart
    final_beliefs = belief_states[end, :]
    p6 = pie(final_beliefs, 
            title="Final Beliefs",
            labels=["State $i" for i in 1:length(final_beliefs)])
    
    dashboard = plot(p3, p4, p5, p6, 
                    layout=(2, 2),
                    size=(800, 600),
                    plot_title="ENHANCED ActiveInference.jl Dashboard")
    
    dashboard_file = joinpath(viz_dir, "ENHANCED_activeinference_dashboard.png")
    savefig(dashboard, dashboard_file)
    push!(viz_files, dashboard_file)
    
    log_success("Visualization", "Generated $(length(viz_files)) enhanced visualization files")
    
    return viz_files
end

function export_enhanced_data(results, output_dir)
    log_success("Data Export", "Exporting comprehensive data")
    
    data_dir = joinpath(output_dir, "data_exports")
    mkpath(data_dir)
    
    # JSON export with timestamp
    timestamp = string(now())
    json_file = joinpath(data_dir, "activeinference_jl_enhanced_$(replace(timestamp, ":" => "_")).json")
    
    open(json_file, "w") do f
        JSON.print(f, results, 2)
    end
    
    # Metadata export
    meta_file = joinpath(data_dir, "ENHANCED_metadata.json")
    metadata = Dict(
        "export_timestamp" => timestamp,
        "model_name" => "actinf_pomdp_agent",
        "framework" => "ActiveInference.jl Enhanced",
        "data_files" => [json_file],
        "summary" => results["summary"]
    )
    
    open(meta_file, "w") do f
        JSON.print(f, metadata, 2)
    end
    
    log_success("Data Export", "Enhanced data exported successfully")
    
    return [json_file, meta_file]
end

function main()
    try
        println("🚀 ENHANCED ActiveInference.jl Simulation")
        println("=" ^ 70)
        
        agent = create_enhanced_agent()
        results = run_enhanced_simulation(agent, 15)
        
        # Generate visualizations
        viz_files = create_enhanced_visualizations(results, ".")
        
        # Export data
        data_files = export_enhanced_data(results, ".")
        
        println("=" ^ 70)
        println("✅ ENHANCED ActiveInference.jl simulation completed!")
        println("📊 Performance: $(round(results["summary"]["total_reward"], digits=2)) total reward")
        println("🎨 Visualizations: $(length(viz_files)) files created")
        println("💾 Data exports: $(length(data_files)) files created")
        println("=" ^ 70)
        
        return results
        
    catch e
        println("❌ Enhanced ActiveInference.jl simulation failed: $e")
        rethrow(e)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
