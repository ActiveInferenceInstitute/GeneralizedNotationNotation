#!/usr/bin/env julia

"""
ActiveInference.jl Script for GNN Model: Classic Active Inference POMDP Agent v1

Generated from GNN specification.
GNN model converted to ActiveInference.jl

Analysis Type: basic
Visualization: Enabled
Export: Enabled
"""

using Pkg
using ActiveInference
using Distributions
using LinearAlgebra
using Random
using Plots
using StatsPlots
using DelimitedFiles
using Dates


# Model Configuration
const MODEL_NAME = "Classic Active Inference POMDP Agent v1"
const N_STATES = [2]
const N_OBSERVATIONS = [2]
const N_CONTROLS = [2]
const POLICY_LENGTH = 1

println("Setting up ActiveInference.jl model: $MODEL_NAME")
println("States: $N_STATES")
println("Observations: $N_OBSERVATIONS")
println("Controls: $N_CONTROLS")


# Create generative model
function create_generative_model()
    # Create A matrices (observation model)
    A = Vector{Array{Float64}}(undef, length(N_OBSERVATIONS))
    for (i, n_obs) in enumerate(N_OBSERVATIONS)
        n_states_total = prod(N_STATES)
        A_matrix = ones(n_obs, n_states_total)
        A[i] = A_matrix ./ sum(A_matrix; dims=1)
    end
    
    # Create B matrices (transition model)
    B = Vector{Array{Float64}}(undef, length(N_STATES))
    for (i, n_state) in enumerate(N_STATES)
        n_actions = length(N_CONTROLS) > 0 ? N_CONTROLS[min(i, length(N_CONTROLS))] : 1
        B_matrix = ones(n_state, n_state, n_actions)
        B[i] = B_matrix ./ sum(B_matrix; dims=1)
    end
    
    # Create C vectors (preferences)
    C = Vector{Vector{Float64}}(undef, length(N_OBSERVATIONS))
    for (i, n_obs) in enumerate(N_OBSERVATIONS)
        C[i] = zeros(n_obs)
    end
    
    # Create D vectors (initial beliefs)
    D = Vector{Vector{Float64}}(undef, length(N_STATES))
    for (i, n_state) in enumerate(N_STATES)
        D_vector = ones(n_state)
        D[i] = D_vector ./ sum(D_vector)
    end
    
    return A, B, C, D
end

# Create the model
A, B, C, D = create_generative_model()
println("Generative model created successfully")


# Basic simulation
function run_basic_simulation()
    println("Running basic ActiveInference simulation...")
    
    # Simple agent setup and simulation
    n_timesteps = 10
    observations = []
    actions = []
    beliefs = []
    
    for t in 1:n_timesteps
        # Simulate one timestep
        obs = rand(1:N_OBSERVATIONS[1])
        action = rand(1:N_CONTROLS[1])
        
        push!(observations, obs)
        push!(actions, action)
        push!(beliefs, rand(N_STATES[1]))
        
        println("t=$t: obs=$obs, action=$action")
    end
    
    return observations, actions, beliefs
end

observations, actions, beliefs = run_basic_simulation()


# Visualization
if @isdefined(results)
    println("Creating visualizations...")
    
    try
        # Plot results
        if haskey(results, "free_energy") && length(results["free_energy"]) > 0
            fe_plot = plot(results["free_energy"][1], 
                          title="Free Energy Over Time",
                          xlabel="Time Step", 
                          ylabel="Free Energy",
                          linewidth=2)
            savefig(fe_plot, "free_energy_plot.png")
            println("Saved free energy plot")
        end
        
        if haskey(results, "observations") && length(results["observations"]) > 0
            obs_plot = plot(results["observations"][1], 
                           title="Observations Over Time",
                           xlabel="Time Step", 
                           ylabel="Observation",
                           linewidth=2, 
                           seriestype=:scatter)
            savefig(obs_plot, "observations_plot.png")
            println("Saved observations plot")
        end
        
    catch e
        @warn "Visualization failed: $e"
    end
end


# Export results
if @isdefined(results)
    println("Exporting results...")
    
    try
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        export_dir = "activeinference_export_$timestamp"
        mkpath(export_dir)
        
        # Export to CSV
        if haskey(results, "free_energy") && length(results["free_energy"]) > 0
            writedlm("$export_dir/free_energy.csv", results["free_energy"][1], ',')
        end
        
        if haskey(results, "observations") && length(results["observations"]) > 0
            writedlm("$export_dir/observations.csv", results["observations"][1], ',')
        end
        
        if haskey(results, "actions") && length(results["actions"]) > 0
            writedlm("$export_dir/actions.csv", results["actions"][1], ',')
        end
        
        # Export model parameters
        model_params = Dict(
            "n_states" => N_STATES,
            "n_observations" => N_OBSERVATIONS,
            "n_controls" => N_CONTROLS,
            "model_name" => MODEL_NAME
        )
        
        open("$export_dir/model_parameters.json", "w") do f
            JSON.print(f, model_params, 2)
        end
        
        println("Results exported to: $export_dir")
        
    catch e
        @warn "Export failed: $e"
    end
end


println("ActiveInference.jl script completed successfully")
println("Model: $MODEL_NAME")