#!/usr/bin/env julia

"""
Example script demonstrating how to use the configuration system
for the Multi-agent Trajectory Planning project.

This script:
1. Loads the configuration file
2. Creates environments and agents based on the configuration
3. Runs a simple experiment using the configured parameters

Usage:
    julia example_config_usage.jl [config_path]

Where:
    config_path: Optional path to a configuration file (default: config.toml)
"""

# Add the project path to LOAD_PATH if running the script directly
if !isinteractive()
    push!(LOAD_PATH, dirname(@__FILE__))
end

using Dates
using TrajectoryPlanning
using TrajectoryPlanning.Environment
using TrajectoryPlanning.Models
using TrajectoryPlanning.Visualizations
using TrajectoryPlanning.ConfigLoader

function main(config_path = "config.toml")
    println("Loading configuration from $config_path...")
    config = load_config(config_path)
    cfg = apply_config(config)
    
    # Display the loaded configuration
    println("Loaded configuration:")
    println("  Number of agents: $(length(cfg.agents))")
    println("  Number of environments: $(length(keys(cfg.environments)))")
    println("  Model parameters:")
    println("    dt = $(cfg.model_params.dt)")
    println("    gamma = $(cfg.model_params.gamma)")
    println("    nr_steps = $(cfg.model_params.nr_steps)")
    println("    nr_iterations = $(cfg.model_params.nr_iterations)")
    
    # Update the softmin temperature from the configuration
    configure_softmin(cfg.model_params.softmin_temperature)
    println("  Configured softmin temperature: $SOFTMIN_TEMPERATURE")
    
    # Create timestamp-based output directory
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSSs")
    output_dir = joinpath("results", "config_example_$timestamp")
    mkpath(output_dir)
    
    # Select an environment to run the experiment on
    env_name = "door"
    environment = cfg.environments[env_name]
    agents = cfg.agents
    
    # Create animated visualizations for each seed
    for seed in cfg.exp_params.seeds
        println("\nRunning experiment with seed $seed...")
        
        # Prepare goals from agent configurations
        goals = prepare_goals(agents)
        
        # Run inference with configured parameters
        result = path_planning(
            environment = environment,
            agents = agents,
            goals = goals,
            nr_steps = cfg.model_params.nr_steps,
            nr_iterations = cfg.model_params.nr_iterations,
            seed = seed,
            save_intermediates = cfg.model_params.save_intermediates,
            intermediate_steps = cfg.model_params.intermediate_steps,
            model_params = cfg.model_params,
            output_dir = output_dir
        )
        
        # Get the resulting paths
        paths = mean.(result.posteriors[:path])
        paths_matrix = reshape(paths, length(agents), cfg.model_params.nr_steps)
        
        # Create animation with configured parameters
        animation_filename = replace(cfg.exp_params.animation_template, 
                                    "{environment}" => env_name,
                                    "{seed}" => seed)
                                    
        animate_paths(
            environment, 
            agents, 
            paths_matrix, 
            filename = joinpath(output_dir, animation_filename),
            fps = cfg.vis_params.fps
        )
        
        println("  Animation saved to $(joinpath(output_dir, animation_filename))")
    end
    
    println("\nExperiment completed. Results saved to $output_dir")
    
    return output_dir
end

"""
    prepare_goals(agents)

Prepare the goals matrix from agent configurations.
"""
function prepare_goals(agents)
    nr_agents = length(agents)
    goals = Array{Vector{Float64}}(undef, 2, nr_agents)
    
    for k in 1:nr_agents
        # Initial position and velocity (velocity = 0)
        goals[1, k] = [
            agents[k].initial_position[1], 
            0.0, 
            agents[k].initial_position[2], 
            0.0
        ]
        
        # Target position and velocity (velocity = 0)
        goals[2, k] = [
            agents[k].target_position[1], 
            0.0, 
            agents[k].target_position[2], 
            0.0
        ]
    end
    
    return goals
end

# Run the main function if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    config_path = length(ARGS) > 0 ? ARGS[1] : "config.toml"
    main(config_path)
end 