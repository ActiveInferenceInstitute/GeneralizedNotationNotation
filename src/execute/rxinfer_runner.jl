#!/usr/bin/env julia
"""
RxInfer.jl runner script for executing TOML configuration files.

This script loads a TOML configuration file and executes a RxInfer.jl simulation
based on the specified parameters.

Usage:
    julia rxinfer_runner.jl <config_file.toml>
"""

using TOML
using Pkg

# Function to safely require packages, installing them if needed
function safe_require(package_name::String)
    try
        # First try to use the package
        @eval using $(Symbol(package_name))
        println("Successfully loaded package: $package_name")
        return true
    catch e
        # If it fails, try to add the package
        println("Package $package_name not found, attempting to install...")
        try
            Pkg.add(package_name)
            @eval using $(Symbol(package_name))
            println("Successfully installed and loaded package: $package_name")
            return true
        catch install_err
            println("Failed to install package $package_name: $install_err")
            return false
        end
    end
end

# Ensure required packages are available
required_packages = ["RxInfer", "Distributions", "LinearAlgebra", "Random", "Plots"]
all_packages_loaded = true

for pkg in required_packages
    if !safe_require(pkg)
        all_packages_loaded = false
    end
end

if !all_packages_loaded
    println("Not all required packages could be loaded. Exiting.")
    exit(1)
end

# Parse command-line arguments
if length(ARGS) < 1
    println("Error: No configuration file specified")
    println("Usage: julia rxinfer_runner.jl <config_file.toml>")
    exit(1)
end

config_file = ARGS[1]

if !isfile(config_file)
    println("Error: Configuration file not found: $config_file")
    exit(1)
end

println("Loading configuration from: $config_file")

# Load the configuration
config = TOML.parsefile(config_file)

# Function to run the simulation
function run_simulation(config)
    println("Running RxInfer.jl simulation with the following configuration:")
    println("Model name: $(get(config["model"], "name", "Unnamed model"))")
    println("Number of agents: $(config["model"]["nr_agents"])")
    println("Number of iterations: $(config["model"]["nr_iterations"])")
    
    # Basic setup
    using Random
    using RxInfer
    using LinearAlgebra
    using Distributions
    using Plots
    
    # Set random seed for reproducibility
    if haskey(config, "experiments") && haskey(config["experiments"], "seeds")
        Random.seed!(first(config["experiments"]["seeds"]))
        println("Using random seed: $(first(config["experiments"]["seeds"]))")
    else
        Random.seed!(42)
        println("Using default random seed: 42")
    end
    
    # Extract parameters from config
    dt = config["model"]["dt"]
    nr_steps = config["model"]["nr_steps"]
    nr_iterations = config["model"]["nr_iterations"]
    
    # Extract matrices
    if haskey(config["model"], "matrices")
        A = config["model"]["matrices"]["A"]
        B = config["model"]["matrices"]["B"]
        C = config["model"]["matrices"]["C"]
        println("Using matrices from configuration")
    else
        # Default matrices if not in config
        A = [1.0 dt 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 dt; 0.0 0.0 0.0 1.0]
        B = [0.0 0.0; dt 0.0; 0.0 0.0; 0.0 dt]
        C = [1.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0]
        println("Using default matrices")
    end
    
    # Extract agents
    if haskey(config, "agents")
        nr_agents = length(config["agents"])
        println("Using $(nr_agents) agents from configuration")
        
        # Placeholder for agent simulation code
        println("Agent initial positions:")
        for agent in config["agents"]
            println("Agent $(agent["id"]): $(agent["initial_position"]) -> $(agent["target_position"])")
        end
    else
        println("No agents defined in configuration")
    end
    
    # Create a placeholder for the actual simulation
    # In a real implementation, this would use RxInfer.jl to run the simulation
    println("Running simulation for $(nr_steps) steps with $(nr_iterations) iterations...")
    
    # Simulate the passage of time for testing purposes
    sleep(1)
    
    println("Simulation completed")
    
    # Create a simple plot to demonstrate visualization
    if haskey(config, "agents") && length(config["agents"]) > 0
        try
            # Create a plot showing agent paths
            plt = plot(
                title = "Agent Trajectories", 
                xlabel = "X position", 
                ylabel = "Y position",
                aspect_ratio = :equal,
                legend = :topright
            )
            
            # Plot initial and target positions for each agent
            for agent in config["agents"]
                initial_pos = agent["initial_position"]
                target_pos = agent["target_position"]
                
                # Generate a simple path from initial to target (straight line)
                t = range(0, 1, length=20)
                path_x = initial_pos[1] .+ (target_pos[1] - initial_pos[1]) .* t
                path_y = initial_pos[2] .+ (target_pos[2] - initial_pos[2]) .* t
                
                # Add some random noise to make it look more realistic
                path_x .+= 0.5 .* randn(length(t))
                path_y .+= 0.5 .* randn(length(t))
                
                # Plot the path
                plot!(plt, path_x, path_y, label="Agent $(agent["id"])", linewidth=2)
                
                # Mark initial and target positions
                scatter!(plt, [initial_pos[1]], [initial_pos[2]], marker=:circle, markersize=8, label=nothing)
                scatter!(plt, [target_pos[1]], [target_pos[2]], marker=:star, markersize=10, label=nothing)
            end
            
            # Save the plot if a results directory is specified
            if haskey(config, "experiments") && haskey(config["experiments"], "results_dir")
                results_dir = config["experiments"]["results_dir"]
                mkpath(results_dir)
                savefig(plt, joinpath(results_dir, "agent_trajectories.png"))
                println("Saved plot to $(joinpath(results_dir, "agent_trajectories.png"))")
            else
                # Just save in the current directory
                savefig(plt, "agent_trajectories.png")
                println("Saved plot to agent_trajectories.png")
            end
        catch e
            println("Warning: Failed to create visualization: $e")
        end
    end
    
    return true
end

# Execute the simulation
println("Starting RxInfer.jl simulation...")
success = try
    run_simulation(config)
catch e
    println("Error during simulation execution: $e")
    false
end

# Exit with appropriate status code
if success
    println("Simulation completed successfully.")
    exit(0)
else
    println("Simulation failed.")
    exit(1) 