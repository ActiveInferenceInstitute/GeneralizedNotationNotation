#!/usr/bin/env julia
# run_experiment.jl - Run a single multi-agent trajectory planning experiment

# Activate the local project environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Set GR backend for headless operation
ENV["GKSwstype"] = "100"

# Load the module
println("Loading TrajectoryPlanning module...")
include("TrajectoryPlanning.jl")
using .TrajectoryPlanning
using Plots
using Dates
using RxInfer
using Distributions
using CSV
using TOML
using LinearAlgebra
using Random
using DelimitedFiles

# Include necessary modules
include("Environment.jl")
include("DistanceFunctions.jl")
include("HalfspaceNode.jl")
include("Models.jl")
include("Visualizations.jl")
include("Experiments.jl")
include("config_loader.jl")

# Load the configuration
config = load_config("config.toml")

# Create or set up result directory
result_dir = joinpath("results", Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))
mkpath(result_dir)

# Run the experiment with the provided configuration
run_experiment(config; result_dir = result_dir)

# Add code to demonstrate the new visualization capabilities
println("\nGenerating additional visualizations...")

# Check if this is being run after an experiment or on existing data
function demonstrate_visualizations(result_dir)
    # Check if the result directory exists
    if !isdir(result_dir)
        println("Result directory not found: $result_dir")
        return
    end
    
    # Load data from files
    paths_file = joinpath(result_dir, "paths.csv")
    controls_file = joinpath(result_dir, "controls.csv")
    uncertainties_file = joinpath(result_dir, "uncertainties.csv")
    
    if !isfile(paths_file) || !isfile(controls_file) || !isfile(uncertainties_file)
        println("Required data files not found in: $result_dir")
        return
    end
    
    # Load the data
    println("Loading data from: $result_dir")
    
    # Load paths data
    paths_data = CSV.read(paths_file, header=false)
    nr_agents = length(unique(paths_data[!, 1]))
    nr_steps = div(size(paths_data, 1), nr_agents)
    
    # Create paths matrix
    paths = Matrix{Tuple{Float64, Float64}}(undef, nr_agents, nr_steps)
    for i in 1:size(paths_data, 1)
        agent_idx = paths_data[i, 1]
        step_idx = paths_data[i, 2]
        x = paths_data[i, 3]
        y = paths_data[i, 4]
        paths[agent_idx, step_idx] = (x, y)
    end
    
    # Load controls data
    controls_data = CSV.read(controls_file, header=false)
    
    # Create controls matrix
    controls = Matrix{Tuple{Float64, Float64}}(undef, nr_agents, nr_steps)
    for i in 1:size(controls_data, 1)
        agent_idx = controls_data[i, 1]
        step_idx = controls_data[i, 2]
        x = controls_data[i, 3]
        y = controls_data[i, 4]
        controls[agent_idx, step_idx] = (x, y)
    end
    
    # Load uncertainties data
    uncertainties_data = CSV.read(uncertainties_file, header=false)
    
    # Create uncertainties matrix
    uncertainties = Matrix{Tuple{Float64, Float64}}(undef, nr_agents, nr_steps)
    for i in 1:size(uncertainties_data, 1)
        agent_idx = uncertainties_data[i, 1]
        step_idx = uncertainties_data[i, 2]
        unc_x = uncertainties_data[i, 3]
        unc_y = uncertainties_data[i, 4]
        uncertainties[agent_idx, step_idx] = (unc_x, unc_y)
    end
    
    # Create a simple environment for visualization
    # Define obstacles based on the experiment configuration
    obstacles = []
    
    # Try to load the experiment log for environment info
    log_file = joinpath(result_dir, "experiment.log")
    if isfile(log_file)
        log_contents = read(log_file, String)
        
        # Extract obstacle information
        obstacle_pattern = r"Obstacle: center=\(([-\d\.]+), ([-\d\.]+)\), size=\(([-\d\.]+), ([-\d\.]+)\)"
        for match in eachmatch(obstacle_pattern, log_contents)
            center_x = parse(Float64, match.captures[1])
            center_y = parse(Float64, match.captures[2])
            size_x = parse(Float64, match.captures[3])
            size_y = parse(Float64, match.captures[4])
            
            push!(obstacles, Environment.Rectangle((center_x, center_y), (size_x, size_y)))
        end
    end
    
    # If no obstacles found in log, create default ones
    if isempty(obstacles)
        push!(obstacles, Environment.Rectangle((0.0, 0.0), (4.0, 2.0)))
    end
    
    # Create environment
    env = Environment.Environment(obstacles)
    
    # Create agent objects with basic properties
    agents = []
    for i in 1:nr_agents
        # Extract initial and target positions from paths
        initial_position = paths[i, 1]
        target_position = paths[i, end]
        
        # Create agent with radius 1.0
        push!(agents, Environment.Agent(initial_position, target_position, 1.0, i))
    end
    
    # Generate various visualizations
    println("Generating visualizations...")
    
    # 1. Path uncertainty visualization
    println("  - Path uncertainty visualization")
    plot_path_uncertainties(agents, paths, uncertainties, 
                          filename = joinpath(result_dir, "path_uncertainty.png"),
                          uncertainty_scale = 3.0,
                          show_step = 5)  # Show every 5th step for clarity
    
    # 2. Control signals visualization
    println("  - Control signals visualization")
    plot_control_signals(controls, 
                       filename = joinpath(result_dir, "control_signals.png"),
                       component = :both)
    
    plot_control_signals(controls, 
                       filename = joinpath(result_dir, "control_magnitudes.png"),
                       component = :magnitude)
    
    # 3. Combined visualization at different timesteps
    println("  - Combined visualization at different timesteps")
    # Start, middle and end points
    for (idx, step) in enumerate([1, div(nr_steps, 2), nr_steps])
        plot_combined_visualization(env, agents, paths, uncertainties, controls,
                                  filename = joinpath(result_dir, "combined_$(idx).png"),
                                  step = step,
                                  uncertainty_scale = 2.0,
                                  control_scale = 5.0)
    end
    
    # 4. 3D path visualization for each agent
    println("  - 3D path visualizations")
    for i in 1:nr_agents
        plot_agent_path_3d(paths, i,
                         uncertainties = uncertainties,
                         filename = joinpath(result_dir, "agent$(i)_path_3d.png"),
                         with_time = true)
        
        plot_agent_path_3d(paths, i,
                         uncertainties = uncertainties,
                         filename = joinpath(result_dir, "agent$(i)_path_3d_uncertainty.png"),
                         with_time = false)
    end
    
    # 5. Path density heatmap
    println("  - Path density heatmap")
    create_path_heatmap(env, paths,
                      filename = joinpath(result_dir, "path_heatmap.png"),
                      resolution = 100)
    
    # 6. Generate animations if requested
    println("  - Generating animations (this may take some time)...")
    
    # Path uncertainty animation
    animate_path_uncertainties(env, agents, paths, uncertainties,
                             filename = joinpath(result_dir, "path_uncertainty.gif"),
                             uncertainty_scale = 3.0,
                             show_full_path = false)
    
    # Control signals animation
    animate_control_signals(env, paths, controls,
                          filename = joinpath(result_dir, "control_signals.gif"),
                          control_scale = 5.0)
    
    # Combined visualization animation
    animate_combined_visualization(env, agents, paths, uncertainties, controls,
                                 filename = joinpath(result_dir, "combined_visualization.gif"),
                                 uncertainty_scale = 2.0,
                                 control_scale = 5.0,
                                 show_trail = true,
                                 trail_length = 10)
    
    println("All visualizations completed and saved to: $result_dir")
end

# Run the visualization demonstration
demonstrate_visualizations(result_dir)

# Also add capability to visualize existing results
if length(ARGS) > 0
    existing_result_dir = ARGS[1]
    println("\nVisualizing existing results from: $existing_result_dir")
    demonstrate_visualizations(existing_result_dir)
end

"""
    create_output_directories(base_dir)

Create organized subdirectories for output files.

# Arguments
- `base_dir`: Base directory path

# Returns
- Dictionary with paths to subdirectories
"""
function create_output_directories(base_dir)
    # Create main subdirectories
    subdirs = Dict(
        "animations" => joinpath(base_dir, "animations"),
        "visualizations" => joinpath(base_dir, "visualizations"),
        "data" => joinpath(base_dir, "data"),
        "heatmaps" => joinpath(base_dir, "heatmaps")
    )
    
    # Create each directory
    for (_, dir) in subdirs
        mkpath(dir)
    end
    
    return subdirs
end