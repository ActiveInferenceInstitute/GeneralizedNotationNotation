using Plots
using DelimitedFiles
using Dates
using Statistics
using LinearAlgebra

# Add the current directory to the load path
if !isinteractive()
    push!(LOAD_PATH, dirname(@__FILE__))
end

# Import TrajectoryPlanning module
include("TrajectoryPlanning.jl")
using .TrajectoryPlanning

"""
    visualize_results(result_dir, subdirs=nothing)

Generate advanced visualizations from result data.

# Arguments
- `result_dir`: Path to the results directory
- `subdirs`: Optional dictionary of subdirectories (animations, visualizations, data, heatmaps)
"""
function visualize_results(result_dir, subdirs=nothing)
    # Check if the directory exists
    if !isdir(result_dir)
        error("Result directory not found: $result_dir")
    end
    
    # Set up directory structure if not provided
    if subdirs === nothing
        subdirs = Dict(
            "animations" => joinpath(result_dir, "animations"),
            "visualizations" => joinpath(result_dir, "visualizations"),
            "data" => joinpath(result_dir, "data"),
            "heatmaps" => joinpath(result_dir, "heatmaps")
        )
        
        # Create directories if they don't exist
        for (_, dir) in subdirs
            if !isdir(dir)
                mkpath(dir)
            end
        end
    end
    
    # Try to load and visualize the path data
    try
        # First check in the data subdirectory, then in the root directory
        path_file = joinpath(subdirs["data"], "paths.csv")
        if !isfile(path_file)
            path_file = joinpath(result_dir, "paths.csv")
        end
        
        if isfile(path_file)
            paths = load_path_data(path_file)
            println("Loaded path data: $(length(paths)) points")
            
            # Create visualization in the visualizations subdirectory
            viz_file = joinpath(subdirs["visualizations"], "path_visualization.png")
            create_path_visualization(paths, viz_file)
            println("Saved path visualization.")
        else
            println("Warning: Path data file not found.")
        end
    catch e
        println("Error visualizing path data: $e")
    end
    
    # Try to load and visualize the control data
    try
        # First check in the data subdirectory, then in the root directory
        control_file = joinpath(subdirs["data"], "controls.csv")
        if !isfile(control_file)
            control_file = joinpath(result_dir, "controls.csv")
        end
        
        if isfile(control_file)
            controls = load_control_data(control_file)
            println("Loaded control data: $(length(controls)) points")
            
            # Create visualization in the visualizations subdirectory
            viz_file = joinpath(subdirs["visualizations"], "control_magnitudes.png")
            create_control_visualization(controls, viz_file)
            println("Saved control magnitude visualization.")
        else
            println("Warning: Control data file not found.")
        end
    catch e
        println("Error visualizing control data: $e")
    end
    
    # Try to load and visualize convergence metrics
    try
        # First check in the data subdirectory, then in the root directory
        metrics_file = joinpath(subdirs["data"], "convergence_metrics.csv")
        if !isfile(metrics_file)
            metrics_file = joinpath(result_dir, "convergence_metrics.csv")
        end
        
        if isfile(metrics_file)
            metrics = load_convergence_metrics(metrics_file)
            
            # Create visualization in the visualizations subdirectory
            viz_file = joinpath(subdirs["visualizations"], "convergence_detailed.png")
            create_convergence_visualization(metrics, viz_file)
            println("Saved detailed convergence visualization.")
        else
            println("Warning: Convergence metrics file not found.")
        end
    catch e
        println("Error visualizing convergence metrics: $e")
    end
end

"""
    load_path_data(file_path)

Load path data from a CSV file.

# Arguments
- `file_path`: Path to the CSV file

# Returns
- Dictionary of path data
"""
function load_path_data(file_path)
    # Parse the CSV file
    paths = Dict()
    
    # Read the file
    try
        lines = readlines(file_path)
        for line in lines
            # Skip header or empty lines
            if isempty(line) || startswith(line, "agent") || occursin("agent,step,x,y", line)
                continue
            end
            
            # Parse the line
            parts = split(line, ',')
            if length(parts) >= 4
                agent = parse(Int, parts[1])
                step = parse(Int, parts[2])
                x = parse(Float64, parts[3])
                y = parse(Float64, parts[4])
                
                if !haskey(paths, agent)
                    paths[agent] = Dict()
                end
                
                paths[agent][step] = (x, y)
            end
        end
    catch e
        println("Error parsing path data: $e")
    end
    
    return paths
end

"""
    load_control_data(file_path)

Load control data from a CSV file.

# Arguments
- `file_path`: Path to the CSV file

# Returns
- Dictionary of control data
"""
function load_control_data(file_path)
    # Parse the CSV file
    controls = Dict()
    
    # Read the file
    try
        lines = readlines(file_path)
        for line in lines
            # Skip header or empty lines
            if isempty(line) || startswith(line, "agent") || occursin("agent,step,x_control,y_control", line)
                continue
            end
            
            # Parse the line
            parts = split(line, ',')
            if length(parts) >= 4
                agent = parse(Int, parts[1])
                step = parse(Int, parts[2])
                x_control = parse(Float64, parts[3])
                y_control = parse(Float64, parts[4])
                
                if !haskey(controls, agent)
                    controls[agent] = Dict()
                end
                
                controls[agent][step] = (x_control, y_control)
            end
        end
    catch e
        println("Error parsing control data: $e")
    end
    
    return controls
end

"""
    load_convergence_metrics(file_path)

Load convergence metrics from a CSV file.

# Arguments
- `file_path`: Path to the CSV file

# Returns
- Array of ELBO values
"""
function load_convergence_metrics(file_path)
    # Parse the CSV file
    metrics = Float64[]
    
    # Read the file
    try
        lines = readlines(file_path)
        for line in lines
            # Skip header or empty lines
            if isempty(line) || startswith(line, "iteration") || occursin("iteration,elbo", line)
                continue
            end
            
            # Parse the line
            parts = split(line, ',')
            if length(parts) >= 2
                elbo = parse(Float64, parts[2])
                push!(metrics, elbo)
            end
        end
    catch e
        println("Error parsing convergence metrics: $e")
    end
    
    return metrics
end

"""
    create_path_visualization(paths, output_file)

Create a visualization of agent paths and save it to a file.

# Arguments
- `paths`: Dictionary of path data
- `output_file`: Path to the output file
"""
function create_path_visualization(paths, output_file)
    # Create a new plot
    p = plot(
        size=(800, 600),
        xlabel="X",
        ylabel="Y",
        title="Agent Paths",
        legend=true,
        xlim=(-20, 20),
        ylim=(-20, 20),
        grid=true
    )
    
    # Colors for different agents
    colors = [:blue, :orange, :green, :red, :purple, :brown, :pink, :gray]
    
    # Plot each agent's path
    for (agent, steps) in paths
        xs = Float64[]
        ys = Float64[]
        
        # Sort steps by step number
        sorted_steps = sort(collect(keys(steps)))
        
        for step in sorted_steps
            x, y = steps[step]
            push!(xs, x)
            push!(ys, y)
        end
        
        # Plot the path
        plot!(p, xs, ys, 
             label="Agent $agent", 
             linewidth=2, 
             color=colors[mod1(agent, length(colors))],
             marker=:circle,
             markersize=3,
             markerstrokewidth=0)
        
        # Mark start and end points
        if length(xs) > 0
            scatter!(p, [xs[1]], [ys[1]], 
                   marker=:star, 
                   color=colors[mod1(agent, length(colors))],
                   markersize=8,
                   label=nothing)
                   
            scatter!(p, [xs[end]], [ys[end]], 
                   marker=:square, 
                   color=colors[mod1(agent, length(colors))],
                   markersize=8,
                   label=nothing)
        end
    end
    
    # Save the plot
    savefig(p, output_file)
end

"""
    create_control_visualization(controls, output_file)

Create a visualization of control magnitudes and save it to a file.

# Arguments
- `controls`: Dictionary of control data
- `output_file`: Path to the output file
"""
function create_control_visualization(controls, output_file)
    # Create a new plot
    p = plot(
        size=(800, 400),
        xlabel="Step",
        ylabel="Magnitude",
        title="Control Magnitudes",
        legend=true,
        grid=true
    )
    
    # Colors for different agents
    colors = [:blue, :orange, :green, :red, :purple, :brown, :pink, :gray]
    
    # Plot each agent's control magnitude
    for (agent, steps) in controls
        xs = Int[]
        magnitudes = Float64[]
        
        # Sort steps by step number
        sorted_steps = sort(collect(keys(steps)))
        
        for step in sorted_steps
            x_control, y_control = steps[step]
            magnitude = sqrt(x_control^2 + y_control^2)
            
            push!(xs, step)
            push!(magnitudes, magnitude)
        end
        
        # Plot the control magnitude
        plot!(p, xs, magnitudes, 
             label="Agent $agent", 
             linewidth=2, 
             color=colors[mod1(agent, length(colors))],
             marker=:circle,
             markersize=3,
             markerstrokewidth=0)
    end
    
    # Save the plot
    savefig(p, output_file)
end

"""
    create_convergence_visualization(metrics, output_file)

Create a visualization of convergence metrics and save it to a file.

# Arguments
- `metrics`: Array of ELBO values
- `output_file`: Path to the output file
"""
function create_convergence_visualization(metrics, output_file)
    if isempty(metrics)
        # Create a placeholder plot if no metrics are available
        p = plot(
            [0, 100], [0, 0], 
            linewidth=0, 
            xlabel="Iteration", 
            ylabel="ELBO", 
            title="Convergence of Inference", 
            legend=false, 
            size=(800, 400),
            annotations=[(50, 0.5, Plots.text("Data Not Found", :red, 14))]
        )
    else
        # Create a new plot
        p = plot(
            size=(800, 400),
            xlabel="Iteration",
            ylabel="ELBO",
            title="Convergence of Inference",
            legend=false,
            grid=true
        )
        
        # Plot the ELBO values
        plot!(p, 1:length(metrics), metrics, 
             linewidth=2, 
             color=:blue,
             marker=:circle,
             markersize=3,
             markerstrokewidth=0)
    end
    
    # Save the plot
    savefig(p, output_file)
end

"""
    visualize_environment(environment_name, output_dir=".")

Generate a visualization of the specified environment.

# Arguments
- `environment_name`: Name of the environment ("door", "wall", or "combined")
- `output_dir`: Directory to save the visualization (default: current directory)

# Returns
- Nothing. Visualization is saved to the output directory.
"""
function visualize_environment(environment_name, output_dir=".")
    println("Visualizing $environment_name environment...")
    
    # Create the environment based on name
    environment = if environment_name == "door"
        TrajectoryPlanning.create_door_environment()
    elseif environment_name == "wall"
        TrajectoryPlanning.create_wall_environment()
    elseif environment_name == "combined"
        TrajectoryPlanning.create_combined_environment()
    else
        error("Unknown environment: $environment_name")
    end
    
    # Create a plot with the environment
    p = plot(title="$environment_name Environment", 
             xlims=(-20, 20), ylims=(-20, 20), 
             aspect_ratio=:equal, size=(800, 600))
    
    # Plot each obstacle
    for obstacle in environment.obstacles
        # Calculate the x-coordinates of the four corners
        x_coords = obstacle.center[1] .+ obstacle.size[1]/2 * [-1, 1, 1, -1, -1]
        # Calculate the y-coordinates of the four corners
        y_coords = obstacle.center[2] .+ obstacle.size[2]/2 * [-1, -1, 1, 1, -1]
        
        # Plot the rectangle with a black fill
        plot!(p, Shape(x_coords, y_coords), 
              label = "", 
              color = :black, 
              alpha = 0.5,
              linewidth = 1.5,
              fillalpha = 0.3)
    end
    
    # Save the plot
    filename = joinpath(output_dir, "$(environment_name)_environment.png")
    savefig(p, filename)
    println("Saved environment visualization to $filename")
    
    return nothing
end

"""
    compare_environments(output_dir=".")

Generate a comparison visualization of all standard environments.

# Arguments
- `output_dir`: Directory to save the visualization (default: current directory)

# Returns
- Nothing. Visualization is saved to the output directory.
"""
function compare_environments(output_dir=".")
    println("Comparing all environments...")
    
    # Create environments
    door_env = TrajectoryPlanning.create_door_environment()
    wall_env = TrajectoryPlanning.create_wall_environment()
    combined_env = TrajectoryPlanning.create_combined_environment()
    
    # Create a plot layout with 3 subplots
    p = plot(layout=(1,3), size=(1200, 400), title=["Door Environment" "Wall Environment" "Combined Environment"])
    
    # Helper function to plot an environment in a subplot
    function plot_env_in_subplot!(p, env, idx)
        plot!(p, subplot=idx, xlims=(-20, 20), ylims=(-20, 20), aspect_ratio=:equal)
        
        for obstacle in env.obstacles
            # Calculate the x-coordinates of the four corners
            x_coords = obstacle.center[1] .+ obstacle.size[1]/2 * [-1, 1, 1, -1, -1]
            # Calculate the y-coordinates of the four corners
            y_coords = obstacle.center[2] .+ obstacle.size[2]/2 * [-1, -1, 1, 1, -1]
            
            # Plot the rectangle with a black fill
            plot!(p, subplot=idx, Shape(x_coords, y_coords), 
                  label = "", 
                  color = :black, 
                  alpha = 0.5,
                  linewidth = 1.5,
                  fillalpha = 0.3)
        end
    end
    
    # Plot each environment in its subplot
    plot_env_in_subplot!(p, door_env, 1)
    plot_env_in_subplot!(p, wall_env, 2)
    plot_env_in_subplot!(p, combined_env, 3)
    
    # Save the plot
    filename = joinpath(output_dir, "environment_comparison.png")
    savefig(p, filename)
    println("Saved environment comparison to $filename")
    
    return nothing
end

"""
    visualize_latest_results()

Visualize results from the most recent experiment.

# Returns
- Nothing. Visualizations are saved to the results directory.
"""
function visualize_latest_results()
    # Try to find results directory
    if isdir("results")
        subdirs = filter(d -> isdir(joinpath("results", d)), readdir("results"))
        if !isempty(subdirs)
            # Sort by directory name (assumes timestamp format)
            sort!(subdirs, rev=true)
            latest_dir = joinpath("results", subdirs[1])
            
            println("Found latest results directory: $latest_dir")
            visualize_results(latest_dir)
        else
            println("No result directories found.")
        end
    else
        println("Results directory not found.")
    end
    
    return nothing
end

# If run as a script, visualize the latest results
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) > 0
        # If a directory is provided, visualize results from that directory
        visualize_results(ARGS[1])
    else
        # Otherwise, visualize the latest results
        visualize_latest_results()
    end
end 