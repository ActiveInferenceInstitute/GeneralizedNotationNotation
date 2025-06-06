module Experiments

using ..Environment
using ..Models
using ..Visualizations
using Dates

export execute_and_save_animation, run_all_experiments

function execute_and_save_animation(environment, agents; gifname = "result.gif", output_subdirs = nothing, kwargs...)
    println("Planning paths for environment with $(length(environment.obstacles)) obstacles...")
    
    # Check if a logger is provided in kwargs
    logger = get(kwargs, :logger, nothing)
    
    # Extract the output directory
    output_dir = dirname(gifname)
    output_dir = (output_dir == "") ? "." : output_dir
    
    # Use subdirectories if provided
    if output_subdirs !== nothing
        # Update output paths
        animation_dir = output_subdirs["animations"]
        visualization_dir = output_subdirs["visualizations"]
        data_dir = output_subdirs["data"]
        
        # Update gifname to use animations subdirectory
        gifname = joinpath(animation_dir, basename(gifname))
    end
    
    # Run the inference
    result = path_planning(environment = environment, agents = agents; kwargs...)
    
    # Extract path means for visualization
    paths = mean.(result.posteriors[:path])
    controls = mean.(result.posteriors[:control])
    path_vars = var.(result.posteriors[:path])
    
    # Create animation and save it
    animate_paths(environment, agents, paths; filename = gifname)
    
    # Check for ELBO tracking
    elbo_tracked = false
    elbo_values = Float64[]
    
    # Extract ELBO values if available
    if hasfield(typeof(result), :diagnostics) && haskey(result.diagnostics, :elbo)
        elbo_values = result.diagnostics[:elbo]
        
        if !isempty(elbo_values)
            elbo_tracked = true
            
            # Log success in ELBO tracking
            log_msg = "ELBO tracking successful. Collected $(length(elbo_values)) values."
            if logger !== nothing
                log_message(log_msg, logger)
            else
                println(log_msg)
            end
            
            # Create ELBO convergence plot
            log_msg = "Generating ELBO convergence plot..."
            if logger !== nothing
                log_message(log_msg, logger)
            else
                println(log_msg)
            end
            
            # Use visualization subdirectory if available
            convergence_file = output_subdirs !== nothing ? 
                joinpath(output_subdirs["visualizations"], "convergence.png") :
                joinpath(output_dir, "convergence.png")
                
            plot_elbo_convergence(elbo_values, filename = convergence_file)
            
            # Save raw ELBO data
            metrics_file = output_subdirs !== nothing ?
                joinpath(output_subdirs["data"], "convergence_metrics.csv") :
                joinpath(output_dir, "convergence_metrics.csv")
                
            open(metrics_file, "w") do f
                for (i, val) in enumerate(elbo_values)
                    println(f, "$i,$val")
                end
            end
            
            log_msg = "Saved convergence metrics to $metrics_file"
            if logger !== nothing
                log_message(log_msg, logger)
            else
                println(log_msg)
            end
            
            # Log convergence quality
            if length(elbo_values) >= 2
                initial_elbo = elbo_values[1]
                final_elbo = elbo_values[end]
                improvement = final_elbo - initial_elbo
                
                log_msg = "Inference converged with ELBO improvement of $improvement (from $initial_elbo to $final_elbo)"
                if logger !== nothing
                    log_message(log_msg, logger)
                else
                    println(log_msg)
                end
            end
        end
    end
    
    # If ELBO wasn't tracked successfully, log the issue
    if !elbo_tracked
        log_msg = "ELBO tracking was not successful. This could be due to:"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "  - Callback mechanism not properly configured"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "  - RxInfer.jl not exposing free_energy field in metadata"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "  - ELBO values not being stored correctly"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        log_msg = "Generating placeholder convergence plot..."
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
        
        # Generate a placeholder convergence plot
        using Plots
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
        
        # Use visualization subdirectory if available
        convergence_file = output_subdirs !== nothing ?
            joinpath(output_subdirs["visualizations"], "convergence.png") :
            joinpath(output_dir, "convergence.png")
            
        savefig(p, convergence_file)
        
        log_msg = "Placeholder convergence plot saved to $convergence_file"
        if logger !== nothing
            log_message(log_msg, logger)
        else
            println(log_msg)
        end
    end
    
    # Save raw data files to the data subdirectory
    if output_subdirs !== nothing
        # Save paths, controls, and uncertainties to data subdirectory
        save_path_data(paths, joinpath(output_subdirs["data"], "paths.csv"))
        save_control_data(controls, joinpath(output_subdirs["data"], "controls.csv"))
        save_uncertainty_data(path_vars, joinpath(output_subdirs["data"], "uncertainties.csv"))
        
        # Create additional visualizations in the visualizations subdirectory
        visualize_obstacle_distance(environment, filename=joinpath(output_subdirs["heatmaps"], "obstacle_distance.png"))
        visualize_path_uncertainty(environment, agents, paths, path_vars, 
                                  filename=joinpath(output_subdirs["visualizations"], "path_uncertainty.png"))
        plot_control_magnitudes(agents, controls, 
                               filename=joinpath(output_subdirs["visualizations"], "control_magnitudes.png"))
    end
    
    return paths
end

function run_all_experiments()
    # Create environments
    door_environment = create_door_environment()
    wall_environment = create_wall_environment()
    combined_environment = create_combined_environment()
    
    # Create agents
    agents = create_standard_agents()
    
    # Create timestamp-based output directory
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_dir = joinpath("results", timestamp)
    mkpath(output_dir)
    
    # Create organized subdirectories
    subdirs = create_output_directories(output_dir)
    
    println("Running experiments for door environment...")
    execute_and_save_animation(door_environment, agents; 
                               seed = 42, 
                               gifname = joinpath(output_dir, "door_42.gif"),
                               output_subdirs = subdirs)

    println("Running experiments with different seed...")
    execute_and_save_animation(door_environment, agents; 
                               seed = 123, 
                               gifname = joinpath(output_dir, "door_123.gif"),
                               output_subdirs = subdirs)

    println("Running experiments for wall environment...")
    execute_and_save_animation(wall_environment, agents; 
                               seed = 42, 
                               gifname = joinpath(output_dir, "wall_42.gif"),
                               output_subdirs = subdirs)

    println("Running experiments with different seed...")
    execute_and_save_animation(wall_environment, agents; 
                               seed = 123, 
                               gifname = joinpath(output_dir, "wall_123.gif"),
                               output_subdirs = subdirs)

    println("Running experiments for combined environment...")
    execute_and_save_animation(combined_environment, agents; 
                               seed = 42, 
                               gifname = joinpath(output_dir, "combined_42.gif"),
                               output_subdirs = subdirs)

    println("Running final experiment...")
    execute_and_save_animation(combined_environment, agents; 
                               seed = 123, 
                               gifname = joinpath(output_dir, "combined_123.gif"),
                               output_subdirs = subdirs)
    
    # Create a README for the results
    create_readme(output_dir, subdirs)

    println("All experiments completed successfully.")
    println("Results saved to: $output_dir")
end

"""
    create_readme(output_dir, subdirs)

Create a README.md file in the output directory describing the contents.
"""
function create_readme(output_dir, subdirs)
    readme_path = joinpath(output_dir, "README.md")
    open(readme_path, "w") do io
        write(io, "# Multi-agent Trajectory Planning Results\n\n")
        write(io, "Generated at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n")
        
        write(io, "## Contents\n\n")
        
        # Animations
        write(io, "### Animations\n")
        animation_files = filter(file -> endswith(file, ".gif"), readdir(subdirs["animations"]))
        for file in animation_files
            desc = if occursin("door", file)
                "Door environment (seed $(replace(file, r"door_|.gif" => "")))"
            elseif occursin("wall", file)
                "Wall environment (seed $(replace(file, r"wall_|.gif" => "")))"
            elseif occursin("combined", file)
                "Combined environment (seed $(replace(file, r"combined_|.gif" => "")))"
            elseif occursin("control_signals", file)
                "Control signals for each agent"
            else
                file
            end
            write(io, "- `$(file)` - $desc\n")
        end
        write(io, "\n")
        
        # Visualizations
        write(io, "### Visualizations\n")
        viz_files = filter(file -> endswith(file, ".png"), readdir(subdirs["visualizations"]))
        for file in viz_files
            desc = if occursin("control_magnitudes", file)
                "Plot of control signal magnitudes over time"
            elseif occursin("path_uncertainty", file)
                "Visualization of path uncertainties"
            elseif occursin("path_visualization", file)
                "Static visualization of agent paths"
            elseif occursin("convergence", file)
                "Convergence plot of the inference (may show placeholder if ELBO tracking unavailable)"
            else
                file
            end
            write(io, "- `$(file)` - $desc\n")
        end
        write(io, "\n")
        
        # Heatmaps
        write(io, "### Environment Heatmaps\n")
        heatmap_files = filter(file -> endswith(file, ".png"), readdir(subdirs["heatmaps"]))
        for file in heatmap_files
            desc = if occursin("obstacle_distance", file)
                "Heatmap of distances to obstacles"
            elseif occursin("door_environment", file)
                "Distance field visualization for door environment"
            elseif occursin("wall_environment", file)
                "Distance field visualization for wall environment"
            elseif occursin("combined_environment", file)
                "Distance field visualization for combined environment"
            else
                file
            end
            write(io, "- `$(file)` - $desc\n")
        end
        write(io, "\n")
        
        # Data files
        write(io, "### Data Files\n")
        data_files = filter(file -> endswith(file, ".csv") || endswith(file, ".log"), readdir(subdirs["data"]))
        for file in data_files
            desc = if occursin("paths.csv", file)
                "Raw path data (agent positions over time)"
            elseif occursin("controls.csv", file)
                "Raw control signals (agent control inputs)"
            elseif occursin("uncertainties.csv", file)
                "Path uncertainties (variance in agent positions)"
            elseif occursin("experiment.log", file)
                "Detailed log of experiment execution"
            elseif occursin("convergence_metrics.csv", file)
                "ELBO convergence metrics from inference"
            else
                file
            end
            write(io, "- `$(file)` - $desc\n")
        end
        write(io, "\n")
        
        # Experiment description
        write(io, "## Experiment Setup\n\n")
        write(io, "The experiments demonstrate multi-agent trajectory planning in three environments:\n")
        write(io, "1. **Door environment**: Two parallel walls with a gap between them\n")
        write(io, "2. **Wall environment**: A single wall obstacle in the center\n")
        write(io, "3. **Combined environment**: A combination of walls and obstacles\n\n")
        write(io, "Each experiment is run with 4 agents that need to navigate from their starting positions\n")
        write(io, "to their target positions while avoiding obstacles and other agents.\n\n")
        
        # Implementation details
        write(io, "## Implementation Details\n\n")
        write(io, "The trajectory planning is implemented using probabilistic inference with the RxInfer.jl framework.\n")
        write(io, "The agents follow a linear dynamical model with control inputs, and constraints are enforced\n")
        write(io, "through observations in the probabilistic model.\n\n")
        write(io, "For more details, see the `DOCUMENTATION.md` file in the project root.\n")
    end
    
    println("README created at $readme_path")
end

end # module 