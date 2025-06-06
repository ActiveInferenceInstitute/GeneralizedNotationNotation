# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Advanced Examples/Multi-agent Trajectory Planning/Multi-agent Trajectory Planning.ipynb
# by notebooks_to_scripts.jl at 2025-06-03T10:14:28.748
#
# Source notebook: Multi-agent Trajectory Planning.ipynb

# Activate the local project environment
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Set GR backend for headless operation
ENV["GKSwstype"] = "100"

# Load the module and run the experiments
println("Loading TrajectoryPlanning module...")
include("TrajectoryPlanning.jl")
using .TrajectoryPlanning
using Plots
using Dates

# Set plot backend
gr()

# Create function to set up directory structure
function create_output_subdirectories(base_dir)
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

# Load the advanced visualization function
include("visualize_results.jl")

# Run all experiments and get the output directory
println("Starting Multi-agent Trajectory Planning experiments...")
output_dir = TrajectoryPlanning.run_all_experiments()

# Create organized subdirectory structure
subdirs = create_output_subdirectories(output_dir)
println("Created organized directory structure in: $output_dir")

# Move files to appropriate subdirectories
println("Organizing output files...")
try
    # Move animations (GIFs) to animations subdirectory
    for file in readdir(output_dir)
        if endswith(file, ".gif") && isfile(joinpath(output_dir, file))
            mv(joinpath(output_dir, file), joinpath(subdirs["animations"], file), force=true)
        elseif occursin("heatmap", file) && endswith(file, ".png") && isfile(joinpath(output_dir, file))
            mv(joinpath(output_dir, file), joinpath(subdirs["heatmaps"], file), force=true)
        elseif endswith(file, ".png") && isfile(joinpath(output_dir, file))
            mv(joinpath(output_dir, file), joinpath(subdirs["visualizations"], file), force=true)
        elseif (endswith(file, ".csv") || endswith(file, ".log")) && isfile(joinpath(output_dir, file))
            mv(joinpath(output_dir, file), joinpath(subdirs["data"], file), force=true)
        end
    end
    println("Files organized successfully.")
catch e
    println("Error organizing files: $e")
end

println("All experiments completed. Results saved to: $output_dir")

# Run advanced visualizations on the results
println("\nGenerating advanced visualizations...")
visualize_results(output_dir, subdirs)
println("Advanced visualizations completed.") 