#!/usr/bin/env julia

"""
Multiagent_GNN_RxInfer.jl

This script demonstrates the GNN-RxInfer pipeline for Multi-agent Trajectory Planning:
1. Runs the original RxInfer implementation
2. Creates a copy with the GNN-rendered configuration
3. Validates the integration between GNN and RxInfer
"""

using Pkg
using Dates
using TOML
using Logging
using FileIO
using Statistics

# Set up logging
log_dir = joinpath(dirname(@__DIR__), "output", "logs")
mkpath(log_dir)
log_file = joinpath(log_dir, "multiagent_gnn_rxinfer_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
io = open(log_file, "w+")
logger = SimpleLogger(io)
global_logger(logger)

# Define paths
REPO_ROOT = dirname(dirname(@__DIR__))
RXINFER_EXAMPLES_DIR = joinpath(@__DIR__, "RxInferExamples.jl")
ORIGINAL_SCRIPT_PATH = joinpath(RXINFER_EXAMPLES_DIR, "scripts", "Advanced Examples", "Multi-agent Trajectory Planning", "Multi-agent Trajectory Planning.jl")
ORIGINAL_DIR = dirname(ORIGINAL_SCRIPT_PATH)
TARGET_DIR = joinpath(@__DIR__, "multiagent_trajectory_planning")
GNN_OUTPUT_DIR = joinpath(REPO_ROOT, "output")
GNN_CONFIG_PATH = joinpath(GNN_OUTPUT_DIR, "multiagent_trajectory_planning_config.toml")

function check_paths_exist()
    @info "Checking required paths..."
    
    if !isfile(ORIGINAL_SCRIPT_PATH)
        @error "Original script not found at: $ORIGINAL_SCRIPT_PATH"
        return false
    end
    
    if !isfile(GNN_CONFIG_PATH)
        @error "GNN-rendered config not found at: $GNN_CONFIG_PATH"
        @info "Looking for alternative config files in output directory..."
        config_files = filter(f -> endswith(f, ".toml"), readdir(GNN_OUTPUT_DIR))
        if !isempty(config_files)
            @info "Found potential config files: $config_files"
            @info "Please specify which one to use by updating the GNN_CONFIG_PATH variable"
        else
            @info "No TOML config files found in output directory"
        end
        return false
    end
    
    return true
end

function run_original_script()
    @info "Running original Multi-agent Trajectory Planning script..."
    
    # Change to the directory containing the script (for relative paths in the script)
    original_dir = pwd()
    cd(dirname(ORIGINAL_SCRIPT_PATH))
    
    try
        # Capture output from the script execution
        output = read(`julia "$(basename(ORIGINAL_SCRIPT_PATH))"`, String)
        @info "Original script executed successfully"
        @debug "Script output: $output"
    catch e
        @error "Failed to run original script" exception=(e, catch_backtrace())
        cd(original_dir)
        return false
    end
    
    cd(original_dir)
    return true
end

function validate_gnn_config()
    @info "Validating GNN-generated configuration..."
    
    try
        config = TOML.parsefile(GNN_CONFIG_PATH)
        @info "GNN config loaded successfully with $(length(keys(config))) top-level keys"
        
        # Check for essential configuration sections
        essential_sections = ["dimensions", "initial_parameters", "constraints", "simulation"]
        missing_sections = filter(s -> !haskey(config, s), essential_sections)
        
        if !isempty(missing_sections)
            @warn "GNN config is missing expected sections: $missing_sections"
            return false
        end
        
        return true
    catch e
        @error "Failed to validate GNN config" exception=(e, catch_backtrace())
        return false
    end
end

function copy_and_modify_scripts()
    @info "Copying scripts and replacing configuration..."
    
    # Create target directory if it doesn't exist
    mkpath(TARGET_DIR)
    
    try
        # Copy all files except config.toml
        for file in readdir(ORIGINAL_DIR)
            src_path = joinpath(ORIGINAL_DIR, file)
            dst_path = joinpath(TARGET_DIR, file)
            
            if file != "config.toml" && isfile(src_path)
                cp(src_path, dst_path, force=true)
                @info "Copied $file to target directory"
            end
        end
        
        # Copy the GNN-generated config
        cp(GNN_CONFIG_PATH, joinpath(TARGET_DIR, "config.toml"), force=true)
        @info "Replaced config.toml with GNN-generated version"
        
        return true
    catch e
        @error "Failed to copy and modify scripts" exception=(e, catch_backtrace())
        return false
    end
end

function run_modified_script()
    @info "Running modified script with GNN-generated config..."
    
    # Change to the target directory
    original_dir = pwd()
    cd(TARGET_DIR)
    
    try
        # Find the main script file (assuming it's the same name as the original)
        script_name = basename(ORIGINAL_SCRIPT_PATH)
        
        # Execute the script
        output = read(`julia "$script_name"`, String)
        @info "Modified script executed successfully"
        @debug "Script output: $output"
    catch e
        @error "Failed to run modified script" exception=(e, catch_backtrace())
        cd(original_dir)
        return false
    end
    
    cd(original_dir)
    return true
end

function main()
    @info "Starting Multiagent_GNN_RxInfer.jl script"
    
    # Check if required paths exist
    if !check_paths_exist()
        @error "Required paths check failed. Exiting."
        return 1
    end
    
    # Run the original script to establish baseline
    if !run_original_script()
        @error "Original script execution failed. Exiting."
        return 1
    end
    
    # Validate the GNN-generated config
    if !validate_gnn_config()
        @error "GNN config validation failed. Exiting."
        return 1
    end
    
    # Copy and modify the scripts
    if !copy_and_modify_scripts()
        @error "Failed to prepare modified scripts. Exiting."
        return 1
    end
    
    # Run the modified script
    if !run_modified_script()
        @error "Modified script execution failed. Exiting."
        return 1
    end
    
    @info "Multiagent_GNN_RxInfer.jl completed successfully"
    @info "Original implementation: $ORIGINAL_DIR"
    @info "GNN-based implementation: $TARGET_DIR"
    
    return 0
end

# Execute the main function
exit_code = main()
close(io)  # Close the log file
exit(exit_code) 