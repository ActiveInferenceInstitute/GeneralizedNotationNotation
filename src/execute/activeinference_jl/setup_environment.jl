#!/usr/bin/env julia

"""
Comprehensive ActiveInference.jl Environment Setup Script

This script provides robust, reproducible environment setup for ActiveInference.jl analysis:
- Comprehensive package installation with error handling
- Environment validation and health checks  
- Dependency resolution and conflict management
- Version compatibility checking
- Fallback mechanisms for failed installations
- Environment reporting and diagnostics

Usage: julia setup_environment.jl [--verbose] [--force-reinstall] [--validate-only]
"""

using Pkg
using Printf
using Dates
using Logging
using TOML

# Setup script configuration
const SETUP_VERSION = "2.0.0"
const LOG_LEVEL = Logging.Info

# Core required packages (always needed)
const CORE_PACKAGES = [
    "ActiveInference",
    "Distributions", 
    "LinearAlgebra",
    "Random",
    "Statistics",
    "Dates",
    "Printf",
    "Logging",
    "DelimitedFiles"
]

# Statistical analysis packages
const STATS_PACKAGES = [
    "StatsBase",
    "HypothesisTests",
    "Bootstrap", 
    "TimeSeries",
    "MultipleTesting"
]

# Visualization packages
const VIZ_PACKAGES = [
    "Plots",
    "PlotlyJS", 
    "StatsPlots",
    "GraphPlot",
    "NetworkLayout",
    "LightGraphs",
    "Colors",
    "ColorSchemes"
]

# Data handling packages
const DATA_PACKAGES = [
    "DataFrames",
    "CSV",
    "MAT",
    "JSON",
    "JSON3"
]

# Advanced analysis packages
const ADVANCED_PACKAGES = [
    "Optim",
    "PDMats",
    "Clustering",
    "Combinatorics",
    "FFTW"
]

# Package groups for selective installation
const PACKAGE_GROUPS = Dict(
    "core" => CORE_PACKAGES,
    "stats" => STATS_PACKAGES,
    "viz" => VIZ_PACKAGES,
    "data" => DATA_PACKAGES,
    "advanced" => ADVANCED_PACKAGES
)

const ALL_PACKAGES = vcat(CORE_PACKAGES, STATS_PACKAGES, VIZ_PACKAGES, DATA_PACKAGES, ADVANCED_PACKAGES)

# Configuration
struct SetupConfig
    verbose::Bool
    force_reinstall::Bool
    validate_only::Bool
    install_groups::Vector{String}
    max_retries::Int
    timeout_seconds::Int
end

function parse_args()
    """Parse command line arguments."""
    config = SetupConfig(
        false,  # verbose
        false,  # force_reinstall  
        false,  # validate_only
        ["core", "stats", "viz", "data", "advanced"],  # install_groups
        3,      # max_retries
        300     # timeout_seconds
    )
    
    args = ARGS
    verbose = "--verbose" in args || "-v" in args
    force_reinstall = "--force-reinstall" in args || "--force" in args
    validate_only = "--validate-only" in args || "--validate" in args
    
    groups = ["core", "stats", "viz", "data", "advanced"]
    if "--core-only" in args
        groups = ["core"]
    elseif "--minimal" in args
        groups = ["core", "stats"]
    end
    
    return SetupConfig(
        verbose,
        force_reinstall,
        validate_only,
        groups,
        config.max_retries,
        config.timeout_seconds
    )
end

function setup_logging(config::SetupConfig)
    """Setup logging configuration."""
    log_level = config.verbose ? Logging.Debug : Logging.Info
    global_logger(ConsoleLogger(stderr, log_level))
end

function check_julia_version()
    """Check if Julia version is compatible."""
    @info "Checking Julia version compatibility..."
    
    min_version = v"1.6.0"
    current_version = VERSION
    
    if current_version >= min_version
        @info "✅ Julia version $current_version is compatible (minimum: $min_version)"
        return true
    else
        @error "❌ Julia version $current_version is too old (minimum: $min_version)"
        @error "Please upgrade Julia to continue"
        return false
    end
end

function setup_project_environment(config::SetupConfig)
    """Setup project environment with proper Project.toml."""
    @info "Setting up project environment..."
    
    # Ensure we're in project mode
    project_dir = @__DIR__
    @info "Using project directory: $project_dir"
    
    # Activate project environment
    try
        Pkg.activate(project_dir)
        @info "✅ Activated project environment"
    catch e
        @error "❌ Failed to activate project environment: $e"
        return false
    end
    
    # Check for Project.toml
    project_toml = joinpath(project_dir, "Project.toml")
    if !isfile(project_toml)
        @warn "Project.toml not found, will be created automatically"
    else
        @info "✅ Found existing Project.toml"
    end
    
    return true
end

function validate_package_installation(package_name::String, config::SetupConfig)
    """Validate that a package is properly installed and can be loaded."""
    try
        @debug "Validating package: $package_name"
        eval(Meta.parse("using $package_name"))
        @debug "✅ Package $package_name loaded successfully"
        return true
    catch e
        @warn "❌ Failed to load package $package_name: $e"
        return false
    end
end

function install_package_with_retry(package_name::String, config::SetupConfig)
    """Install a package with retry logic and error handling."""
    
    for attempt in 1:config.max_retries
        try
            @info "Installing $package_name (attempt $attempt/$(config.max_retries))..."
            
            if config.force_reinstall
                @debug "Force reinstalling $package_name"
                # Remove first if force reinstall
                try
                    Pkg.rm(package_name)
                catch
                    # Ignore errors if package wasn't installed
                end
            end
            
            # Install package with timeout
            install_task = @async Pkg.add(package_name)
            if Base.istaskstarted(install_task)
                # Wait with timeout
                sleep_counter = 0
                while !Base.istaskdone(install_task) && sleep_counter < config.timeout_seconds
                    sleep(1)
                    sleep_counter += 1
                end
                
                if !Base.istaskdone(install_task)
                    @warn "Installation of $package_name timed out after $(config.timeout_seconds) seconds"
                    continue
                end
            end
            
            # Validate installation
            if validate_package_installation(package_name, config)
                @info "✅ Successfully installed and validated $package_name"
                return true
            else
                @warn "❌ Package $package_name installed but failed validation"
                if attempt < config.max_retries
                    @info "Retrying installation..."
                    sleep(2)  # Brief pause before retry
                end
            end
            
        catch e
            @error "❌ Error installing $package_name (attempt $attempt): $e"
            if attempt < config.max_retries
                @info "Retrying in 2 seconds..."
                sleep(2)
            end
        end
    end
    
    @error "❌ Failed to install $package_name after $(config.max_retries) attempts"
    return false
end

function install_package_group(group_name::String, packages::Vector{String}, config::SetupConfig)
    """Install a group of packages with progress reporting."""
    @info "Installing $group_name packages ($(length(packages)) packages)..."
    
    successful = String[]
    failed = String[]
    
    for (i, package) in enumerate(packages)
        @info "[$i/$(length(packages))] Installing $package..."
        
        if install_package_with_retry(package, config)
            push!(successful, package)
        else
            push!(failed, package)
        end
    end
    
    @info "✅ $group_name installation complete: $(length(successful))/$(length(packages)) successful"
    
    if !isempty(failed)
        @warn "⚠️  Failed packages in $group_name: $(join(failed, ", "))"
    end
    
    return length(failed) == 0, successful, failed
end

function validate_environment(config::SetupConfig)
    """Comprehensive environment validation."""
    @info "Validating environment..."
    
    validation_results = Dict{String, Bool}()
    
    # Check all package groups
    for (group_name, packages) in PACKAGE_GROUPS
        if group_name in config.install_groups
            @info "Validating $group_name packages..."
            group_valid = true
            
            for package in packages
                is_valid = validate_package_installation(package, config)
                validation_results[package] = is_valid
                if !is_valid
                    group_valid = false
                end
            end
            
            if group_valid
                @info "✅ All $group_name packages are valid"
            else
                @warn "⚠️  Some $group_name packages failed validation"
            end
        end
    end
    
    # Summary
    total_packages = length(validation_results)
    valid_packages = count(values(validation_results))
    
    @info "Environment validation summary:"
    @info "  Total packages checked: $total_packages"
    @info "  Valid packages: $valid_packages"
    @info "  Failed packages: $(total_packages - valid_packages)"
    
    if valid_packages == total_packages
        @info "✅ Environment validation passed completely"
        return true
    else
        failed_packages = [pkg for (pkg, valid) in validation_results if !valid]
        @warn "⚠️  Environment validation incomplete"
        @warn "Failed packages: $(join(failed_packages, ", "))"
        return false
    end
end

function generate_environment_report(config::SetupConfig)
    """Generate comprehensive environment report."""
    @info "Generating environment report..."
    
    report = Dict{String, Any}()
    report["timestamp"] = Dates.now()
    report["julia_version"] = string(VERSION)
    report["setup_version"] = SETUP_VERSION
    report["config"] = Dict(
        "verbose" => config.verbose,
        "force_reinstall" => config.force_reinstall,
        "install_groups" => config.install_groups
    )
    
    # Package status
    package_status = Dict{String, Any}()
    for (group_name, packages) in PACKAGE_GROUPS
        if group_name in config.install_groups
            group_status = Dict{String, Bool}()
            for package in packages
                group_status[package] = validate_package_installation(package, config)
            end
            package_status[group_name] = group_status
        end
    end
    report["package_status"] = package_status
    
    # Save report
    report_file = joinpath(@__DIR__, "environment_report.json")
    try
        open(report_file, "w") do f
            JSON3.pretty(f, report)
        end
        @info "✅ Environment report saved to: $report_file"
    catch e
        @warn "⚠️  Could not save environment report: $e"
    end
    
    return report
end

function run_setup_process(config::SetupConfig)
    """Run the complete setup process."""
    
    @info "="^70
    @info "ActiveInference.jl Environment Setup v$SETUP_VERSION"
    @info "="^70
    @info "Julia version: $(VERSION)"
    @info "Date: $(now())"
    @info "Configuration: $(config.install_groups) groups"
    if config.validate_only
        @info "Mode: Validation only (no installation)"
    elseif config.force_reinstall
        @info "Mode: Force reinstall all packages"
    else
        @info "Mode: Standard installation"
    end
    @info ""
    
    # Pre-flight checks
    if !check_julia_version()
        return false
    end
    
    if !setup_project_environment(config)
        return false
    end
    
    # Environment setup or validation
    if config.validate_only
        @info "🔍 Running validation only..."
        return validate_environment(config)
    else
        @info "🚀 Starting package installation..."
        
        overall_success = true
        installation_summary = Dict{String, Any}()
        
        # Install package groups
        for group_name in config.install_groups
            if haskey(PACKAGE_GROUPS, group_name)
                packages = PACKAGE_GROUPS[group_name]
                group_success, successful, failed = install_package_group(group_name, packages, config)
                
                installation_summary[group_name] = Dict(
                    "success" => group_success,
                    "successful_packages" => successful,
                    "failed_packages" => failed
                )
                
                if !group_success
                    overall_success = false
                end
            else
                @warn "Unknown package group: $group_name"
            end
        end
        
        @info ""
        @info "📦 Installation Summary:"
        for (group, summary) in installation_summary
            success_count = length(summary["successful_packages"])
            total_count = success_count + length(summary["failed_packages"])
            status = summary["success"] ? "✅" : "⚠️"
            @info "  $status $group: $success_count/$total_count packages"
        end
        
        # Final validation
        @info ""
        @info "🔍 Running final validation..."
        validation_success = validate_environment(config)
        
        # Generate report
        generate_environment_report(config)
        
        final_success = overall_success && validation_success
        
        @info ""
        if final_success
            @info "🎉 Environment setup completed successfully!"
            @info "ActiveInference.jl is ready for use."
        else
            @warn "⚠️  Environment setup completed with issues."
            @warn "Some packages may not be available for advanced analysis."
            @info "Basic functionality should still work."
        end
        
        return final_success
    end
end

function main()
    """Main entry point."""
    try
        config = parse_args()
        setup_logging(config)
        success = run_setup_process(config)
        exit(success ? 0 : 1)
    catch e
        @error "Fatal error during setup: $e"
        exit(1)
    end
end

# Run main if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 