#!/usr/bin/env julia

"""
RxInfer.jl Environment Setup Script

This script provides robust environment setup for RxInfer.jl simulations:
- Package installation with error handling
- Environment validation
- Dependency resolution

Usage: julia setup_environment.jl [--verbose] [--validate-only]
"""

using Pkg
using Printf
using Dates
using Logging

# Setup script configuration
const SETUP_VERSION = "1.0.0"

# Core required packages for RxInfer.jl simulations
const CORE_PACKAGES = [
    "RxInfer",
    "Distributions",
    "LinearAlgebra",
    "Random",
    "Statistics"
]

# Utility packages
const UTIL_PACKAGES = [
    "Dates",
    "Printf",
    "Logging",
    "DelimitedFiles",
    "JSON3"
]

# Visualization packages (optional)
const VIZ_PACKAGES = [
    "Plots",
    "StatsPlots"
]

const PACKAGE_GROUPS = Dict(
    "core" => CORE_PACKAGES,
    "util" => UTIL_PACKAGES,
    "viz" => VIZ_PACKAGES
)

# Configuration
struct SetupConfig
    verbose::Bool
    validate_only::Bool
    install_viz::Bool
    max_retries::Int
    timeout_seconds::Int
end

function parse_args()
    """Parse command line arguments."""
    args = ARGS
    verbose = "--verbose" in args || "-v" in args
    validate_only = "--validate-only" in args || "--validate" in args
    install_viz = "--with-viz" in args || "--viz" in args
    
    return SetupConfig(
        verbose,
        validate_only,
        install_viz,
        3,    # max_retries
        300   # timeout_seconds
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
        return false
    end
end

function setup_project_environment(config::SetupConfig)
    """Setup project environment."""
    @info "Setting up project environment..."
    
    project_dir = @__DIR__
    @info "Using project directory: $project_dir"
    
    try
        Pkg.activate(project_dir)
        @info "✅ Activated project environment"
    catch e
        @error "❌ Failed to activate project environment: $e"
        return false
    end
    
    return true
end

function validate_package(package_name::String, config::SetupConfig)
    """Validate that a package can be loaded."""
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
    """Install a package with retry logic."""
    
    for attempt in 1:config.max_retries
        try
            @info "Installing $package_name (attempt $attempt/$(config.max_retries))..."
            
            Pkg.add(package_name)
            
            if validate_package(package_name, config)
                @info "✅ Successfully installed $package_name"
                return true
            else
                @warn "Package installed but validation failed"
                if attempt < config.max_retries
                    sleep(2)
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
    """Install a group of packages."""
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
    
    @info "✅ $group_name: $(length(successful))/$(length(packages)) successful"
    
    if !isempty(failed)
        @warn "⚠️  Failed packages: $(join(failed, ", "))"
    end
    
    return length(failed) == 0, successful, failed
end

function validate_environment(config::SetupConfig)
    """Validate the environment."""
    @info "Validating environment..."
    
    validation_results = Dict{String, Bool}()
    
    # Check core packages
    for package in CORE_PACKAGES
        validation_results[package] = validate_package(package, config)
    end
    
    # Check utility packages
    for package in UTIL_PACKAGES
        validation_results[package] = validate_package(package, config)
    end
    
    # Summary
    total = length(validation_results)
    valid_count = count(values(validation_results))
    
    @info "Validation summary: $valid_count/$total packages valid"
    
    if valid_count == total
        @info "✅ Environment validation passed"
        return true
    else
        failed = [pkg for (pkg, valid) in validation_results if !valid]
        @warn "⚠️  Failed packages: $(join(failed, ", "))"
        return false
    end
end

function run_setup(config::SetupConfig)
    """Run the complete setup process."""
    
    @info "="^60
    @info "RxInfer.jl Environment Setup v$SETUP_VERSION"
    @info "="^60
    @info "Julia version: $(VERSION)"
    @info "Date: $(now())"
    @info ""
    
    # Pre-flight checks
    if !check_julia_version()
        return false
    end
    
    if !setup_project_environment(config)
        return false
    end
    
    if config.validate_only
        @info "🔍 Running validation only..."
        return validate_environment(config)
    end
    
    @info "🚀 Starting package installation..."
    
    overall_success = true
    
    # Install core packages (required)
    success, _, failed = install_package_group("core", CORE_PACKAGES, config)
    if !success
        overall_success = false
    end
    
    # Install utility packages
    success, _, failed = install_package_group("util", UTIL_PACKAGES, config)
    if !success
        @warn "Some utility packages failed - non-critical"
    end
    
    # Install visualization packages if requested
    if config.install_viz
        success, _, failed = install_package_group("viz", VIZ_PACKAGES, config)
        if !success
            @warn "Visualization packages failed - non-critical"
        end
    end
    
    # Final validation
    @info ""
    @info "🔍 Running final validation..."
    validation_success = validate_environment(config)
    
    @info ""
    if overall_success && validation_success
        @info "🎉 RxInfer.jl environment setup completed successfully!"
        @info "You can now run RxInfer.jl simulations."
    else
        @warn "⚠️  Environment setup completed with some issues."
        @warn "Core functionality should still work."
    end
    
    return overall_success && validation_success
end

function main()
    """Main entry point."""
    try
        config = parse_args()
        setup_logging(config)
        success = run_setup(config)
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


