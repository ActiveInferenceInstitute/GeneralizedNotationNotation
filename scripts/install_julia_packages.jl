#!/usr/bin/env julia
"""
Julia Package Installation Script for GNN Pipeline

This script installs the required Julia packages for the GNN processing pipeline.
It ensures that RxInfer.jl, ActiveInference.jl, and other dependencies are available.
"""

using Pkg

println("🔧 Installing Julia packages for GNN Pipeline...")

# Add required packages
packages_to_add = [
    "RxInfer",
    "ActiveInference", 
    "Distributions",
    "LinearAlgebra",
    "Plots",
    "Random",
    "StatsBase",
    "SpecialFunctions",
    "Optim",
    "ForwardDiff",
    "Zygote",
    "Flux",
    "MLJ",
    "DataFrames",
    "CSV",
    "JSON",
    "YAML",
    "ArgParse",
    "Logging",
    "ProgressMeter"
]

println("📦 Adding packages to Julia environment...")

for pkg in packages_to_add
    try
        println("  Adding $pkg...")
        Pkg.add(pkg)
        println("  ✅ $pkg added successfully")
    catch e
        println("  ⚠️  Warning: Failed to add $pkg: $e")
    end
end

println("🔍 Checking package availability...")

# Test package availability
available_packages = []
missing_packages = []

for pkg in packages_to_add
    try
        eval(Meta.parse("using $pkg"))
        push!(available_packages, pkg)
        println("  ✅ $pkg is available")
    catch e
        push!(missing_packages, pkg)
        println("  ❌ $pkg is not available: $e")
    end
end

println("\n📊 Installation Summary:")
println("  Available packages: $(length(available_packages))")
println("  Missing packages: $(length(missing_packages))")

if !isempty(missing_packages)
    println("\n⚠️  Missing packages:")
    for pkg in missing_packages
        println("    - $pkg")
    end
    println("\n💡 Try running: julia --project=. -e 'using Pkg; Pkg.instantiate()'")
else
    println("\n🎉 All packages installed successfully!")
end

println("\n🔧 Julia package installation completed.")

