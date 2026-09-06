#!/usr/bin/env julia

"""
ActiveInference.jl Export Enhancement Module

This module provides enhanced export capabilities for ActiveInference.jl simulation results.
Supports multiple formats: MATLAB, R, Python, JSON, LaTeX, Excel, SQLite.
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Dates

# Ensure required packages are installed
for pkg in ["MAT", "JSON3", "DataFrames", "CSV"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using MAT
using JSON3
using DataFrames
using CSV

# ====================================
# MATLAB EXPORT
# ====================================

"""Export data to MATLAB .mat format."""
function export_to_matlab(data_dict::Dict{String, Any}, output_dir::String)
    matlab_dir = joinpath(output_dir, "MATLAB_exports")
    mkpath(matlab_dir)
    
    try
        # Convert data to MATLAB-compatible format
        matlab_dict = Dict{String, Any}()
        
        for (key, value) in data_dict
            # Clean key name for MATLAB variable naming
            clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
            clean_key = replace(clean_key, r"^([0-9])" => s"var_\1")  # Can't start with number
            
            if isa(value, AbstractMatrix)
                matlab_dict[clean_key] = convert(Matrix{Float64}, value)
            elseif isa(value, AbstractVector)
                matlab_dict[clean_key] = convert(Vector{Float64}, value)
            elseif isa(value, Number)
                matlab_dict[clean_key] = Float64(value)
            else
                matlab_dict[clean_key] = string(value)
            end
        end
        
        # Add metadata
        matlab_dict["export_info"] = Dict(
            "exported_from" => "ActiveInference.jl",
            "export_date" => string(now()),
            "variable_count" => length(matlab_dict) - 1
        )
        
        output_path = joinpath(matlab_dir, "activeinference_data.mat")
        matwrite(output_path, matlab_dict)
        println("📊 MATLAB export: $output_path")
        return true
        
    catch e
        @warn "MATLAB export failed: $e"
        return false
    end
end

# ====================================
# R EXPORT
# ====================================

"""Export data to R-compatible CSV with metadata."""
function export_to_r_csv(data_dict::Dict{String, Any}, output_dir::String)
    r_dir = joinpath(output_dir, "R_exports")
    mkpath(r_dir)
    
    try
        # Create R script to load data
        r_script = """
# ActiveInference.jl R Data Import Script
# Generated: $(now())

library(data.table)

# Function to load ActiveInference.jl data
load_activeinference_data <- function(data_dir = "$(r_dir)") {
  data_list <- list()
  
"""
        
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                csv_file = joinpath(r_dir, "$(clean_key).csv")
                
                # Write CSV with proper headers
                if size(value, 2) == 2
                    headers = ["time", clean_key]
                else
                    headers = ["time"; [clean_key * "_" * string(i) for i in 1:size(value,2)-1]]
                end
                
                open(csv_file, "w") do f
                    println(f, "# ActiveInference.jl Data Export")
                    println(f, "# Variable: $key")
                    println(f, "# Exported: $(now())")
                    println(f, join(headers, ","))
                    for i in 1:size(value, 1)
                        println(f, join(value[i, :], ","))
                    end
                end
                
                # Add to R script
                r_script *= """
  # Load $key
  $(clean_key) <- fread(file.path(data_dir, "$(clean_key).csv"), skip = 3)
  data_list[["$(clean_key)"]] <- $(clean_key)
  
"""
            end
        end
        
        r_script *= """
  return(data_list)
}

# Load all data
activeinference_data <- load_activeinference_data()

# Print summary
cat("Loaded", length(activeinference_data), "datasets\\n")
for(name in names(activeinference_data)) {
  cat("- ", name, ": ", nrow(activeinference_data[[name]]), " rows\\n")
}
"""
        
        # Write R script
        open(joinpath(r_dir, "load_data.R"), "w") do f
            write(f, r_script)
        end
        
        println("📈 R export: $(r_dir)")
        return true
        
    catch e
        @warn "R export failed: $e"
        return false
    end
end

# ====================================
# PYTHON EXPORT
# ====================================

"""Export data to Python-compatible formats."""
function export_to_python(data_dict::Dict{String, Any}, output_dir::String)
    python_dir = joinpath(output_dir, "Python_exports")
    mkpath(python_dir)
    
    try
        # Create Python script to load data
        python_script = """
#!/usr/bin/env python3
'''
ActiveInference.jl Python Data Import Module
Generated: $(now())
'''

import numpy as np
import pandas as pd
import json
from pathlib import Path

def load_activeinference_data(data_dir="$(python_dir)"):
    \"\"\"Load ActiveInference.jl exported data.\"\"\"
    data_dict = {}
    data_dir = Path(data_dir)
    
"""
        
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                
                # Export as NumPy array
                npy_file = joinpath(python_dir, "$(clean_key).npy")
                # Manual NPY format (simplified)
                open(npy_file, "w") do f
                    println(f, "# NumPy array: $key")
                    println(f, "# Shape: $(size(value))")
                    println(f, "# Data:")
                    for i in 1:size(value, 1)
                        println(f, join(value[i, :], " "))
                    end
                end
                
                # Export as CSV for pandas
                csv_file = joinpath(python_dir, "$(clean_key).csv")
                headers = size(value, 2) == 2 ? ["time", clean_key] : 
                         ["time"; [clean_key * "_" * string(i) for i in 1:size(value,2)-1]]
                
                open(csv_file, "w") do f
                    println(f, join(headers, ","))
                    for i in 1:size(value, 1)
                        println(f, join(value[i, :], ","))
                    end
                end
                
                python_script *= """
    # Load $key
    try:
        $(clean_key) = pd.read_csv(data_dir / "$(clean_key).csv")
        data_dict["$(clean_key)"] = $(clean_key)
    except Exception as e:
        print(f"Failed to load $(clean_key): {e}")
    
"""
            end
        end
        
        python_script *= """
    return data_dict

def summary_statistics(data_dict):
    \"\"\"Compute summary statistics for all datasets.\"\"\"
    for name, df in data_dict.items():
        print(f"\\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            print(f"  Numeric summary:")
            print(df.describe().to_string(max_cols=5))

if __name__ == "__main__":
    # Load data
    data = load_activeinference_data()
    print(f"Loaded {len(data)} datasets")
    
    # Show summary
    summary_statistics(data)
"""
        
        open(joinpath(python_dir, "load_data.py"), "w") do f
            write(f, python_script)
        end
        
        # Create requirements.txt
        open(joinpath(python_dir, "requirements.txt"), "w") do f
            println(f, "numpy>=1.19.0")
            println(f, "pandas>=1.3.0")
            println(f, "matplotlib>=3.3.0")
            println(f, "scipy>=1.7.0")
        end
        
        println("🐍 Python export: $(python_dir)")
        return true
        
    catch e
        @warn "Python export failed: $e"
        return false
    end
end

# ====================================
# JSON SCHEMA EXPORT
# ====================================

"""Export data as JSON with comprehensive schema."""
function export_to_json_schema(data_dict::Dict{String, Any}, output_path::String)
    try
        # Create comprehensive JSON structure
        json_export = Dict(
            "metadata" => Dict(
                "source" => "ActiveInference.jl",
                "export_date" => string(now()),
                "version" => "1.0.0",
                "description" => "ActiveInference.jl simulation results with comprehensive metadata"
            ),
            "schema" => Dict(
                "datasets" => Dict(),
                "statistics" => Dict(),
                "relationships" => []
            ),
            "data" => Dict()
        )
        
        # Process each dataset
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                
                # Dataset schema
                json_export["schema"]["datasets"][clean_key] = Dict(
                    "original_name" => key,
                    "type" => "matrix",
                    "dimensions" => size(value),
                    "data_type" => eltype(value),
                    "description" => "ActiveInference.jl simulation data"
                )
                
                # Statistics
                json_export["schema"]["statistics"][clean_key] = Dict(
                    "mean" => mean(value),
                    "std" => std(value),
                    "min" => minimum(value),
                    "max" => maximum(value),
                    "median" => median(value)
                )
                
                # Data (convert to nested arrays for JSON)
                json_export["data"][clean_key] = [value[i, :] for i in 1:size(value, 1)]
            end
        end
        
        # Write JSON file
        open(output_path, "w") do f
            write(f, JSON3.write(json_export, 2))
        end
        
        println("📄 JSON export: $output_path")
        return true
        
    catch e
        @warn "JSON export failed: $e"
        return false
    end
end

# ====================================
# LATEX EXPORT
# ====================================

"""Export data to LaTeX format with tables and plots."""
function export_to_latex(data_dict::Dict{String, Any}, output_dir::String)
    latex_dir = joinpath(output_dir, "LaTeX_exports")
    mkpath(latex_dir)
    
    try
        # Create main LaTeX document
        latex_content = """
\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{graphicx}
\\usepackage{amsmath}
\\usepackage{geometry}
\\geometry{margin=1in}

\\title{ActiveInference.jl Simulation Results}
\\author{Generated by ActiveInference.jl}
\\date{$(now())}

\\begin{document}

\\maketitle

\\section{Simulation Overview}
This document contains the results from ActiveInference.jl simulation analysis.

\\section{Data Summary}
"""
        
        # Add summary statistics
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                
                latex_content *= """
\\subsection{$(key)}
\\begin{itemize}
\\item Dimensions: $(size(value, 1)) × $(size(value, 2))
\\item Mean: $(round(mean(value), digits=4))
\\item Standard Deviation: $(round(std(value), digits=4))
\\item Range: [$(round(minimum(value), digits=4)), $(round(maximum(value), digits=4))]
\\end{itemize}

"""
            end
        end
        
        latex_content *= """
\\section{Data Tables}

"""
        
        # Add data tables
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0 && size(value, 1) <= 20
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                
                latex_content *= """
\\subsection{$(key) Data}
\\begin{table}[h]
\\centering
\\begin{tabular}{$(repeat("c", size(value, 2)))}
\\toprule
"""
                
                # Headers
                if size(value, 2) == 2
                    latex_content *= "Time & Value \\\\\n"
                else
                    latex_content *= "Time & " * join(["Col$i" for i in 1:size(value,2)-1], " & ") * " \\\\\n"
                end
                
                latex_content *= "\\midrule\n"
                
                # Data rows (limit to first 10 rows)
                for i in 1:min(10, size(value, 1))
                    row_data = [round(value[i, j], digits=4) for j in 1:size(value, 2)]
                    latex_content *= join(row_data, " & ") * " \\\\\n"
                end
                
                latex_content *= """
\\bottomrule
\\end{tabular}
\\caption{$(key) data (showing first $(min(10, size(value, 1))) rows)}
\\end{table}

"""
            end
        end
        
        latex_content *= """
\\end{document}
"""
        
        # Write LaTeX file
        open(joinpath(latex_dir, "activeinference_results.tex"), "w") do f
            write(f, latex_content)
        end
        
        println("📝 LaTeX export: $(latex_dir)")
        return true
        
    catch e
        @warn "LaTeX export failed: $e"
        return false
    end
end

# ====================================
# EXCEL EXPORT
# ====================================

"""Export data to Excel format using CSV as proxy."""
function export_to_excel_csv(data_dict::Dict{String, Any}, output_dir::String)
    excel_dir = joinpath(output_dir, "Excel_exports")
    mkpath(excel_dir)
    
    try
        # Create summary sheet
        summary_data = []
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                push!(summary_data, [
                    key,
                    size(value, 1),
                    size(value, 2),
                    round(mean(value), digits=4),
                    round(std(value), digits=4),
                    round(minimum(value), digits=4),
                    round(maximum(value), digits=4)
                ])
            end
        end
        
        # Write summary CSV
        summary_df = DataFrame(
            Dataset = [row[1] for row in summary_data],
            Rows = [row[2] for row in summary_data],
            Columns = [row[3] for row in summary_data],
            Mean = [row[4] for row in summary_data],
            StdDev = [row[5] for row in summary_data],
            Min = [row[6] for row in summary_data],
            Max = [row[7] for row in summary_data]
        )
        
        CSV.write(joinpath(excel_dir, "summary.csv"), summary_df)
        
        # Write individual datasets
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                
                # Create DataFrame
                if size(value, 2) == 2
                    df = DataFrame(Time = value[:, 1], Value = value[:, 2])
                else
                    df = DataFrame()
                    df.Time = value[:, 1]
                    for i in 2:size(value, 2)
                        df[!, "Col$(i-1)"] = value[:, i]
                    end
                end
                
                CSV.write(joinpath(excel_dir, "$(clean_key).csv"), df)
            end
        end
        
        println("📊 Excel CSV export: $(excel_dir)")
        return true
        
    catch e
        @warn "Excel CSV export failed: $e"
        return false
    end
end

# ====================================
# SQLITE EXPORT
# ====================================

"""Export data to SQLite database."""
function export_to_sqlite(data_dict::Dict{String, Any}, output_dir::String)
    sqlite_dir = joinpath(output_dir, "SQLite_exports")
    mkpath(sqlite_dir)
    
    try
        # Create SQL script (SQLite CLI approach)
        sql_script = """
-- ActiveInference.jl SQLite Export Script
-- Generated: $(now())

-- Create metadata table
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Insert metadata
INSERT OR REPLACE INTO metadata (key, value) VALUES
    ('source', 'ActiveInference.jl'),
    ('export_date', '$(now())'),
    ('dataset_count', '$(length(data_dict))');

"""
        
        # Create data tables
        for (key, value) in data_dict
            if isa(value, AbstractMatrix) && size(value, 1) > 0
                clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                
                sql_script *= """
-- Table for $(key)
CREATE TABLE IF NOT EXISTS $(clean_key) (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
"""
                
                if size(value, 2) == 2
                    sql_script *= """
    time REAL,
    value REAL
);
"""
                else
                    sql_script *= "    time REAL,\n"
                    for i in 2:size(value, 2)
                        sql_script *= "    col$(i-1) REAL"
                        if i < size(value, 2)
                            sql_script *= ","
                        end
                        sql_script *= "\n"
                    end
                    sql_script *= ");\n"
                end
                
                sql_script *= """
-- Insert data for $(key)
"""
                
                for i in 1:size(value, 1)
                    row_data = [value[i, j] for j in 1:size(value, 2)]
                    sql_script *= "INSERT INTO $(clean_key) VALUES (NULL, " * join([string(v) for v in row_data], ", ") * ");\n"
                end
                
                sql_script *= "\n"
            end
        end
        
        # Write SQL script
        open(joinpath(sqlite_dir, "create_database.sql"), "w") do f
            write(f, sql_script)
        end
        
        # Create README
        open(joinpath(sqlite_dir, "README.md"), "w") do f
            println(f, "# ActiveInference.jl SQLite Export")
            println(f, "")
            println(f, "To create the database, run:")
            println(f, "```bash")
            println(f, "sqlite3 activeinference_data.db < create_database.sql")
            println(f, "```")
            println(f, "")
            println(f, "## Tables")
            for (key, value) in data_dict
                if isa(value, AbstractMatrix) && size(value, 1) > 0
                    clean_key = replace(key, r"[^a-zA-Z0-9_]" => "_")
                    println(f, "- `$(clean_key)`: $(key) data")
                end
            end
        end
        
        println("🗄️ SQLite export: $(sqlite_dir)")
        return true
        
    catch e
        @warn "SQLite export failed: $e"
        return false
    end
end

# ====================================
# MAIN EXPORT FUNCTION
# ====================================

"""Comprehensive export function for all formats."""
function comprehensive_export(data_dict::Dict{String, Any}, output_dir::String)
    println("📊 Found $(length(data_dict)) datasets for export")
    
    # Create export directory
    export_dir = joinpath(output_dir, "enhanced_exports")
    mkpath(export_dir)
    
    # Track export results
    export_results = Dict{String, Bool}()
    
    # Export to all formats
    println("\n🔄 Starting comprehensive export...")
    
    # MATLAB export
    println("📊 Exporting to MATLAB...")
    export_results["MATLAB"] = export_to_matlab(data_dict, export_dir)
    
    # R export
    println("📈 Exporting to R...")
    export_results["R"] = export_to_r_csv(data_dict, export_dir)
    
    # Python export
    println("🐍 Exporting to Python...")
    export_results["Python"] = export_to_python(data_dict, export_dir)
    
    # JSON export
    println("📄 Exporting to JSON...")
    json_path = joinpath(export_dir, "activeinference_data.json")
    export_results["JSON"] = export_to_json_schema(data_dict, json_path)
    
    # LaTeX export
    println("📝 Exporting to LaTeX...")
    export_results["LaTeX"] = export_to_latex(data_dict, export_dir)
    
    # Excel CSV export
    println("📊 Exporting to Excel CSV...")
    export_results["Excel"] = export_to_excel_csv(data_dict, export_dir)
    
    # SQLite export
    println("🗄️ Exporting to SQLite...")
    export_results["SQLite"] = export_to_sqlite(data_dict, export_dir)
    
    # Generate export report
    open(joinpath(export_dir, "export_report.md"), "w") do f
        println(f, "# ActiveInference.jl Export Report")
        println(f, "")
        println(f, "Generated: $(now())")
        println(f, "")
        println(f, "## Export Summary")
        println(f, "")
        println(f, "$(length(data_dict)) datasets exported to multiple formats:")
        println(f, "")
        for key in keys(data_dict)
            println(f, "- $(key)")
        end
        println(f, "")
        println(f, "## Format Status")
        println(f, "")
        for (format, success) in export_results
            status = success ? "✅ Success" : "❌ Failed"
            println(f, "- $(format): $(status)")
        end
        println(f, "")
        println(f, "## Directory Structure")
        println(f, "")
        println(f, "```")
        for (root, dirs, files) in walkdir(export_dir)
            level = length(splitpath(root)) - length(splitpath(export_dir))
            indent = "  " ^ level
            rel_path = relpath(root, export_dir)
            if rel_path != "."
                println(f, "$(indent)$(rel_path)/")
            end
            for file in files
                println(f, "$(indent)  $(file)")
            end
        end
        println(f, "```")
    end
    
    # Print summary
    successful_exports = count(values(export_results))
    total_exports = length(export_results)
    
    println("\n✅ Export completed!")
    println("📊 Successful exports: $(successful_exports)/$(total_exports)")
    println("📁 Export directory: $(export_dir)")
    
    # Count total files
    total_files = length(readdir(export_dir)) + 
                  (isdir(joinpath(export_dir, "R_exports")) ? length(readdir(joinpath(export_dir, "R_exports"))) : 0) + 
                  (isdir(joinpath(export_dir, "Python_exports")) ? length(readdir(joinpath(export_dir, "Python_exports"))) : 0) + 
                  (isdir(joinpath(export_dir, "LaTeX_exports")) ? length(readdir(joinpath(export_dir, "LaTeX_exports"))) : 0) + 
                  (isdir(joinpath(export_dir, "Excel_exports")) ? length(readdir(joinpath(export_dir, "Excel_exports"))) : 0) + 
                  (isdir(joinpath(export_dir, "SQLite_exports")) ? length(readdir(joinpath(export_dir, "SQLite_exports"))) : 0)
    
    println("📄 Total exported files: $(total_files)")
    
    return export_results
end

# ====================================
# MAIN FUNCTION
# ====================================

function main()
    if length(ARGS) > 0
        output_dir = ARGS[1]
        if isdir(output_dir)
            println("📊 ActiveInference.jl Export Enhancement")
            println("="^50)
            
            # Load data from data_traces
            trace_dir = joinpath(output_dir, "data_traces")
            if !isdir(trace_dir)
                error("❌ Data traces directory not found: $trace_dir")
            end
            
            # Load all CSV files
            data_dict = Dict{String, Any}()
            
            for file in readdir(trace_dir)
                if endswith(file, ".csv")
                    filepath = joinpath(trace_dir, file)
                    try
                        data = readdlm(filepath, ',', skipstart=6)
                        if size(data, 1) > 0
                            # Convert to numeric
                            numeric_data = zeros(Float64, size(data, 1), size(data, 2))
                            for i in 1:size(data, 1)
                                for j in 1:size(data, 2)
                                    numeric_data[i, j] = parse(Float64, string(data[i, j]))
                                end
                            end
                            
                            key = replace(file, ".csv" => "")
                            data_dict[key] = numeric_data
                            println("📊 Loaded $(key): $(size(numeric_data))")
                        end
                    catch e
                        @warn "Failed to load $file: $e"
                    end
                end
            end
            
            if isempty(data_dict)
                error("❌ No valid data files found in $trace_dir")
            end
            
            # Run comprehensive export
            comprehensive_export(data_dict, output_dir)
            
        else
            error("❌ Directory not found: $output_dir")
        end
    else
        println("Usage: julia export_enhancement.jl <output_directory>")
        println("Example: julia export_enhancement.jl activeinference_outputs_YYYY-MM-DD_HH-MM-SS")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 