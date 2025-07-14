#!/usr/bin/env julia

"""
Enhanced Integration Suite for ActiveInference.jl - Maximum POMDP Understanding

This script provides a comprehensive analysis pipeline integrating both existing and new capabilities:

EXISTING MODULES:
- Advanced POMDP analysis (information theory, convergence, bounds)
- Enhanced visualizations (interactive plots, 3D, networks, animations)
- Statistical analysis (hypothesis testing, Bayesian comparison)
- Multi-format export (MATLAB, R, Python, JSON, LaTeX)

NEW STRATEGIC ENHANCEMENTS:
- Meta-cognitive analysis (hierarchical reasoning, meta-awareness)
- Adaptive precision and attention mechanisms
- Counterfactual reasoning and what-if analysis
- Multi-scale temporal analysis across different horizons
- Advanced uncertainty quantification (epistemic vs aleatoric)

This integration provides maximum POMDP understanding and flexibility for research applications.

Usage: julia enhanced_integration_suite.jl <output_directory>
"""

using Pkg
using Dates
using Printf

# Include existing analysis modules
const SCRIPT_DIR = @__DIR__
include(joinpath(SCRIPT_DIR, "advanced_pomdp_analysis.jl"))
include(joinpath(SCRIPT_DIR, "enhanced_visualization.jl"))
include(joinpath(SCRIPT_DIR, "statistical_analysis.jl"))
include(joinpath(SCRIPT_DIR, "export_enhancement.jl"))

# Include new strategic enhancement modules
include(joinpath(SCRIPT_DIR, "meta_cognitive_analysis.jl"))
include(joinpath(SCRIPT_DIR, "adaptive_precision_attention.jl"))
include(joinpath(SCRIPT_DIR, "counterfactual_reasoning.jl"))
include(joinpath(SCRIPT_DIR, "multi_scale_temporal_analysis.jl"))
include(joinpath(SCRIPT_DIR, "uncertainty_quantification.jl"))

"""Run the complete enhanced integration suite with maximum POMDP understanding."""
function run_enhanced_integration_suite(output_dir::String)
    if !isdir(output_dir)
        error("❌ Output directory not found: $output_dir")
    end
    
    println("🚀 ActiveInference.jl Enhanced Integration Suite - Maximum POMDP Understanding")
    println("="^80)
    println("📁 Analyzing data in: $output_dir")
    println("⏰ Started at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()
    
    start_time = now()
    analysis_log = String[]
    
    # ====================================
    # PHASE 1: EXISTING ADVANCED ANALYSIS
    # ====================================
    println("🔬 PHASE 1: Existing Advanced Analysis")
    println("="^50)
    
    # Step 1: Advanced POMDP Analysis
    println("🔍 Step 1: Advanced POMDP Analysis")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_pomdp_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Advanced POMDP Analysis: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Advanced POMDP Analysis: $status ($(step_duration))")
        @warn "Advanced POMDP analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")
    
    # Step 2: Enhanced Visualizations
    println("🎨 Step 2: Enhanced Visualizations")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        create_enhanced_visualizations(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Enhanced Visualizations: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Enhanced Visualizations: $status ($(step_duration))")
        @warn "Enhanced visualizations failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")
    
        # Step 3: Statistical Analysis
    println("📊 Step 3: Statistical Analysis")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_statistical_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Statistical Analysis: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Statistical Analysis: $status ($(step_duration))")
        @warn "Statistical analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    println()

    # ====================================
    # PHASE 2: NEW STRATEGIC ENHANCEMENTS
    # ====================================
    println("🧠 PHASE 2: Strategic Enhancements for Maximum POMDP Understanding")
    println("="^70)

    # Step 4: Meta-Cognitive Analysis
    println("🧠 Step 4: Meta-Cognitive Analysis")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_metacognitive_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Meta-Cognitive Analysis: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Meta-Cognitive Analysis: $status ($(step_duration))")
        @warn "Meta-cognitive analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    # Step 5: Adaptive Precision and Attention
    println("🎯 Step 5: Adaptive Precision and Attention Analysis")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_precision_attention_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Adaptive Precision & Attention: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Adaptive Precision & Attention: $status ($(step_duration))")
        @warn "Adaptive precision and attention analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    # Step 6: Counterfactual Reasoning
    println("🔀 Step 6: Counterfactual Reasoning Analysis")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_counterfactual_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Counterfactual Reasoning: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Counterfactual Reasoning: $status ($(step_duration))")
        @warn "Counterfactual reasoning analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    # Step 7: Multi-Scale Temporal Analysis
    println("⏰ Step 7: Multi-Scale Temporal Analysis")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_temporal_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Multi-Scale Temporal Analysis: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Multi-Scale Temporal Analysis: $status ($(step_duration))")
        @warn "Multi-scale temporal analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    # Step 8: Advanced Uncertainty Quantification
    println("📊 Step 8: Advanced Uncertainty Quantification")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        comprehensive_uncertainty_analysis(output_dir)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Uncertainty Quantification: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Uncertainty Quantification: $status ($(step_duration))")
        @warn "Uncertainty quantification analysis failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    println()

    # ====================================
    # PHASE 3: INTEGRATION AND EXPORT
    # ====================================
    println("📦 PHASE 3: Integration and Multi-Format Export")
    println("="^50)

    # Step 9: Multi-Format Export
    println("📦 Step 9: Multi-Format Export")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        # Load data for export
        data_dict = Dict{String, Any}()
        trace_dir = joinpath(output_dir, "data_traces")
        if isdir(trace_dir)
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
                        end
                    catch e
                        @warn "Failed to load $file for export: $e"
                    end
                end
            end
        end
        
        if !isempty(data_dict)
            comprehensive_export(data_dict, output_dir)
        else
            @warn "No data found for export"
        end
        
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Multi-Format Export: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Multi-Format Export: $status ($(step_duration))")
        @warn "Multi-format export failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")

    # Step 10: Enhanced Comprehensive Report
    println("📋 Step 10: Enhanced Comprehensive Report Generation")
    step_start = now()
    status = "❌ FAILED: Unknown error"
    step_duration = Millisecond(0)
    try
        generate_enhanced_comprehensive_report(output_dir, analysis_log, start_time)
        step_duration = now() - step_start
        status = "✅ COMPLETED"
        push!(analysis_log, "Enhanced Report Generation: $status ($(step_duration))")
    catch e
        step_duration = now() - step_start
        status = "❌ FAILED: $e"
        push!(analysis_log, "Enhanced Report Generation: $status ($(step_duration))")
        @warn "Enhanced report generation failed" exception=e
    end
    println("Status: $status | Duration: $(step_duration)")
    
    println()
    
    # ====================================
    # FINAL SUMMARY
    # ====================================
    total_duration = now() - start_time
    
    println("🎉 ENHANCED INTEGRATION SUITE - MAXIMUM POMDP UNDERSTANDING COMPLETED")
    println("="^80)
    println("⏱️  Total Duration: $total_duration")
    println("📁 Results saved to: $output_dir")
    println()
    
    println("📊 Complete Analysis Summary:")
    for log_entry in analysis_log
        status_symbol = occursin("COMPLETED", log_entry) ? "✅" : "❌"
        println("  $status_symbol $log_entry")
    end
    
    println()
    println("🔍 Enhanced Analysis Directories:")
    
    # Check which directories were created
    output_dirs = [
        ("advanced_analysis", "🔬 Advanced POMDP Analysis"),
        ("enhanced_visualizations", "🎨 Enhanced Visualizations"),
        ("statistical_analysis", "📊 Statistical Analysis"),
        ("metacognitive_analysis", "🧠 Meta-Cognitive Analysis"),
        ("adaptive_precision_attention", "🎯 Adaptive Precision & Attention"),
        ("counterfactual_reasoning", "🔀 Counterfactual Reasoning"),
        ("multi_scale_temporal", "⏰ Multi-Scale Temporal Analysis"),
        ("uncertainty_quantification", "📊 Uncertainty Quantification"),
        ("enhanced_exports", "📦 Multi-Format Exports"),
        ("comprehensive_report", "📋 Enhanced Comprehensive Report")
    ]
    
    for (dir_name, description) in output_dirs
        full_path = joinpath(output_dir, dir_name)
        if isdir(full_path)
            file_count = length(readdir(full_path))
            println("  • $description: $(file_count) files")
            println("    📁 $full_path")
        end
    end
    
    println()
    println("🌐 Enhanced Results Access:")
    
    # Check for main dashboard and reports
    dashboard_path = joinpath(output_dir, "enhanced_visualizations", "dashboard.html")
    if isfile(dashboard_path)
        println("  • Interactive Dashboard: $dashboard_path")
    end
    
    comprehensive_report_path = joinpath(output_dir, "comprehensive_report", "enhanced_comprehensive_report.html")
    if isfile(comprehensive_report_path)
        println("  • Enhanced Comprehensive Report: $comprehensive_report_path")
    end
    
    # Show key analysis files
    println()
    println("🎯 Key Analysis Results:")
    
    key_files = [
        ("metacognitive_analysis/metacognitive_analysis_report.md", "Meta-Cognitive Analysis Report"),
        ("adaptive_precision_attention/precision_attention_report.md", "Precision & Attention Report"),
        ("counterfactual_reasoning/counterfactual_analysis_report.md", "Counterfactual Reasoning Report"),
        ("multi_scale_temporal/temporal_analysis_report.md", "Multi-Scale Temporal Report"),
        ("uncertainty_quantification/uncertainty_analysis_report.md", "Uncertainty Quantification Report")
    ]
    
    for (file_path, description) in key_files
        full_file_path = joinpath(output_dir, file_path)
        if isfile(full_file_path)
            println("  • $description: $full_file_path")
        end
    end
    
    println()
    println("🏆 MAXIMUM POMDP UNDERSTANDING ACHIEVED!")
    println("This enhanced suite provides comprehensive analysis across:")
    println("  • Hierarchical cognitive reasoning and meta-awareness")
    println("  • Dynamic attention and precision mechanisms")
    println("  • Counterfactual and alternative scenario analysis")
    println("  • Multi-scale temporal reasoning and planning")
    println("  • Advanced uncertainty decomposition and quantification")
    println("  • Cross-platform research integration and export")
    
    return analysis_log
end

"""Generate enhanced comprehensive report integrating all analysis modules."""
function generate_enhanced_comprehensive_report(output_dir::String, analysis_log::Vector{String}, start_time::DateTime)
    println("📋 Generating Enhanced Comprehensive Report")
    
    # Create comprehensive report directory
    report_dir = joinpath(output_dir, "comprehensive_report")
    mkpath(report_dir)
    
    # Generate HTML report
    html_path = joinpath(report_dir, "enhanced_comprehensive_report.html")
    
    open(html_path, "w") do f
        write_html_header(f)
        write_executive_summary(f, analysis_log, start_time)
        write_analysis_sections(f, output_dir)
        write_html_footer(f)
    end
    
    # Generate markdown summary
    md_path = joinpath(report_dir, "enhanced_comprehensive_summary.md")
    
    open(md_path, "w") do f
        write_markdown_summary(f, analysis_log, start_time, output_dir)
    end
    
    println("📋 Enhanced comprehensive report generated:")
    println("  • HTML Report: $html_path")
    println("  • Markdown Summary: $md_path")
end

"""Write HTML header for comprehensive report."""
function write_html_header(f::IOStream)
    println(f, """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ActiveInference.jl Enhanced Analysis Report - Maximum POMDP Understanding</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f7fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .summary-box { background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .success { color: #27ae60; font-weight: bold; }
        .failure { color: #e74c3c; font-weight: bold; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #3498db; color: white; border-radius: 5px; text-align: center; min-width: 150px; }
        .analysis-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .analysis-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }
        .file-link { color: #3498db; text-decoration: none; font-weight: bold; }
        .file-link:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #3498db; color: white; }
        .enhancement-highlight { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ActiveInference.jl Enhanced Analysis Report</h1>
        <div class="enhancement-highlight">
            <h2 style="color: white; border: none; margin: 0;">🏆 Maximum POMDP Understanding & Flexibility Achieved</h2>
            <p>This comprehensive analysis integrates advanced POMDP understanding with strategic enhancements for meta-cognitive reasoning, adaptive precision, counterfactual analysis, multi-scale temporal understanding, and advanced uncertainty quantification.</p>
        </div>
    """)
end

"""Write executive summary section."""
function write_executive_summary(f::IOStream, analysis_log::Vector{String}, start_time::DateTime)
    total_duration = now() - start_time
    completed_analyses = count(log -> occursin("COMPLETED", log), analysis_log)
    failed_analyses = length(analysis_log) - completed_analyses
    
    println(f, """
        <div class="summary-box">
            <h2>📊 Executive Summary</h2>
            <p><strong>Analysis Completed:</strong> $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))</p>
            <p><strong>Total Duration:</strong> $total_duration</p>
            
            <div style="margin: 20px 0;">
                <div class="metric">
                    <div style="font-size: 24px;">$completed_analyses</div>
                    <div>Completed</div>
                </div>
                <div class="metric" style="background: #e74c3c;">
                    <div style="font-size: 24px;">$failed_analyses</div>
                    <div>Failed</div>
                </div>
                <div class="metric" style="background: #f39c12;">
                    <div style="font-size: 24px;">$(length(analysis_log))</div>
                    <div>Total Analyses</div>
                </div>
            </div>
            
            <h3>Analysis Status:</h3>
            <ul>
    """)
    
    for log_entry in analysis_log
        if occursin("COMPLETED", log_entry)
            println(f, "        <li class=\"success\">✅ $log_entry</li>")
        else
            println(f, "        <li class=\"failure\">❌ $log_entry</li>")
        end
    end
    
    println(f, """
            </ul>
        </div>
    """)
end

"""Write analysis sections for each module."""
function write_analysis_sections(f::IOStream, output_dir::String)
    println(f, """
        <h2>🔬 Analysis Modules</h2>
        <div class="analysis-grid">
    """)
    
    # Define analysis modules with their descriptions
    modules = [
        ("advanced_analysis", "🔬 Advanced POMDP Analysis", "Information-theoretic measures, convergence analysis, and theoretical bounds"),
        ("enhanced_visualizations", "🎨 Enhanced Visualizations", "Interactive plots, 3D visualizations, and comprehensive dashboards"),
        ("statistical_analysis", "📊 Statistical Analysis", "Hypothesis testing, Bayesian comparison, and model validation"),
        ("metacognitive_analysis", "🧠 Meta-Cognitive Analysis", "Hierarchical reasoning, meta-awareness, and higher-order beliefs"),
        ("adaptive_precision_attention", "🎯 Adaptive Precision & Attention", "Dynamic precision modulation and attention mechanisms"),
        ("counterfactual_reasoning", "🔀 Counterfactual Reasoning", "What-if analysis and alternative scenario exploration"),
        ("multi_scale_temporal", "⏰ Multi-Scale Temporal Analysis", "Hierarchical temporal reasoning across multiple scales"),
        ("uncertainty_quantification", "📊 Uncertainty Quantification", "Epistemic vs aleatoric uncertainty decomposition"),
        ("enhanced_exports", "📦 Multi-Format Export", "MATLAB, R, Python, JSON, and LaTeX export capabilities")
    ]
    
    for (dir_name, title, description) in modules
        full_path = joinpath(output_dir, dir_name)
        if isdir(full_path)
            file_count = length(readdir(full_path))
            
            println(f, """
                <div class="analysis-card">
                    <h3>$title</h3>
                    <p>$description</p>
                    <p><strong>Files Generated:</strong> $file_count</p>
                    <p><strong>Location:</strong> <code>$dir_name/</code></p>
                </div>
            """)
        end
    end
    
    println(f, """
        </div>
    """)
end

"""Write HTML footer."""
function write_html_footer(f::IOStream)
    println(f, """
        <div class="enhancement-highlight" style="margin-top: 40px;">
            <h2 style="color: white; border: none; margin: 0;">🎯 Strategic Enhancements Summary</h2>
            <p>This enhanced suite provides unprecedented POMDP understanding through:</p>
            <ul style="color: white;">
                <li><strong>Meta-Cognitive Analysis:</strong> Understanding how the agent thinks about its own thinking</li>
                <li><strong>Adaptive Precision:</strong> Dynamic focus and attention allocation mechanisms</li>
                <li><strong>Counterfactual Reasoning:</strong> Alternative scenario analysis and regret/relief quantification</li>
                <li><strong>Multi-Scale Temporal:</strong> Hierarchical time reasoning across different planning horizons</li>
                <li><strong>Advanced Uncertainty:</strong> Decomposition of knowledge vs environmental uncertainty</li>
            </ul>
        </div>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by ActiveInference.jl Enhanced Integration Suite</p>
            <p>Maximum POMDP Understanding & Flexibility Framework</p>
        </footer>
    </div>
</body>
</html>
    """)
end

"""Write markdown summary."""
function write_markdown_summary(f::IOStream, analysis_log::Vector{String}, start_time::DateTime, output_dir::String)
    total_duration = now() - start_time
    
    println(f, """
# ActiveInference.jl Enhanced Analysis Summary

**Maximum POMDP Understanding & Flexibility Achieved**

## Executive Summary

- **Analysis Completed:** $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
- **Total Duration:** $total_duration
- **Output Directory:** $output_dir

## Analysis Results

""")
    
    for log_entry in analysis_log
        status_symbol = occursin("COMPLETED", log_entry) ? "✅" : "❌"
        println(f, "- $status_symbol $log_entry")
    end
    
    println(f, """

## Strategic Enhancements

This enhanced suite provides maximum POMDP understanding through:

### 🧠 Meta-Cognitive Analysis
- Hierarchical reasoning and meta-awareness assessment
- Higher-order belief monitoring and confidence estimation
- Theory of mind modeling for multi-agent scenarios

### 🎯 Adaptive Precision and Attention
- Dynamic precision modulation based on context and uncertainty
- Attention allocation and resource distribution mechanisms
- Multi-modal attention coordination

### 🔀 Counterfactual Reasoning
- Alternative scenario generation and what-if analysis
- Regret and relief quantification
- Causal intervention modeling and outcome prediction

### ⏰ Multi-Scale Temporal Analysis
- Hierarchical temporal reasoning across different time horizons
- Planning depth optimization and adaptive horizon selection
- Temporal coherence and consistency analysis

### 📊 Advanced Uncertainty Quantification
- Epistemic vs aleatoric uncertainty decomposition
- Model uncertainty and parameter uncertainty assessment
- Uncertainty-aware decision making analysis

## Research Integration

The enhanced suite provides comprehensive export capabilities for:
- **MATLAB/Octave:** .mat files with complete analysis results
- **R/RStudio:** CSV exports with automated loading scripts
- **Python/SciPy:** NumPy arrays and pandas DataFrames
- **JSON:** Structured metadata for web applications
- **LaTeX:** Publication-ready tables and figures

## Conclusion

This enhanced ActiveInference.jl implementation provides unprecedented insight into POMDP reasoning processes, combining traditional analysis with cutting-edge meta-cognitive, temporal, and uncertainty quantification capabilities for maximum research flexibility and understanding.
""")
end

# Export the main function
export run_enhanced_integration_suite

# Main execution
function main()
    if length(ARGS) >= 1
        output_directory = ARGS[1]
        println("🚀 Starting Enhanced Integration Suite")
        println("Output directory: $output_directory")
        
        try
            run_enhanced_integration_suite(output_directory)
        catch e
            @error "Enhanced integration suite failed" exception=e
            exit(1)
        end
    else
        println("Usage: julia enhanced_integration_suite.jl <output_directory>")
        println("Example: julia enhanced_integration_suite.jl /path/to/activeinference/output")
        exit(1)
    end
end

# Run main if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

println("🚀 Enhanced Integration Suite for Maximum POMDP Understanding Loaded Successfully") 