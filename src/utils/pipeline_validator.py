#!/usr/bin/env python3
"""
Pipeline Validation Utilities

This module provides comprehensive validation capabilities for pipeline steps,
including dependency checking, prerequisite validation, and step sequencing.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

def validate_step_prerequisites(script_name: str, args, logger) -> Dict[str, Any]:
    """Validate prerequisites for a pipeline step before execution."""
    result = {
        "passed": True,
        "warnings": [],
        "errors": []
    }
    
    # Import get_output_dir_for_script to use consistent output directory mapping
    from pipeline.config import get_output_dir_for_script
    
    # Define step dependencies (step name -> [required prerequisite steps])
    step_dependencies = {
        "11_render.py": ["3_gnn.py"],  # Render needs parsed GNN files
        "12_execute.py": ["11_render.py"],  # Execute needs rendered code
        "8_visualization.py": ["3_gnn.py"],  # Visualization needs parsed files
        "9_advanced_viz.py": ["8_visualization.py"],  # Advanced viz builds on basic viz
        "13_llm.py": ["3_gnn.py"],  # LLM analysis needs parsed files
        "23_report.py": ["8_visualization.py", "13_llm.py"],  # Report needs viz and analysis
        "20_website.py": ["8_visualization.py"],  # Website needs visualizations
        "5_type_checker.py": ["3_gnn.py"],  # Type checking needs parsed files
        "6_validation.py": ["5_type_checker.py"],  # Validation builds on type checking
        "7_export.py": ["3_gnn.py"],  # Export needs parsed files
        "10_ontology.py": ["3_gnn.py"],  # Ontology processing needs parsed files
        "15_audio.py": ["3_gnn.py"],  # Audio generation needs parsed files
        "16_analysis.py": ["7_export.py"],  # Analysis needs exported data
    }
    
    required_steps = step_dependencies.get(script_name, [])
    
    if required_steps:
        # Check if prerequisite step outputs exist
        for req_step in required_steps:
            # Use centralized function to get correct output directory
            expected_output_dir = get_output_dir_for_script(req_step, args.output_dir)
            
            # Also check for nested directories (legacy pattern)
            nested_output_dir = expected_output_dir / expected_output_dir.name
            
            # Check both possible locations
            has_expected = expected_output_dir.exists()
            has_nested = nested_output_dir.exists()
            
            if has_expected and has_nested:
                # Both exist - nested structure detected (issue warning but don't fail)
                warning_msg = f"Nested directory structure detected for {req_step}: {nested_output_dir}. Consider using only {expected_output_dir}."
                result["warnings"].append(warning_msg)
            elif has_nested and not has_expected:
                # Only nested exists - issue a warning but don't fail
                warning_msg = f"Prerequisite output found at nested location: {nested_output_dir}. Consider fixing nested directory structure."
                result["warnings"].append(warning_msg)
            elif not has_expected and not has_nested:
                # Not found at either location
                warning_msg = f"Missing prerequisite output directory: {expected_output_dir}. Step {req_step} may not have been executed."
                result["warnings"].append(warning_msg)
                
            # Check for specific required files based on step type
            if req_step == "3_gnn.py":
                # Check for parsed GNN files
                gnn_output_dir = args.output_dir / "3_gnn_output"
                if gnn_output_dir.exists():
                    # Look for parsed files at any depth
                    parsed_files = list(gnn_output_dir.rglob("*_parsed.json"))
                    if not parsed_files:
                        result["warnings"].append("No parsed GNN files found in 3_gnn_output")
                    # Note: Parsed files in model-specific subdirectories (e.g., 3_gnn_output/model_name/model_name_parsed.json)
                    # is the correct and intended structure - no warning needed
                        
            elif req_step == "11_render.py":
                # Check for rendered simulation code
                render_output_dir = args.output_dir / "11_render_output"
                if render_output_dir.exists():
                    rendered_files = list(render_output_dir.rglob("*.py")) + list(render_output_dir.rglob("*.jl"))
                    if not rendered_files:
                        result["warnings"].append("No rendered simulation files found in 11_render_output")
    
    return result

def validate_pipeline_step_sequence(steps_to_execute: List[tuple], logger) -> Dict[str, Any]:
    """Validate the sequence of pipeline steps for dependency issues."""
    validation_result = {
        "valid": True,
        "warnings": [],
        "recommendations": []
    }
    
    # Extract script names from steps_to_execute
    script_names = [step[0] for step in steps_to_execute]
    
    # Define critical dependency chains
    dependency_chains = [
        ["3_gnn.py", "5_type_checker.py", "6_validation.py"],
        ["3_gnn.py", "7_export.py"],
        ["3_gnn.py", "8_visualization.py", "9_advanced_viz.py"],
        ["3_gnn.py", "11_render.py", "12_execute.py"],
        ["8_visualization.py", "20_website.py"],
        ["8_visualization.py", "23_report.py"]
    ]
    
    for chain in dependency_chains:
        # Check if any step in the chain is being executed
        chain_steps_in_execution = [step for step in chain if step in script_names]
        
        if len(chain_steps_in_execution) > 1:
            # Verify order is correct
            chain_indices = [script_names.index(step) for step in chain_steps_in_execution]
            expected_indices = [chain.index(step) for step in chain_steps_in_execution]
            
            # Sort both to compare order
            if chain_indices != sorted(chain_indices, key=lambda x: expected_indices[chain_indices.index(x)]):
                validation_result["warnings"].append(
                    f"Dependency chain order issue detected: {' → '.join(chain_steps_in_execution)}"
                )
                validation_result["recommendations"].append(
                    f"Consider reordering steps to: {' → '.join(chain)}"
                )
    
    # Check for missing critical dependencies
    critical_steps = ["3_gnn.py"]  # Core parsing step
    script_names = [step[0] for step in steps_to_execute]
    
    for critical_step in critical_steps:
        if critical_step not in script_names:
            dependent_steps = []
            for step in script_names:
                if step in ["5_type_checker.py", "8_visualization.py", "11_render.py", "12_execute.py"]:
                    dependent_steps.append(step)
            
            if dependent_steps:
                validation_result["warnings"].append(
                    f"Critical step {critical_step} not included, but dependent steps found: {', '.join(dependent_steps)}"
                )
                validation_result["recommendations"].append(
                    f"Add {critical_step} to the execution sequence for complete functionality"
                )
    
    return validation_result

def validate_step_outputs(script_name: str, output_dir: Path) -> Dict[str, Any]:
    """Validate that a step produced expected outputs."""
    validation = {
        "step_name": script_name,
        "outputs_created": [],
        "missing_outputs": [],
        "validation_passed": True
    }
    
    # Define expected outputs for each step
    expected_outputs = {
        "3_gnn.py": ["*_parsed.json", "*.gnn"],
        "8_visualization.py": ["*.png", "*_analysis.json"],
        "11_render.py": ["*.py", "*.jl"],
        "12_execute.py": ["*_results.json", "*_report.md"],
        "7_export.py": ["*.json", "*.xml", "*.graphml"]
    }
    
    step_number = script_name.split('_')[0]
    step_name = script_name.split('_')[1].replace('.py', '')
    step_output_dir = output_dir / f"{step_number}_{step_name}_output"
    
    if not step_output_dir.exists():
        validation["missing_outputs"].append(f"Output directory {step_output_dir} not found")
        validation["validation_passed"] = False
        return validation
    
    patterns = expected_outputs.get(script_name, [])
    for pattern in patterns:
        files = list(step_output_dir.rglob(pattern))
        if files:
            validation["outputs_created"].extend([str(f.relative_to(output_dir)) for f in files])
        else:
            validation["missing_outputs"].append(f"No files matching pattern {pattern}")
    
    if validation["missing_outputs"]:
        validation["validation_passed"] = False
    
    return validation

def check_pipeline_readiness(steps_to_execute: List[tuple], args) -> Dict[str, Any]:
    """Comprehensive pipeline readiness check before execution."""
    readiness_check = {
        "ready": True,
        "blocking_issues": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Check basic requirements
    if not args.target_dir.exists():
        readiness_check["blocking_issues"].append(f"Target directory {args.target_dir} does not exist")
        readiness_check["ready"] = False
    
    # Check for GNN files if GNN-dependent steps are included
    gnn_dependent_steps = ["5_type_checker.py", "8_visualization.py", "11_render.py", "13_llm.py"]
    script_names = [step[0] for step in steps_to_execute]
    
    if any(step in script_names for step in gnn_dependent_steps):
        gnn_files = list(args.target_dir.rglob("*.md")) + list(args.target_dir.rglob("*.gnn"))
        if not gnn_files:
            readiness_check["blocking_issues"].append(f"No GNN files found in {args.target_dir}")
            readiness_check["ready"] = False
    
    # Check output directory is writable
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        test_file = args.output_dir / ".pipeline_test"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        readiness_check["blocking_issues"].append(f"Output directory {args.output_dir} is not writable: {e}")
        readiness_check["ready"] = False
    
    return readiness_check
