#!/usr/bin/env python3
"""
Pipeline Execution Planning Utilities

This module provides execution planning, risk assessment, and resource estimation
capabilities for the GNN processing pipeline.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_execution_plan(steps_to_execute: List[tuple], args, logger) -> Dict[str, Any]:
    """Generate a detailed execution plan with risk assessment and estimates."""
    execution_plan = {
        "steps": [],
        "estimated_duration_minutes": 0.0,
        "high_risk_count": 0,
        "expected_warnings": 0,
        "resource_requirements": {
            "peak_memory_mb": 0,
            "disk_space_mb": 0
        }
    }
    
    # Step duration estimates (in seconds) based on typical execution times
    step_duration_estimates = {
        "0_template.py": 5,
        "1_setup.py": 30,
        "2_tests.py": 120,
        "3_gnn.py": 15,
        "4_model_registry.py": 10,
        "5_type_checker.py": 20,
        "6_validation.py": 25,
        "7_export.py": 15,
        "8_visualization.py": 45,
        "9_advanced_viz.py": 60,
        "10_ontology.py": 20,
        "11_render.py": 30,
        "12_execute.py": 40,
        "13_llm.py": 90,
        "14_ml_integration.py": 120,
        "15_audio.py": 35,
        "16_analysis.py": 50,
        "17_integration.py": 25,
        "18_security.py": 30,
        "19_research.py": 60,
        "20_website.py": 40,
        "21_mcp.py": 25,
        "22_gui.py": 45,
        "23_report.py": 35
    }
    
    # Risk assessment for each step
    step_risk_levels = {
        "8_visualization.py": "medium",  # Matplotlib issues
        "12_execute.py": "medium",     # Dependency on rendered code
        "13_llm.py": "high",           # API dependencies
        "15_audio.py": "medium",       # Audio library dependencies
        "14_ml_integration.py": "high", # Complex ML dependencies
        "2_tests.py": "medium",        # Long-running tests
        "19_research.py": "medium"     # Experimental features
    }
    
    # Resource usage estimates
    step_memory_estimates = {
        "8_visualization.py": 200,    # Matplotlib and large plots
        "9_advanced_viz.py": 300,     # Complex visualizations
        "14_ml_integration.py": 500,  # ML model training
        "2_tests.py": 150,           # Comprehensive test suite
        "13_llm.py": 100            # LLM API processing
    }
    
    step_disk_estimates = {
        "8_visualization.py": 50,    # PNG/SVG files
        "9_advanced_viz.py": 100,    # Interactive HTML plots
        "7_export.py": 25,          # Multiple export formats
        "11_render.py": 15,         # Generated code files
        "20_website.py": 30         # HTML website
    }
    
    for script_name, description in steps_to_execute:
        step_plan = {
            "script_name": script_name,
            "description": description,
            "estimated_duration_seconds": step_duration_estimates.get(script_name, 30),
            "risk_level": step_risk_levels.get(script_name, "low"),
            "dependencies": [],
            "warnings": [],
            "resource_impact": {
                "memory_mb": step_memory_estimates.get(script_name, 50),
                "disk_mb": step_disk_estimates.get(script_name, 10)
            }
        }
        
        # Add dependencies
        step_dependencies = {
            "11_render.py": ["3_gnn.py"],
            "12_execute.py": ["11_render.py"],
            "8_visualization.py": ["3_gnn.py"],
            "9_advanced_viz.py": ["8_visualization.py"],
            "13_llm.py": ["3_gnn.py"],
            "23_report.py": ["8_visualization.py", "13_llm.py"],
            "20_website.py": ["8_visualization.py"],
            "5_type_checker.py": ["3_gnn.py"],
            "6_validation.py": ["5_type_checker.py"],
            "7_export.py": ["3_gnn.py"],
            "10_ontology.py": ["3_gnn.py"],
            "15_audio.py": ["3_gnn.py"],
            "16_analysis.py": ["7_export.py"]
        }
        
        step_plan["dependencies"] = step_dependencies.get(script_name, [])
        
        # Add warnings based on common issues
        if script_name == "12_execute.py":
            if "11_render.py" not in [s[0] for s in steps_to_execute]:
                step_plan["warnings"].append("Render step not included - execution may degrade gracefully")
                
        if script_name == "8_visualization.py":
            step_plan["warnings"].append("May encounter matplotlib backend issues in headless environments")
            
        if script_name == "13_llm.py":
            step_plan["warnings"].append("Requires API keys for full functionality")
            
        if script_name == "15_audio.py":
            step_plan["warnings"].append("Audio libraries may not be available in all environments")
        
        # Update counters and resource estimates
        execution_plan["estimated_duration_minutes"] += step_plan["estimated_duration_seconds"] / 60
        execution_plan["resource_requirements"]["peak_memory_mb"] += step_plan["resource_impact"]["memory_mb"]
        execution_plan["resource_requirements"]["disk_space_mb"] += step_plan["resource_impact"]["disk_mb"]
        
        if step_plan["risk_level"] == "high":
            execution_plan["high_risk_count"] += 1
            
        if step_plan["warnings"]:
            execution_plan["expected_warnings"] += len(step_plan["warnings"])
            
        execution_plan["steps"].append(step_plan)
    
    return execution_plan

def estimate_pipeline_resources(steps_to_execute: List[tuple]) -> Dict[str, Any]:
    """Estimate resource requirements for pipeline execution."""
    
    base_memory = 100  # Base memory for Python process
    base_disk = 50     # Base disk space for outputs
    
    # Resource multipliers based on step types
    memory_intensive_steps = ["8_visualization.py", "9_advanced_viz.py", "14_ml_integration.py", "2_tests.py"]
    disk_intensive_steps = ["8_visualization.py", "9_advanced_viz.py", "7_export.py", "20_website.py"]
    
    estimated_memory = base_memory
    estimated_disk = base_disk
    
    script_names = [step[0] for step in steps_to_execute]
    
    for script_name in script_names:
        if script_name in memory_intensive_steps:
            estimated_memory += 150
        else:
            estimated_memory += 50
            
        if script_name in disk_intensive_steps:
            estimated_disk += 100
        else:
            estimated_disk += 20
    
    return {
        "estimated_memory_mb": estimated_memory,
        "estimated_disk_mb": estimated_disk,
        "memory_intensive_steps": len([s for s in script_names if s in memory_intensive_steps]),
        "disk_intensive_steps": len([s for s in script_names if s in disk_intensive_steps])
    }

def generate_risk_assessment(steps_to_execute: List[tuple]) -> Dict[str, Any]:
    """Generate detailed risk assessment for pipeline execution."""
    
    risk_factors = {
        "dependency_risks": [],
        "external_service_risks": [],
        "resource_risks": [],
        "compatibility_risks": [],
        "overall_risk_score": 0  # 0-100 scale
    }
    
    script_names = [step[0] for step in steps_to_execute]
    risk_score = 0
    
    # Dependency risks
    if "12_execute.py" in script_names and "11_render.py" not in script_names:
        risk_factors["dependency_risks"].append("Execute step without render step - high failure risk")
        risk_score += 20
        
    if any(step in script_names for step in ["8_visualization.py", "9_advanced_viz.py"]) and "3_gnn.py" not in script_names:
        risk_factors["dependency_risks"].append("Visualization steps without GNN parsing - will fail")
        risk_score += 30
    
    # External service risks
    if "13_llm.py" in script_names:
        risk_factors["external_service_risks"].append("LLM step depends on external API availability")
        risk_score += 15
        
    # Resource risks
    memory_intensive = ["8_visualization.py", "9_advanced_viz.py", "14_ml_integration.py"]
    if len([s for s in script_names if s in memory_intensive]) > 2:
        risk_factors["resource_risks"].append("Multiple memory-intensive steps may cause resource exhaustion")
        risk_score += 10
        
    # Compatibility risks
    if "8_visualization.py" in script_names:
        risk_factors["compatibility_risks"].append("Matplotlib backend compatibility issues in headless environments")
        risk_score += 10
        
    if "15_audio.py" in script_names:
        risk_factors["compatibility_risks"].append("Audio library compatibility varies across systems")
        risk_score += 5
    
    risk_factors["overall_risk_score"] = min(risk_score, 100)  # Cap at 100
    
    return risk_factors

def create_execution_timeline(steps_to_execute: List[tuple]) -> List[Dict[str, Any]]:
    """Create a detailed execution timeline with parallel execution opportunities."""
    
    timeline = []
    current_time = 0
    
    # Step duration estimates
    durations = {
        "3_gnn.py": 15, "5_type_checker.py": 20, "7_export.py": 15,
        "8_visualization.py": 45, "12_execute.py": 40, "15_audio.py": 35
    }
    
    # Dependencies that prevent parallel execution
    dependencies = {
        "5_type_checker.py": ["3_gnn.py"],
        "8_visualization.py": ["3_gnn.py"],
        "12_execute.py": ["11_render.py"]
    }
    
    completed_steps = set()
    
    for script_name, description in steps_to_execute:
        step_deps = dependencies.get(script_name, [])
        can_start_immediately = all(dep.replace('.py', '') in [s.replace('.py', '') for s in completed_steps] for dep in step_deps)
        
        if not can_start_immediately:
            # Add wait time for dependencies
            current_time += 5
            
        duration = durations.get(script_name, 30)
        
        timeline_entry = {
            "step": script_name,
            "description": description,
            "start_time": current_time,
            "duration": duration,
            "end_time": current_time + duration,
            "dependencies": step_deps,
            "parallel_opportunities": []
        }
        
        # Check for steps that could run in parallel (no interdependencies)
        for other_script, _ in steps_to_execute:
            if (other_script != script_name and 
                other_script not in step_deps and 
                script_name not in dependencies.get(other_script, [])):
                timeline_entry["parallel_opportunities"].append(other_script)
        
        timeline.append(timeline_entry)
        current_time += duration
        completed_steps.add(script_name.replace('.py', ''))
    
    return timeline
