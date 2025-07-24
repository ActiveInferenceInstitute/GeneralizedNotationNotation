"""
LLM module for GNN Processing Pipeline.

This module provides LLM-enhanced analysis and processing for GNN models.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import re
from datetime import datetime

from utils.pipeline_template import (
    log_step_start,
    log_step_success,
    log_step_error,
    log_step_warning
)

def process_llm(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs
) -> bool:
    """
    Process GNN files with LLM-enhanced analysis.
    
    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments
        
    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("llm")
    
    try:
        log_step_start(logger, "Processing LLM")
        
        # Create results directory
        results_dir = output_dir / "llm_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "analysis_results": [],
            "model_insights": [],
            "code_suggestions": [],
            "documentation_generated": []
        }
        
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        if not gnn_files:
            logger.warning("No GNN files found for LLM processing")
            results["success"] = False
            results["errors"].append("No GNN files found")
        else:
            results["processed_files"] = len(gnn_files)
            
            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    file_analysis = analyze_gnn_file_with_llm(gnn_file, verbose)
                    results["analysis_results"].append(file_analysis)
                    
                    # Generate insights
                    insights = generate_model_insights(file_analysis)
                    results["model_insights"].append(insights)
                    
                    # Generate code suggestions
                    suggestions = generate_code_suggestions(file_analysis)
                    results["code_suggestions"].append(suggestions)
                    
                    # Generate documentation
                    docs = generate_documentation(file_analysis)
                    results["documentation_generated"].append(docs)
                    
                except Exception as e:
                    error_info = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")
        
        # Save detailed results
        results_file = results_dir / "llm_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate summary report
        summary = generate_llm_summary(results)
        summary_file = results_dir / "llm_summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        if results["success"]:
            log_step_success(logger, "LLM processing completed successfully")
        else:
            log_step_error(logger, "LLM processing failed")
        
        return results["success"]
        
    except Exception as e:
        log_step_error(logger, "LLM processing failed", {"error": str(e)})
        return False

def analyze_gnn_file_with_llm(file_path: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a GNN file using LLM-enhanced techniques.
    
    Args:
        file_path: Path to the GNN file
        verbose: Enable verbose output
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract GNN structure
        variables = extract_variables(content)
        connections = extract_connections(content)
        sections = extract_sections(content)
        
        # Perform semantic analysis
        semantic_analysis = perform_semantic_analysis(content, variables, connections)
        
        # Generate model complexity metrics
        complexity_metrics = calculate_complexity_metrics(variables, connections)
        
        # Identify patterns and anti-patterns
        patterns = identify_patterns(content, variables, connections)
        
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "line_count": len(content.splitlines()),
            "variables": variables,
            "connections": connections,
            "sections": sections,
            "semantic_analysis": semantic_analysis,
            "complexity_metrics": complexity_metrics,
            "patterns": patterns,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise Exception(f"Failed to analyze {file_path}: {e}")

def extract_variables(content: str) -> List[Dict[str, Any]]:
    """Extract variables from GNN content."""
    variables = []
    
    # Look for variable definitions
    var_patterns = [
        r'(\w+)\s*:\s*(\w+)',  # name: type
        r'(\w+)\s*=\s*([^;\n]+)',  # name = value
        r'(\w+)\s*\[([^\]]+)\]',  # name[dimensions]
    ]
    
    for pattern in var_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            variables.append({
                "name": match.group(1),
                "definition": match.group(0),
                "line": content[:match.start()].count('\n') + 1
            })
    
    return variables

def extract_connections(content: str) -> List[Dict[str, Any]]:
    """Extract connections from GNN content."""
    connections = []
    
    # Look for connection patterns
    conn_patterns = [
        r'(\w+)\s*->\s*(\w+)',  # source -> target
        r'(\w+)\s*→\s*(\w+)',   # source → target
        r'(\w+)\s*connects\s*(\w+)',  # source connects target
    ]
    
    for pattern in conn_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            connections.append({
                "source": match.group(1),
                "target": match.group(2),
                "definition": match.group(0),
                "line": content[:match.start()].count('\n') + 1
            })
    
    return connections

def extract_sections(content: str) -> List[Dict[str, Any]]:
    """Extract sections from GNN content."""
    sections = []
    
    # Look for section headers
    section_pattern = r'^#+\s*(.+)$'
    matches = re.finditer(section_pattern, content, re.MULTILINE)
    
    for match in matches:
        sections.append({
            "title": match.group(1).strip(),
            "level": len(match.group(0)) - len(match.group(0).lstrip('#')),
            "line": content[:match.start()].count('\n') + 1
        })
    
    return sections

def perform_semantic_analysis(content: str, variables: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
    """Perform semantic analysis of the GNN model."""
    analysis = {
        "model_type": "unknown",
        "complexity_level": "unknown",
        "domain": "unknown",
        "key_concepts": [],
        "potential_issues": []
    }
    
    # Determine model type based on content
    content_lower = content.lower()
    
    if any(term in content_lower for term in ['pomdp', 'partially observable', 'belief state']):
        analysis["model_type"] = "POMDP"
    elif any(term in content_lower for term in ['mdp', 'markov decision process']):
        analysis["model_type"] = "MDP"
    elif any(term in content_lower for term in ['active inference', 'free energy']):
        analysis["model_type"] = "Active Inference"
    
    # Determine complexity level
    var_count = len(variables)
    conn_count = len(connections)
    
    if var_count < 5 and conn_count < 5:
        analysis["complexity_level"] = "simple"
    elif var_count < 15 and conn_count < 15:
        analysis["complexity_level"] = "moderate"
    else:
        analysis["complexity_level"] = "complex"
    
    # Extract key concepts
    key_terms = ['state', 'action', 'observation', 'reward', 'transition', 'policy', 'belief']
    analysis["key_concepts"] = [term for term in key_terms if term in content_lower]
    
    return analysis

def calculate_complexity_metrics(variables: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
    """Calculate complexity metrics for the model."""
    return {
        "variable_count": len(variables),
        "connection_count": len(connections),
        "connectivity_ratio": len(connections) / max(len(variables), 1),
        "complexity_score": len(variables) * len(connections) / 100.0,
        "estimated_computation_time": len(variables) * len(connections) * 0.1,  # rough estimate
        "memory_requirements": len(variables) * 8  # rough estimate in bytes
    }

def identify_patterns(content: str, variables: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
    """Identify common patterns and anti-patterns."""
    patterns = {
        "design_patterns": [],
        "anti_patterns": [],
        "recommendations": []
    }
    
    # Check for common design patterns
    if len(variables) > 0 and len(connections) > 0:
        patterns["design_patterns"].append("Structured Model")
    
    if any('state' in var.get('name', '').lower() for var in variables):
        patterns["design_patterns"].append("State-Based Design")
    
    if any('action' in var.get('name', '').lower() for var in variables):
        patterns["design_patterns"].append("Action-Oriented Design")
    
    # Check for anti-patterns
    if len(variables) > 20:
        patterns["anti_patterns"].append("High Complexity")
        patterns["recommendations"].append("Consider breaking into smaller modules")
    
    if len(connections) > len(variables) * 2:
        patterns["anti_patterns"].append("High Coupling")
        patterns["recommendations"].append("Reduce inter-module dependencies")
    
    return patterns

def generate_model_insights(file_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights about the model."""
    insights = {
        "file_path": file_analysis["file_path"],
        "model_summary": "",
        "strengths": [],
        "weaknesses": [],
        "improvement_opportunities": []
    }
    
    # Generate model summary
    model_type = file_analysis["semantic_analysis"]["model_type"]
    complexity = file_analysis["semantic_analysis"]["complexity_level"]
    var_count = file_analysis["complexity_metrics"]["variable_count"]
    
    insights["model_summary"] = f"This is a {complexity} {model_type} model with {var_count} variables."
    
    # Identify strengths
    if var_count > 0:
        insights["strengths"].append("Well-defined variable structure")
    
    if file_analysis["complexity_metrics"]["connectivity_ratio"] < 1.0:
        insights["strengths"].append("Reasonable coupling between components")
    
    # Identify weaknesses
    if var_count == 0:
        insights["weaknesses"].append("No variables defined")
    
    if file_analysis["complexity_metrics"]["complexity_score"] > 10:
        insights["weaknesses"].append("High complexity may impact maintainability")
    
    # Generate improvement opportunities
    if file_analysis["complexity_metrics"]["variable_count"] > 15:
        insights["improvement_opportunities"].append("Consider modularization")
    
    return insights

def generate_code_suggestions(file_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate code suggestions for the model."""
    suggestions = {
        "file_path": file_analysis["file_path"],
        "implementation_suggestions": [],
        "optimization_opportunities": [],
        "testing_recommendations": []
    }
    
    # Implementation suggestions
    model_type = file_analysis["semantic_analysis"]["model_type"]
    if model_type == "POMDP":
        suggestions["implementation_suggestions"].append("Use PyMDP library for POMDP implementation")
        suggestions["implementation_suggestions"].append("Implement belief state updates")
    
    elif model_type == "Active Inference":
        suggestions["implementation_suggestions"].append("Use ActiveInference.jl for Julia implementation")
        suggestions["implementation_suggestions"].append("Implement free energy minimization")
    
    # Optimization opportunities
    if file_analysis["complexity_metrics"]["complexity_score"] > 5:
        suggestions["optimization_opportunities"].append("Consider vectorization for performance")
        suggestions["optimization_opportunities"].append("Implement caching for repeated computations")
    
    # Testing recommendations
    suggestions["testing_recommendations"].append("Create unit tests for each variable")
    suggestions["testing_recommendations"].append("Implement integration tests for connections")
    
    return suggestions

def generate_documentation(file_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate documentation for the model."""
    docs = {
        "file_path": file_analysis["file_path"],
        "model_description": "",
        "api_documentation": [],
        "usage_examples": [],
        "troubleshooting": []
    }
    
    # Generate model description
    model_type = file_analysis["semantic_analysis"]["model_type"]
    complexity = file_analysis["semantic_analysis"]["complexity_level"]
    var_count = file_analysis["complexity_metrics"]["variable_count"]
    
    docs["model_description"] = f"""
# {file_analysis['file_name']}

This is a {complexity} {model_type} model with {var_count} variables.

## Model Overview
- **Type**: {model_type}
- **Complexity**: {complexity}
- **Variables**: {var_count}
- **Connections**: {file_analysis['complexity_metrics']['connection_count']}

## Key Components
"""
    
    # Add variable documentation
    for var in file_analysis["variables"][:5]:  # Limit to first 5
        docs["api_documentation"].append(f"- `{var['name']}`: Variable defined at line {var['line']}")
    
    # Add usage examples
    docs["usage_examples"].append("```python")
    docs["usage_examples"].append("# Example usage will be generated here")
    docs["usage_examples"].append("```")
    
    return docs

def generate_llm_summary(results: Dict[str, Any]) -> str:
    """Generate a markdown summary of LLM processing results."""
    summary = f"""# LLM Processing Summary

Generated on: {results['timestamp']}

## Overview
- **Files Processed**: {results['processed_files']}
- **Success**: {results['success']}
- **Errors**: {len(results['errors'])}

## Analysis Results
"""
    
    for analysis in results["analysis_results"]:
        summary += f"""
### {analysis['file_name']}
- **Model Type**: {analysis['semantic_analysis']['model_type']}
- **Complexity**: {analysis['semantic_analysis']['complexity_level']}
- **Variables**: {analysis['complexity_metrics']['variable_count']}
- **Connections**: {analysis['complexity_metrics']['connection_count']}
"""
    
    if results["errors"]:
        summary += "\n## Errors\n"
        for error in results["errors"]:
            summary += f"- {error}\n"
    
    return summary

# Module metadata
__version__ = "1.0.0"
__author__ = "Active Inference Institute"
__description__ = "LLM-enhanced analysis for GNN Processing Pipeline"

# Feature availability flags
FEATURES = {
    'semantic_analysis': True,
    'complexity_metrics': True,
    'pattern_recognition': True,
    'code_suggestions': True,
    'documentation_generation': True
}

__all__ = [
    'process_llm',
    'analyze_gnn_file_with_llm',
    'generate_model_insights',
    'generate_code_suggestions',
    'generate_documentation',
    'FEATURES',
    '__version__'
]
