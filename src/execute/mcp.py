"""
Model Context Protocol (MCP) integration for the 'execute' module.

This file exposes execution functionalities via MCP, including simulation management,
execution monitoring, and result analysis for GNN-generated simulators.
"""

import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

def execute_simulation_mcp(simulation_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a simulation script and return results.
    
    Args:
        simulation_path: Path to the simulation script to execute
        options: Optional execution parameters
        
    Returns:
        Dictionary containing execution results and metadata
    """
    try:
        sim_path = Path(simulation_path)
        if not sim_path.exists():
            return {
                "success": False,
                "error": f"Simulation file not found: {simulation_path}",
                "simulation_path": simulation_path
            }
        
        # Determine execution method based on file type
        if sim_path.suffix == '.py':
            cmd = ['python', str(sim_path)]
        elif sim_path.suffix == '.jl':
            cmd = ['julia', str(sim_path)]
        elif sim_path.suffix == '.sh':
            cmd = ['bash', str(sim_path)]
        else:
            return {
                "success": False,
                "error": f"Unsupported simulation file type: {sim_path.suffix}",
                "simulation_path": simulation_path
            }
        
        # Add options as environment variables or command line arguments
        env = {}
        if options:
            for key, value in options.items():
                env[f"SIM_{key.upper()}"] = str(value)
        
        # Execute simulation with timeout
        timeout = options.get('timeout', 300) if options else 300
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=sim_path.parent
        )
        
        execution_time = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "simulation_path": simulation_path,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": ' '.join(cmd),
            "options": options or {}
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Simulation timed out after {timeout} seconds",
            "simulation_path": simulation_path,
            "execution_time": timeout
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "simulation_path": simulation_path
        }

def list_available_simulations_mcp(directory_path: str, recursive: bool = False) -> Dict[str, Any]:
    """
    List all available simulation scripts in a directory.
    
    Args:
        directory_path: Directory to search for simulations
        recursive: Whether to search recursively
        
    Returns:
        Dictionary containing list of available simulations
    """
    try:
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}",
                "directory_path": directory_path
            }
        
        # Search for simulation files
        patterns = ['*.py', '*.jl', '*.sh']
        simulations = []
        
        for pattern in patterns:
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))
            
            for file_path in files:
                # Check if file looks like a simulation (has execution permissions or contains simulation keywords)
                try:
                    content = file_path.read_text(encoding='utf-8')[:1000]  # First 1000 chars
                    is_simulation = any(keyword in content.lower() for keyword in [
                        'simulation', 'simulate', 'run', 'main', 'if __name__', 'function main'
                    ])
                    
                    simulations.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "type": file_path.suffix,
                        "size": file_path.stat().st_size,
                        "is_simulation": is_simulation
                    })
                except Exception:
                    # If we can't read the file, still include it
                    simulations.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "type": file_path.suffix,
                        "size": file_path.stat().st_size,
                        "is_simulation": False
                    })
        
        return {
            "success": True,
            "directory_path": directory_path,
            "simulations": simulations,
            "total_count": len(simulations),
            "simulation_count": sum(1 for s in simulations if s["is_simulation"])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "directory_path": directory_path
        }

def analyze_execution_results_mcp(results_file: str) -> Dict[str, Any]:
    """
    Analyze execution results from a results file.
    
    Args:
        results_file: Path to the results file to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        results_path = Path(results_file)
        if not results_path.exists():
            return {
                "success": False,
                "error": f"Results file not found: {results_file}",
                "results_file": results_file
            }
        
        # Try to parse as JSON first
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Analyze JSON results
            analysis = {
                "file_type": "json",
                "file_size": results_path.stat().st_size,
                "data_type": type(data).__name__,
                "structure": _analyze_data_structure(data)
            }
            
            # Extract key metrics if available
            if isinstance(data, dict):
                analysis["metrics"] = _extract_metrics_from_dict(data)
            
        except json.JSONDecodeError:
            # Try to analyze as text
            with open(results_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                "file_type": "text",
                "file_size": results_path.stat().st_size,
                "line_count": len(content.splitlines()),
                "word_count": len(content.split()),
                "content_preview": content[:500] + "..." if len(content) > 500 else content
            }
        
        return {
            "success": True,
            "results_file": results_file,
            "analysis": analysis
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results_file": results_file
        }

def _analyze_data_structure(data: Any, max_depth: int = 3) -> Dict[str, Any]:
    """Analyze the structure of data recursively."""
    if max_depth <= 0:
        return {"type": "max_depth_reached"}
    
    if isinstance(data, dict):
        return {
            "type": "dict",
            "keys": list(data.keys()),
            "key_count": len(data),
            "sample_values": {k: type(v).__name__ for k, v in list(data.items())[:5]}
        }
    elif isinstance(data, list):
        return {
            "type": "list",
            "length": len(data),
            "sample_types": [type(item).__name__ for item in data[:5]]
        }
    else:
        return {"type": type(data).__name__}

def _extract_metrics_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common metrics from a results dictionary."""
    metrics = {}
    
    # Look for common metric keys
    metric_keys = ['accuracy', 'loss', 'error', 'success', 'time', 'duration', 'performance']
    
    for key in metric_keys:
        if key in data:
            metrics[key] = data[key]
    
    # Look for nested metrics
    for key, value in data.items():
        if isinstance(value, dict):
            nested_metrics = _extract_metrics_from_dict(value)
            if nested_metrics:
                metrics[f"{key}_metrics"] = nested_metrics
    
    return metrics

def register_tools(mcp_instance):
    """
    Registers MCP tools related to execution tasks with the MCP instance.

    Args:
        mcp_instance: The main MCP instance to register tools with.
    """
    logger.info("Registering execution tools with MCP...")
    
    # Register simulation execution tool
    mcp_instance.register_tool(
        "execute_simulation",
        execute_simulation_mcp,
        {
            "simulation_path": {"type": "string", "description": "Path to the simulation script to execute"},
            "options": {"type": "object", "description": "Optional execution parameters", "optional": True}
        },
        "Execute a simulation script and return detailed results including execution time, return code, and output."
    )
    
    # Register simulation discovery tool
    mcp_instance.register_tool(
        "list_simulations",
        list_available_simulations_mcp,
        {
            "directory_path": {"type": "string", "description": "Directory to search for simulations"},
            "recursive": {"type": "boolean", "description": "Search recursively in subdirectories", "optional": True}
        },
        "List all available simulation scripts in a directory with metadata about each simulation."
    )
    
    # Register results analysis tool
    mcp_instance.register_tool(
        "analyze_results",
        analyze_execution_results_mcp,
        {
            "results_file": {"type": "string", "description": "Path to the results file to analyze"}
        },
        "Analyze execution results from a file, providing insights into performance, structure, and key metrics."
    )
    
    logger.info("Successfully registered 3 execution tools with MCP")

if __name__ == '__main__':
    # Test the tools if run directly
    logger.info("Testing execution tools...")
    
    # Test simulation listing
    test_dir = Path(__file__).parent.parent / "output"
    if test_dir.exists():
        result = list_available_simulations_mcp(str(test_dir))
        logger.info(f"Found {result.get('total_count', 0)} potential simulation files")
    
    logger.info("Execution tools test complete.") 