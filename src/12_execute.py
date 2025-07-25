#!/usr/bin/env python3
"""
Step 12: Execute Processing

This step executes rendered simulation code from step 11, with special focus on
PyMDP simulations that use GNN-extracted state space parameters.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.pipeline_template import (
    setup_step_logging,
    log_step_start,
    log_step_success,
    log_step_error
)
from utils.argument_utils import EnhancedArgumentParser
from pipeline.config import get_output_dir_for_script, get_pipeline_config

def main():
    """Main execute processing function."""
    args = EnhancedArgumentParser.parse_step_arguments("12_execute")
    
    # Setup logging
    logger = setup_step_logging("execute", args)
    
    try:
        # Get pipeline configuration
        config = get_pipeline_config()
        output_dir = get_output_dir_for_script("12_execute.py", Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        log_step_start(logger, "Processing execute")
        
        # Load GNN processing results to get full specifications
        gnn_output_dir = get_output_dir_for_script("3_gnn.py", Path(args.output_dir))
        gnn_results_file = gnn_output_dir / "gnn_processing_results.json"
        
        gnn_specs = {}
        if gnn_results_file.exists():
            logger.info("Loading GNN specifications for enhanced execution")
            with open(gnn_results_file, 'r') as f:
                gnn_results = json.load(f)
            
            # Build map of file names to GNN specifications
            for file_result in gnn_results.get("processed_files", []):
                if file_result.get("parse_success"):
                    file_name = file_result["file_name"]
                    gnn_specs[file_name] = file_result
                    logger.info(f"Loaded GNN spec for {file_name}")
        else:
            logger.warning("No GNN processing results found - execution will use basic parameters")
        
        # Check for rendered PyMDP scripts from step 11
        render_output_dir = get_output_dir_for_script("11_render.py", Path(args.output_dir))
        pymdp_scripts = list(render_output_dir.glob("**/*_pymdp*.py"))
        
        execution_results = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(args.target_dir),
            "output_directory": str(output_dir),
            "executions": [],
            "summary": {
                "total_scripts": len(pymdp_scripts),
                "successful_executions": 0,
                "failed_executions": 0,
                "gnn_based_executions": 0
            }
        }
        
        success = False
        
        if pymdp_scripts:
            logger.info(f"Found {len(pymdp_scripts)} PyMDP scripts to execute")
            
            try:
                from execute.pymdp import execute_pymdp_simulation_from_gnn
                pymdp_available = True
            except ImportError:
                logger.warning("PyMDP pipeline module not available, using basic execution")
                pymdp_available = False
            
            for script_path in pymdp_scripts:
                execution_result = {
                    "script_path": str(script_path),
                    "script_name": script_path.name,
                    "execution_success": False,
                    "gnn_based": False,
                    "output_directory": None,
                    "error": None
                }
                
                try:
                    logger.info(f"Executing PyMDP script: {script_path.name}")
                    
                    # Try to find corresponding GNN specification
                    corresponding_gnn = None
                    for gnn_file, gnn_spec in gnn_specs.items():
                        if gnn_file.replace('.md', '') in script_path.name:
                            corresponding_gnn = gnn_spec
                            break
                    
                    if corresponding_gnn and pymdp_available:
                        # Enhanced GNN-based execution
                        logger.info(f"Using GNN specification from {corresponding_gnn['file_name']}")
                        
                        # Create execution output directory  
                        exec_output_dir = output_dir / f"gnn_pymdp_{script_path.stem}"
                        exec_output_dir.mkdir(exist_ok=True)
                        
                        # Configuration for GNN-based execution
                        config_overrides = {
                            'num_episodes': 20,
                            'max_steps_per_episode': 30,
                            'planning_horizon': 3,
                            'verbose_output': True,
                            'save_visualizations': True,
                            'random_seed': 42,
                            'use_gnn_matrices': True
                        }
                        
                        # Execute using pipeline PyMDP module with full GNN spec
                        exec_success, results = execute_pymdp_simulation_from_gnn(
                            gnn_spec=corresponding_gnn,
                            output_dir=exec_output_dir,
                            config_overrides=config_overrides
                        )
                        
                        if exec_success:
                            execution_result["execution_success"] = True
                            execution_result["gnn_based"] = True
                            execution_result["output_directory"] = str(exec_output_dir)
                            execution_result["gnn_matrices_used"] = results.get('gnn_matrices_used', [])
                            execution_result["simulation_metrics"] = {
                                "total_episodes": results.get('total_episodes'),
                                "success_rate": results.get('success_rate'),
                                "avg_episode_length": results.get('avg_episode_length')
                            }
                            
                            execution_results["summary"]["successful_executions"] += 1
                            execution_results["summary"]["gnn_based_executions"] += 1
                            logger.info(f"✓ GNN-based execution successful: {script_path.name}")
                            logger.info(f"  Used matrices: {results.get('gnn_matrices_used', [])}")
                            logger.info(f"  Success rate: {results.get('success_rate', 0):.2%}")
                        else:
                            execution_result["error"] = results.get('error', 'Unknown error')
                            execution_results["summary"]["failed_executions"] += 1
                            logger.error(f"✗ GNN-based execution failed: {script_path.name}")
                    
                    elif pymdp_available:
                        # Basic PyMDP execution without GNN matrices
                        logger.info("Using basic PyMDP execution (no GNN spec available)")
                        
                        exec_output_dir = output_dir / f"basic_pymdp_{script_path.stem}"
                        exec_output_dir.mkdir(exist_ok=True)
                        
                        # Create minimal spec for basic execution
                        basic_spec = {
                            'model_name': script_path.stem,
                            'model_parameters': {
                                'num_hidden_states': 3,
                                'num_obs': 3,
                                'num_actions': 3
                            }
                        }
                        
                        from execute.pymdp import execute_pymdp_simulation
                        exec_success, results = execute_pymdp_simulation(
                            gnn_spec=basic_spec,
                            output_dir=exec_output_dir
                        )
                        
                        if exec_success:
                            execution_result["execution_success"] = True
                            execution_result["output_directory"] = str(exec_output_dir)
                            execution_results["summary"]["successful_executions"] += 1
                            logger.info(f"✓ Basic execution successful: {script_path.name}")
                        else:
                            execution_result["error"] = results.get('error', 'Unknown error')
                            execution_results["summary"]["failed_executions"] += 1
                            logger.error(f"✗ Basic execution failed: {script_path.name}")
                    
                    else:
                        # Fallback to script execution
                        logger.info("Executing script directly (PyMDP module not available)")
                        
                        exec_output_dir = output_dir / f"script_exec_{script_path.stem}"
                        exec_output_dir.mkdir(exist_ok=True)
                        
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, str(script_path)],
                            cwd=exec_output_dir,
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minute timeout
                        )
                        
                        if result.returncode == 0:
                            execution_result["execution_success"] = True
                            execution_result["output_directory"] = str(exec_output_dir)
                            execution_results["summary"]["successful_executions"] += 1
                            logger.info(f"✓ Script execution successful: {script_path.name}")
                        else:
                            execution_result["error"] = f"Script failed with code {result.returncode}: {result.stderr}"
                            execution_results["summary"]["failed_executions"] += 1
                            logger.error(f"✗ Script execution failed: {script_path.name}")
                        
                except Exception as e:
                    execution_result["error"] = str(e)
                    execution_results["summary"]["failed_executions"] += 1
                    logger.error(f"Error executing {script_path}: {e}")
                
                execution_results["executions"].append(execution_result)
            
            success = execution_results["summary"]["successful_executions"] > 0
            
            # Save execution results
            results_file = output_dir / "execution_results.json"
            with open(results_file, 'w') as f:
                json.dump(execution_results, f, indent=2, default=str)
            
            if success:
                successful = execution_results["summary"]["successful_executions"]
                gnn_based = execution_results["summary"]["gnn_based_executions"]
                log_step_success(logger, f"Executed {successful} simulations ({gnn_based} using GNN matrices)")
            else:
                log_step_error(logger, "No simulations executed successfully")
        else:
            logger.info("No PyMDP scripts found for execution")
            success = True  # Don't fail if no scripts to execute
        
        return 0 if success else 1
            
    except Exception as e:
        log_step_error(logger, "Execute processing failed", {"error": str(e)})
        return 1

if __name__ == "__main__":
    sys.exit(main())
