#!/usr/bin/env python3
"""
PyMDP Execution Module

Main execution function for PyMDP simulations configured from GNN specifications.
This module provides the pipeline interface for running PyMDP Active Inference
simulations.

Features:
- GNN specification parsing and configuration
- Real PyMDP simulation execution
- Comprehensive output generation
- Pipeline integration support

Author: GNN PyMDP Integration
Date: 2024
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import traceback

from .pymdp_simulation import PyMDPSimulation
from .pymdp_utils import extract_gnn_dimensions, validate_gnn_pomdp_structure

def execute_pymdp_simulation(
    gnn_spec: Dict[str, Any],
    output_dir: Path,
    num_episodes: int = 5,
    verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation configured from GNN specification.
    
    This is the main pipeline interface for PyMDP simulation execution.
    
    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_dir: Directory for simulation outputs
        num_episodes: Number of simulation episodes to run
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (success, results) where results contains output paths and metrics
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate GNN specification
        logger.info("Validating GNN specification for PyMDP simulation...")
        
        validation = validate_gnn_pomdp_structure(gnn_spec)
        if not validation['valid']:
            error_msg = f"Invalid GNN POMDP structure: {validation['errors']}"
            logger.error(error_msg)
            return False, {'error': error_msg, 'validation': validation}
        
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"GNN validation warning: {warning}")
        
        # Extract dimensions and configuration
        logger.info("Extracting POMDP configuration from GNN specification...")
        
        dimensions = extract_gnn_dimensions(gnn_spec)
        logger.info(f"Extracted dimensions: {dimensions}")
        
        # Create simulation configuration
        simulation_config = {
            'model_name': gnn_spec.get('model_name', 'GNN_PyMDP_Simulation'),
            'num_states': dimensions['num_states'],
            'num_observations': dimensions['num_observations'], 
            'num_actions': dimensions['num_actions'],
            'planning_horizon': gnn_spec.get('model_parameters', {}).get('planning_horizon', 3),
            'num_episodes': num_episodes,
            'verbose': verbose
        }
        
        # Add any additional parameters from GNN spec
        model_params = gnn_spec.get('model_parameters', {})
        if 'action_precision' in model_params:
            simulation_config['action_precision'] = model_params['action_precision']
        if 'learning_rate' in model_params:
            simulation_config['learning_rate'] = model_params['learning_rate']
        if 'inference_iterations' in model_params:
            simulation_config['inference_iterations'] = model_params['inference_iterations']
        
        logger.info(f"Simulation configuration: {simulation_config}")
        
        # Create and run simulation
        logger.info("Creating PyMDP simulation instance...")
        
        simulation = PyMDPSimulation(config=simulation_config)
        
        # Configure from GNN specification if available
        if 'initial_matrices' in gnn_spec:
            logger.info("Configuring simulation matrices from GNN specification...")
            simulation.configure_from_gnn(gnn_spec)
        
        logger.info(f"Running PyMDP simulation with {num_episodes} episodes...")
        
        # Run the simulation
        results = simulation.run_simulation(output_dir=output_dir)
        
        if results['success']:
            logger.info(f"✓ PyMDP simulation completed successfully!")
            logger.info(f"Episodes completed: {results.get('episodes_completed', 0)}")
            logger.info(f"Success rate: {results.get('overall_success_rate', 0.0):.1%}")
            logger.info(f"Output directory: {results.get('output_directory')}")
            
            return True, results
        else:
            error_msg = f"PyMDP simulation failed: {results.get('error', 'Unknown error')}"
            logger.error(error_msg)
            return False, {'error': error_msg, 'results': results}
        
    except Exception as e:
        error_msg = f"PyMDP execution failed with exception: {str(e)}"
        logger.error(error_msg)
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        return False, {
            'error': error_msg,
            'exception': str(e),
            'traceback': traceback.format_exc()
        }

def execute_from_gnn_file(
    gnn_file: Path,
    output_dir: Path,
    num_episodes: int = 5,
    verbose: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute PyMDP simulation from GNN file.
    
    Args:
        gnn_file: Path to GNN file
        output_dir: Output directory
        num_episodes: Number of episodes
        verbose: Enable verbose output
        
    Returns:
        Tuple of (success, results)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Parse GNN file
        logger.info(f"Parsing GNN file: {gnn_file}")
        
        # Import GNN parser (defer import to avoid circular dependencies)
        from ...gnn import parse_gnn_file
        
        gnn_spec = parse_gnn_file(gnn_file)
        
        if not gnn_spec:
            error_msg = f"Failed to parse GNN file: {gnn_file}"
            logger.error(error_msg)
            return False, {'error': error_msg}
        
        # Convert to dictionary if needed
        if hasattr(gnn_spec, 'to_dict'):
            gnn_spec_dict = gnn_spec.to_dict()
        else:
            gnn_spec_dict = gnn_spec
        
        # Execute simulation
        return execute_pymdp_simulation(
            gnn_spec=gnn_spec_dict,
            output_dir=output_dir,
            num_episodes=num_episodes,
            verbose=verbose
        )
        
    except Exception as e:
        error_msg = f"Failed to execute from GNN file {gnn_file}: {str(e)}"
        logger.error(error_msg)
        return False, {'error': error_msg, 'exception': str(e)}

def batch_execute_pymdp(
    gnn_specs: list,
    base_output_dir: Path,
    num_episodes: int = 5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute multiple PyMDP simulations in batch.
    
    Args:
        gnn_specs: List of GNN specifications
        base_output_dir: Base output directory
        num_episodes: Episodes per simulation
        verbose: Enable verbose output
        
    Returns:
        Dictionary with batch execution results
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting batch execution of {len(gnn_specs)} PyMDP simulations")
    
    batch_results = {
        'total_simulations': len(gnn_specs),
        'successful_simulations': 0,
        'failed_simulations': 0,
        'results': [],
        'errors': []
    }
    
    for i, gnn_spec in enumerate(gnn_specs):
        try:
            logger.info(f"Executing simulation {i+1}/{len(gnn_specs)}")
            
            # Create simulation-specific output directory
            model_name = gnn_spec.get('model_name', f'simulation_{i+1}')
            sim_output_dir = base_output_dir / model_name
            
            # Execute simulation
            success, results = execute_pymdp_simulation(
                gnn_spec=gnn_spec,
                output_dir=sim_output_dir,
                num_episodes=num_episodes,
                verbose=verbose
            )
            
            if success:
                batch_results['successful_simulations'] += 1
                logger.info(f"✓ Simulation {i+1} completed successfully")
            else:
                batch_results['failed_simulations'] += 1
                logger.error(f"✗ Simulation {i+1} failed: {results.get('error')}")
                batch_results['errors'].append({
                    'simulation_index': i,
                    'model_name': model_name,
                    'error': results.get('error')
                })
            
            batch_results['results'].append({
                'simulation_index': i,
                'model_name': model_name,
                'success': success,
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Exception in simulation {i+1}: {e}")
            batch_results['failed_simulations'] += 1
            batch_results['errors'].append({
                'simulation_index': i,
                'error': str(e),
                'exception': True
            })
    
    success_rate = batch_results['successful_simulations'] / batch_results['total_simulations']
    logger.info(f"Batch execution completed: {success_rate:.1%} success rate")
    logger.info(f"Successful: {batch_results['successful_simulations']}, Failed: {batch_results['failed_simulations']}")
    
    return batch_results

# Main execution function for pipeline integration
def execute_pymdp_step(
    target_dir: Path,
    output_dir: Path,
    num_episodes: int = 5,
    verbose: bool = True
) -> bool:
    """
    Execute PyMDP step in pipeline.
    
    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory
        num_episodes: Episodes per simulation
        verbose: Enable verbose output
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Find GNN files
        gnn_files = list(target_dir.glob("*.md"))
        
        if not gnn_files:
            logger.warning(f"No GNN files found in {target_dir}")
            return True  # Not an error, just nothing to process
        
        logger.info(f"Found {len(gnn_files)} GNN files to process")
        
        # Execute each GNN file
        overall_success = True
        
        for gnn_file in gnn_files:
            logger.info(f"Processing GNN file: {gnn_file}")
            
            # Create file-specific output directory
            file_output_dir = output_dir / gnn_file.stem
            
            success, results = execute_from_gnn_file(
                gnn_file=gnn_file,
                output_dir=file_output_dir,
                num_episodes=num_episodes,
                verbose=verbose
            )
            
            if not success:
                logger.error(f"Failed to process {gnn_file}: {results.get('error')}")
                overall_success = False
            else:
                logger.info(f"✓ Successfully processed {gnn_file}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"PyMDP step execution failed: {e}")
        return False 