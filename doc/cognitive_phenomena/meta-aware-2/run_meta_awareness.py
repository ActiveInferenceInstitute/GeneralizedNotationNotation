#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Aware-2 Golden Spike Entry Point

Main executable for the GNN-configurable meta-awareness computational 
phenomenology simulation. This is the "golden spike" entry point that
demonstrates the complete pipeline from GNN configuration to results.

Based on Sandved-Smith et al. (2021) computational phenomenology of mental action.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Optional, List

# Add current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from execution.simulation_runner import run_simulation_from_config, SimulationRunner

def main():
    """Main entry point for meta-awareness simulation."""
    
    parser = argparse.ArgumentParser(
        description="Meta-Aware-2: GNN-Configurable Meta-Awareness Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_meta_awareness.py config/meta_awareness_gnn.toml
  
  # Run specific simulation modes
  python run_meta_awareness.py config/meta_awareness_gnn.toml -m figure_10 figure_11
  
  # Run with custom output directory and seed
  python run_meta_awareness.py config/meta_awareness_gnn.toml -o ./results -s 42
  
  # Run in quiet mode
  python run_meta_awareness.py config/meta_awareness_gnn.toml -l ERROR
  
  # Show configuration and exit
  python run_meta_awareness.py config/meta_awareness_gnn.toml --show-config
        """
    )
    
    # Required arguments
    parser.add_argument(
        "config", 
        help="Path to GNN configuration file (TOML format)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o", 
        default="./output",
        help="Output directory for results, figures, and logs (default: ./output)"
    )
    
    parser.add_argument(
        "--modes", "-m", 
        nargs="+",
        help="Simulation modes to run (e.g., figure_7, figure_10, figure_11, default)"
    )
    
    parser.add_argument(
        "--seed", "-s", 
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true", 
        help="Skip saving detailed results"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration and exit"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test suite and exit"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Meta-Aware-2 v1.0.0 (GNN-Configurable Meta-Awareness Simulation)"
    )
    
    args = parser.parse_args()
    
    # Handle special modes
    if args.test:
        run_tests()
        return
    
    if args.show_config:
        show_configuration(args.config)
        return
    
    # Run main simulation
    try:
        print("=" * 80)
        print("Meta-Aware-2: GNN-Configurable Meta-Awareness Simulation")
        print("Based on Sandved-Smith et al. (2021)")
        print("=" * 80)
        print()
        
        print(f"Configuration file: {args.config}")
        print(f"Output directory: {args.output}")
        print(f"Random seed: {args.seed}")
        print(f"Log level: {args.log_level}")
        
        if args.modes:
            print(f"Simulation modes: {', '.join(args.modes)}")
        else:
            print("Simulation modes: All available modes")
        
        print()
        print("Starting simulation...")
        print("-" * 40)
        
        # Run complete analysis
        results = run_simulation_from_config(
            config_path=args.config,
            output_dir=args.output,
            simulation_modes=args.modes,
            random_seed=args.seed,
            log_level=args.log_level
        )
        
        print("-" * 40)
        print("Simulation completed successfully!")
        print()
        
        # Print summary
        print_results_summary(results, args.output)
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during simulation: {e}")
        if args.log_level == "DEBUG":
            traceback.print_exc()
        sys.exit(1)

def show_configuration(config_path: str):
    """Show configuration details and exit."""
    try:
        from config.gnn_parser import load_gnn_config
        
        print("=" * 60)
        print("GNN Configuration Summary")
        print("=" * 60)
        
        config = load_gnn_config(config_path)
        
        print(f"Model Name: {config.name}")
        print(f"Description: {config.description}")
        print(f"Number of Levels: {config.num_levels}")
        print(f"Level Names: {', '.join(config.level_names)}")
        print(f"Time Steps: {config.time_steps}")
        print()
        
        print("Level Configuration:")
        for name, level in config.levels.items():
            print(f"  {name}:")
            print(f"    State Dimension: {level.state_dim}")
            print(f"    Observation Dimension: {level.obs_dim}")
            print(f"    Action Dimension: {level.action_dim}")
        print()
        
        print("Precision Bounds:")
        for level_name, bounds in config.precision_bounds.items():
            print(f"  {level_name}: [{bounds[0]}, {bounds[1]}]")
        print()
        
        print("Available Simulation Modes:")
        for mode_name, mode_type in config.simulation_modes.items():
            print(f"  {mode_name}: {mode_type}")
        print()
        
        print("Configuration loaded successfully!")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def run_tests():
    """Run the test suite."""
    try:
        import unittest
        from tests.test_simulation import TestSimulation, TestValidationAgainstPaper
        
        print("=" * 60)
        print("Running Meta-Aware-2 Test Suite")
        print("=" * 60)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(TestSimulation))
        suite.addTests(loader.loadTestsFromTestCase(TestValidationAgainstPaper))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        print("\n" + "=" * 60)
        if result.wasSuccessful():
            print("All tests passed successfully!")
            sys.exit(0)
        else:
            print(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
            sys.exit(1)
            
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)

def print_results_summary(results: dict, output_dir: str):
    """Print a summary of simulation results."""
    
    print("Results Summary:")
    print(f"  Output directory: {output_dir}")
    
    # Execution info
    exec_info = results.get('execution_info', {})
    if 'total_duration' in exec_info:
        duration = exec_info['total_duration']
        print(f"  Total duration: {duration:.2f} seconds")
    
    # Simulation modes run
    sim_results = results.get('simulation_results', {})
    if sim_results:
        print(f"  Simulation modes completed: {len(sim_results)}")
        for mode in sim_results.keys():
            print(f"    - {mode}")
    
    # Figures generated
    figure_paths = results.get('figure_paths', {})
    if figure_paths:
        total_figures = sum(len(paths) if isinstance(paths, dict) else 1 
                           for paths in figure_paths.values())
        print(f"  Figures generated: {total_figures}")
    
    # Analysis summary
    analysis = results.get('analysis_summary', {})
    if analysis:
        print(f"  Analysis components: {len(analysis)}")
    
    print()
    print("Output Files:")
    output_path = Path(output_dir)
    
    # List key output directories
    for subdir in ['results', 'figures', 'logs']:
        subdir_path = output_path / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.glob('*')))
            print(f"  {subdir}/: {file_count} files")
    
    print()
    print("Simulation completed! Check the output directory for detailed results.")

def create_example_config():
    """Create an example configuration file."""
    example_config = """
# Meta-Aware-2 GNN Configuration Example
# Based on Sandved-Smith et al. (2021)

[model]
name = "meta_awareness_example"
description = "Example meta-awareness model with hierarchical active inference"
num_levels = 3
level_names = ["perception", "attention", "meta_awareness"]
time_steps = 100
oddball_pattern = "default"

[levels.perception]
state_dim = 2
obs_dim = 2  
action_dim = 0

[levels.attention]
state_dim = 2
obs_dim = 2
action_dim = 2

[levels.meta_awareness]
state_dim = 2
obs_dim = 2
action_dim = 2

[precision_bounds]
perception = [0.5, 2.0]
attention = [2.0, 4.0]

[policy_precision]
2_level = 2.0
3_level = 4.0

[simulation_modes]
default = "natural_dynamics"
figure_7 = "fixed_attention_schedule" 
figure_10 = "two_level_mind_wandering"
figure_11 = "three_level_meta_awareness"

[validation_config]
check_matrix_dimensions = true
check_probability_normalization = true
tolerance = 1e-10
"""
    
    config_path = Path("example_config.toml")
    with open(config_path, 'w') as f:
        f.write(example_config)
    
    print(f"Example configuration created: {config_path}")

if __name__ == "__main__":
    main() 