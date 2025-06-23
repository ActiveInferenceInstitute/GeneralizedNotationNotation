#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Aware-2: Streamlined GNN-Configurable Meta-Awareness Simulation

Main executable for the complete computational phenomenology implementation
based on Sandved-Smith et al. (2021). This is the streamlined, production-ready
version that exactly replicates the original implementation functionality.

Part of the GeneralizedNotationNotation (GNN) project.
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
  
  # Run specific simulation modes (reproduce figures)
  python run_meta_awareness.py config/meta_awareness_gnn.toml -m figure_7 figure_10 figure_11
  
  # Run with custom settings
  python run_meta_awareness.py config/meta_awareness_gnn.toml -s 42 -l DEBUG
  
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
        print("Meta-Aware-2: Computational Phenomenology of Meta-Awareness")
        print("Based on Sandved-Smith et al. (2021)")
        print("=" * 80)
        print()
        
        print(f"Configuration: {args.config}")
        print(f"Random seed: {args.seed}")
        print(f"Log level: {args.log_level}")
        
        if args.modes:
            print(f"Simulation modes: {', '.join(args.modes)}")
        else:
            print("Running all figure reproduction modes")
        
        print()
        print("Starting simulation...")
        print("-" * 40)
        
        # Run complete analysis
        results = run_simulation_from_config(
            config_path=args.config,
            output_dir=".",  # Use current directory structure
            simulation_modes=args.modes,
            random_seed=args.seed,
            log_level=args.log_level
        )
        
        print("-" * 40)
        print("Simulation completed successfully!")
        print()
        
        # Print summary
        print_results_summary(results)
        
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

def print_results_summary(results: dict):
    """Print a summary of simulation results."""
    
    print("Results Summary:")
    
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
        print(f"    Location: ./figures/")
    
    # Results saved
    print(f"  Results saved to: ./results/")
    print(f"  Logs saved to: ./logs/")
    
    print()
    print("✓ Meta-awareness computational phenomenology simulation complete!")
    print("  Check figures/, results/, and logs/ directories for outputs.")

if __name__ == "__main__":
    main() 