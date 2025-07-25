#!/usr/bin/env python3
"""
Test Runner Helper

This script provides convenient commands for running different test configurations
in the GNN pipeline. It supports all the staging options and provides quick access
to common test scenarios.

Usage:
    python src/tests/test_runner_helper.py --help
    python src/tests/test_runner_helper.py fast
    python src/tests/test_runner_helper.py full
    python src/tests/test_runner_helper.py debug
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test_configuration(config_name: str, target_dir: str = "input/gnn_files", output_dir: str = "output", verbose: bool = False):
    """Run a predefined test configuration."""
    
    # Define test configurations
    configurations = {
        "fast": {
            "args": ["--fast-only"],
            "description": "Run only fast tests (< 3 minutes) for quick validation",
            "expected_duration": "2-3 minutes"
        },
        "standard": {
            "args": [],
            "description": "Run fast and standard tests (recommended for development)",
            "expected_duration": "5-10 minutes"
        },
        "full": {
            "args": ["--include-slow"],
            "description": "Run all tests including slow integration tests",
            "expected_duration": "15-25 minutes"
        },
        "performance": {
            "args": ["--include-performance"],
            "description": "Include performance benchmark tests",
            "expected_duration": "20-30 minutes"
        },
        "debug": {
            "args": ["--verbose"],
            "description": "Run with verbose output for debugging test issues",
            "expected_duration": "5-15 minutes"
        },
        "coverage": {
            "args": ["--verbose"],
            "description": "Run with detailed coverage reporting",
            "expected_duration": "10-15 minutes"
        },
        "minimal": {
            "args": ["--fast-only", "--no-coverage"],
            "description": "Minimal test run with no coverage for maximum speed",
            "expected_duration": "1-2 minutes"
        }
    }
    
    if config_name not in configurations:
        print(f"Unknown configuration: {config_name}")
        print("Available configurations:")
        for name, config in configurations.items():
            print(f"  {name}: {config['description']} ({config['expected_duration']})")
        return False
    
    config = configurations[config_name]
    
    # Build command
    test_script = Path(__file__).parent.parent / "2_tests.py"
    cmd = [
        sys.executable, str(test_script),
        "--target-dir", target_dir,
        "--output-dir", output_dir
    ]
    
    # Add configuration-specific args
    cmd.extend(config["args"])
    
    # Add verbose if requested
    if verbose and "--verbose" not in cmd:
        cmd.append("--verbose")
    
    print(f"🚀 Running test configuration: {config_name}")
    print(f"📋 Description: {config['description']}")
    print(f"⏱️  Expected duration: {config['expected_duration']}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Test configuration '{config_name}' completed successfully in {duration:.1f}s")
        else:
            print(f"⚠️ Test configuration '{config_name}' completed with issues in {duration:.1f}s (exit code: {result.returncode})")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        duration = time.time() - start_time
        print(f"\n🛑 Test execution interrupted after {duration:.1f}s")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Test execution failed after {duration:.1f}s: {e}")
        return False

def list_configurations():
    """List all available test configurations."""
    configurations = {
        "fast": "Quick validation tests (< 3 minutes)",
        "standard": "Fast and standard tests (5-10 minutes)", 
        "full": "All tests including slow integration (15-25 minutes)",
        "performance": "Include performance benchmarks (20-30 minutes)",
        "debug": "Verbose output for debugging (5-15 minutes)",
        "coverage": "Detailed coverage reporting (10-15 minutes)",
        "minimal": "Minimal run with no coverage (1-2 minutes)"
    }
    
    print("Available test configurations:")
    print("=" * 50)
    for name, description in configurations.items():
        print(f"  {name:<12} - {description}")
    print()
    print("Usage examples:")
    print("  python src/tests/test_runner_helper.py fast")
    print("  python src/tests/test_runner_helper.py full --verbose")
    print("  python src/tests/test_runner_helper.py debug --target-dir custom/path")

def run_custom_tests(markers: str = None, timeout: int = None, **kwargs):
    """Run tests with custom configuration."""
    test_script = Path(__file__).parent.parent / "2_tests.py"
    cmd = [sys.executable, str(test_script)]
    
    # Add standard arguments
    for key, value in kwargs.items():
        if value is not None:
            if key.replace('_', '-') in ['target-dir', 'output-dir']:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            elif isinstance(value, bool) and value:
                cmd.append(f"--{key.replace('_', '-')}")
    
    # Add custom markers if specified
    if markers:
        print(f"🎯 Running tests with custom markers: {markers}")
    
    if timeout:
        print(f"⏱️  Custom timeout: {timeout} seconds")
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Custom test execution failed: {e}")
        return False

def main():
    """Main entry point for test runner helper."""
    parser = argparse.ArgumentParser(
        description="GNN Test Runner Helper - Easy access to different test configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s fast                    # Quick validation tests
  %(prog)s standard --verbose     # Standard tests with verbose output  
  %(prog)s full                   # All tests including slow ones
  %(prog)s debug                  # Debug mode with verbose output
  %(prog)s list                   # List all available configurations
        """
    )
    
    parser.add_argument(
        "configuration", 
        nargs="?",
        help="Test configuration to run (fast, standard, full, performance, debug, coverage, minimal, list)"
    )
    parser.add_argument(
        "--target-dir", 
        default="input/gnn_files",
        help="Directory containing GNN files to test (default: input/gnn_files)"
    )
    parser.add_argument(
        "--output-dir",
        default="output", 
        help="Output directory for test results (default: output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--custom-markers",
        help="Custom pytest markers to use (advanced)"
    )
    parser.add_argument(
        "--custom-timeout", 
        type=int,
        help="Custom timeout in seconds (advanced)"
    )
    
    args = parser.parse_args()
    
    if args.configuration == "list" or args.configuration is None:
        list_configurations()
        return 0
    
    # Handle custom configuration
    if args.custom_markers or args.custom_timeout:
        success = run_custom_tests(
            markers=args.custom_markers,
            timeout=args.custom_timeout,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    else:
        # Handle predefined configurations
        success = run_test_configuration(
            args.configuration,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 