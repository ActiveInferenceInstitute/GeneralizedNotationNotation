#!/usr/bin/env python3
"""
Integration Test for GNN-PyMDP Pipeline

Tests the integration between GNN parsing and PyMDP simulation execution.
This test verifies that GNN specifications can be properly parsed and used
to configure PyMDP simulations.

Features:
- GNN file parsing validation
- Parameter extraction testing
- PyMDP simulation configuration testing
- End-to-end pipeline integration testing

Author: GNN PyMDP Integration
Date: 2024
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.gnn.parsers.markdown_parser import MarkdownGNNParser
from src.render.pymdp.pymdp_renderer import PyMDPRenderer
from .pymdp_simulation import PyMDPSimulation
from .pymdp_utils import format_duration


def test_gnn_parsing():
    """Test GNN file parsing and parameter extraction."""
    print("\n=== Testing GNN File Parsing ===")
    
    # Test with the example GNN file
    gnn_file = project_root / "input" / "gnn_files" / "actinf_pomdp_agent.md"
    
    if not gnn_file.exists():
        print(f"❌ GNN file not found: {gnn_file}")
        return False
        
    try:
        parser = MarkdownGNNParser()
        parsed_data = parser.parse_file(gnn_file)
        
        print(f"✅ Successfully parsed GNN file: {gnn_file.name}")
        print(f"   - Found {len(parsed_data.get('sections', []))} sections")
        
        # Check for POMDP-specific sections
        pomdp_sections = [s for s in parsed_data.get('sections', []) 
                         if 'pomdp' in s.get('title', '').lower() or 
                            'state' in s.get('title', '').lower()]
        print(f"   - Found {len(pomdp_sections)} POMDP-related sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to parse GNN file: {e}")
        return False


def test_pymdp_renderer():
    """Test PyMDP renderer configuration extraction."""
    print("\n=== Testing PyMDP Renderer ===")
    
    gnn_file = project_root / "input" / "gnn_files" / "actinf_pomdp_agent.md"
    
    if not gnn_file.exists():
        print(f"❌ GNN file not found: {gnn_file}")
        return False
        
    try:
        renderer = PyMDPRenderer()
        config = renderer.extract_pomdp_config(gnn_file)
        
        print(f"✅ Successfully extracted POMDP configuration")
        print(f"   - Number of states: {config.get('num_states', 'Not specified')}")
        print(f"   - Number of observations: {config.get('num_observations', 'Not specified')}")
        print(f"   - Number of actions: {config.get('num_actions', 'Not specified')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract POMDP configuration: {e}")
        return False


def test_pymdp_simulation():
    """Test PyMDP simulation with GNN-derived parameters."""
    print("\n=== Testing PyMDP Simulation ===")
    
    try:
        # Create a simple test configuration
        config = {
            'num_states': 9,  # 3x3 grid
            'num_observations': 9,  # One observation per state
            'num_actions': 4,  # Up, Down, Left, Right
            'num_timesteps': 10,
            'learning_rate': 0.1,
            'action_precision': 2.0
        }
        
        simulation = PyMDPSimulation(config)
        
        print(f"✅ Successfully created PyMDP simulation")
        print(f"   - State space: {config['num_states']}")
        print(f"   - Action space: {config['num_actions']}")
        print(f"   - Observation space: {config['num_observations']}")
        
        # Test running a short simulation
        start_time = time.time()
        results = simulation.run()
        duration = time.time() - start_time
        
        print(f"✅ Successfully ran simulation in {format_duration(duration)}")
        print(f"   - Generated {len(results.get('states', []))} state samples")
        print(f"   - Generated {len(results.get('actions', []))} action samples")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  PyMDP not available: {e}")
        print("   Install with: pip install inferactively-pymdp")
        return False
        
    except Exception as e:
        print(f"❌ Failed to run PyMDP simulation: {e}")
        return False


def test_full_integration():
    """Test full GNN-to-PyMDP integration pipeline."""
    print("\n=== Testing Full Integration ===")
    
    gnn_file = project_root / "input" / "gnn_files" / "actinf_pomdp_agent.md"
    
    if not gnn_file.exists():
        print(f"❌ GNN file not found: {gnn_file}")
        return False
        
    try:
        # Step 1: Parse GNN file
        renderer = PyMDPRenderer()
        config = renderer.extract_pomdp_config(gnn_file)
        
        # Step 2: Create and run simulation
        simulation = PyMDPSimulation(config)
        results = simulation.run()
        
        print(f"✅ Full integration test successful")
        print(f"   - Parsed GNN specification")
        print(f"   - Configured PyMDP simulation")
        print(f"   - Generated {len(results.get('states', []))} simulation steps")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  PyMDP not available: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🧪 GNN-PyMDP Integration Tests")
    print("=" * 50)
    
    tests = [
        ("GNN Parsing", test_gnn_parsing),
        ("PyMDP Renderer", test_pymdp_renderer),
        ("PyMDP Simulation", test_pymdp_simulation),
        ("Full Integration", test_full_integration)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    duration = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"📊 Integration Test Summary")
    print(f"   Tests passed: {passed}/{total}")
    print(f"   Duration: {format_duration(duration)}")
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} integration tests failed.")
        return 1


if __name__ == "__main__":
    exit(main()) 