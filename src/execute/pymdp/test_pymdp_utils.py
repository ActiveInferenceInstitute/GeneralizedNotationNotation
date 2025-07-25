#!/usr/bin/env python3
"""
Test script for PyMDP utilities

This script tests the PyMDP utilities for proper numpy serialization and 
GNN integration functionality within the pipeline.
"""

import numpy as np
from pathlib import Path
import tempfile
from .pymdp_utils import (
    convert_numpy_for_json,
    safe_json_dump,
    clean_trace_for_serialization,
    save_simulation_results,
    parse_gnn_matrix_string,
    parse_gnn_vector_string,
    extract_gnn_dimensions,
    validate_gnn_pomdp_structure
)

def test_numpy_serialization():
    """Test numpy type serialization functionality."""
    print("Testing PyMDP numpy serialization utilities...")
    
    # Create test data that mimics PyMDP simulation traces
    test_trace = {
        'episode': 1,
        'true_states': [np.int64(2), np.int64(1), np.int64(0)],
        'observations': [0, 1, 2],
        'actions': [np.float64(0.0), np.float64(1.0)],
        'rewards': [np.float64(-0.10), np.float64(1.00)],
        'beliefs': [
            np.array([0.1, 0.6, 0.3]),
            np.array([0.05, 0.25, 0.7])
        ]
    }
    
    print(f"Original trace sample: {test_trace}")
    
    # Test convert_numpy_for_json function
    print("\n1. Testing convert_numpy_for_json...")
    converted = convert_numpy_for_json(test_trace)
    print(f"Converted trace: {converted}")
    
    # Test that it's now JSON serializable
    try:
        import json
        json_str = json.dumps(converted)
        print("✓ Successfully converted to JSON string")
    except Exception as e:
        print(f"✗ JSON conversion failed: {e}")
        return False
    
    # Test clean_trace_for_serialization
    print("\n2. Testing clean_trace_for_serialization...")
    cleaned = clean_trace_for_serialization(test_trace)
    print(f"Cleaned trace keys: {list(cleaned.keys())}")
    
    try:
        json_str = json.dumps(cleaned)
        print("✓ Successfully cleaned and converted to JSON")
    except Exception as e:
        print(f"✗ Cleaned trace JSON conversion failed: {e}")
        return False
    
    # Test safe_json_dump
    print("\n3. Testing safe_json_dump...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_trace.json"
        
        success = safe_json_dump(test_trace, test_file)
        if success and test_file.exists():
            print(f"✓ Successfully saved to {test_file}")
            print(f"File size: {test_file.stat().st_size} bytes")
            
            # Try to load it back
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            print("✓ Successfully loaded back from JSON")
        else:
            print("✗ Failed to save JSON file")
            return False
    
    print("\n🎉 All numpy serialization tests passed!")
    return True

def test_gnn_parsing():
    """Test GNN parsing utilities."""
    print("\nTesting GNN parsing utilities...")
    
    # Test matrix string parsing
    print("\n1. Testing GNN matrix parsing...")
    matrix_str = "{(0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)}"
    
    try:
        matrix = parse_gnn_matrix_string(matrix_str)
        print(f"✓ Parsed matrix: {matrix}")
        assert matrix.shape == (3, 3), f"Expected shape (3,3), got {matrix.shape}"
    except Exception as e:
        print(f"✗ Matrix parsing failed: {e}")
        return False
    
    # Test vector string parsing
    print("\n2. Testing GNN vector parsing...")
    vector_str = "{(0.33333, 0.33333, 0.33333)}"
    
    try:
        vector = parse_gnn_vector_string(vector_str)
        print(f"✓ Parsed vector: {vector}")
        assert len(vector) == 3, f"Expected length 3, got {len(vector)}"
    except Exception as e:
        print(f"✗ Vector parsing failed: {e}")
        return False
    
    # Test dimension extraction
    print("\n3. Testing GNN dimension extraction...")
    test_gnn_spec = {
        'model_parameters': {
            'num_hidden_states': 3,
            'num_obs': 3,
            'num_actions': 3
        },
        'variables': [
            {'name': 'A', 'dimensions': [3, 3]},
            {'name': 'B', 'dimensions': [3, 3, 3]},
            {'name': 'C', 'dimensions': [3]}
        ]
    }
    
    try:
        dimensions = extract_gnn_dimensions(test_gnn_spec)
        print(f"✓ Extracted dimensions: {dimensions}")
        assert dimensions['num_states'] == 3, "States dimension mismatch"
        assert dimensions['num_observations'] == 3, "Observations dimension mismatch"
        assert dimensions['num_actions'] == 3, "Actions dimension mismatch"
    except Exception as e:
        print(f"✗ Dimension extraction failed: {e}")
        return False
    
    # Test POMDP structure validation
    print("\n4. Testing POMDP structure validation...")
    try:
        validation = validate_gnn_pomdp_structure(test_gnn_spec)
        print(f"✓ Validation result: {validation}")
        assert validation['valid'], "POMDP structure should be valid"
    except Exception as e:
        print(f"✗ POMDP validation failed: {e}")
        return False
    
    print("\n🎉 All GNN parsing tests passed!")
    return True

def test_integration():
    """Test full integration with simulated data."""
    print("\nTesting full integration...")
    
    # Create test simulation results
    test_traces = [
        {
            'episode': 0,
            'true_states': [0, 1, 2],
            'observations': [0, 1, 2],
            'actions': [0, 1],
            'rewards': [-0.1, 1.0],
            'beliefs': [np.array([0.8, 0.2, 0.0]), np.array([0.1, 0.1, 0.8])]
        }
    ]
    
    test_metrics = {
        'episode_rewards': [0.9],
        'episode_lengths': [3],
        'belief_entropies': [1.5],
        'success_rates': [1.0]
    }
    
    test_config = {
        'num_states': 3,
        'num_observations': 3,
        'num_actions': 3,
        'model_name': 'TestModel'
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            save_results = save_simulation_results(
                traces=test_traces,
                metrics=test_metrics,
                config=test_config,
                model_matrices=None,
                output_dir=temp_path
            )
            
            print(f"Save results: {save_results}")
            
            # Check all files were created
            expected_files = [
                'simulation_config.json',
                'performance_metrics.json',
                'simulation_traces.pkl',
                'simulation_traces.json'
            ]
            
            all_success = True
            for filename in expected_files:
                filepath = temp_path / filename
                if filepath.exists():
                    print(f"✓ {filename} created ({filepath.stat().st_size} bytes)")
                else:
                    print(f"✗ {filename} missing")
                    all_success = False
            
            if not all_success:
                return False
                
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            return False
    
    print("\n🎉 All integration tests passed!")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("PYMDP UTILITIES TEST SUITE")
    print("=" * 60)
    
    success = True
    
    success &= test_numpy_serialization()
    success &= test_gnn_parsing()
    success &= test_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 