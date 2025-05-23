#!/usr/bin/env python3
"""
Simple test script to verify PyMDP converter fixes.
"""

import sys
import os
sys.path.append('src')

from src.tests.render.test_pymdp_converter import create_basic_gnn_spec
from src.render.pymdp_converter import GnnToPyMdpConverter

def test_a_matrix():
    """Test A matrix conversion."""
    print("Testing A matrix conversion...")
    gnn_spec = create_basic_gnn_spec(
        obs_modality_names=['Visual'], 
        num_obs_modalities=[2], 
        hidden_state_factor_names=['Location'], 
        num_hidden_states_factors=[3], 
        A_spec={'Visual': 'np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])'} 
    )
    converter = GnnToPyMdpConverter(gnn_spec)
    a_matrix_str = converter.convert_A_matrix()
    
    expected = 'A_Visual = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])'
    passed = expected.replace(' ', '') in a_matrix_str.replace(' ', '')
    print(f"  A matrix test: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got: {a_matrix_str}")
    return passed

def test_b_matrix():
    """Test B matrix conversion."""
    print("Testing B matrix conversion...")
    gnn_spec = create_basic_gnn_spec(
        hidden_state_factor_names=['Position'], 
        num_hidden_states_factors=[3],
        num_control_factors=[2], 
        control_action_names_per_factor={0: ['Stay', 'Move']},
        B_spec={'Position': 'np.array([[[1,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,0,1],[1,0,0]]])'} 
    )
    converter = GnnToPyMdpConverter(gnn_spec)
    b_matrix_str = converter.convert_B_matrix()
    
    expected = 'B_Position = np.array([[[1,0,0],[0,1,0],[0,0,1]], [[0,1,0],[0,0,1],[1,0,0]]])'
    passed = expected.replace(' ', '') in b_matrix_str.replace(' ', '')
    print(f"  B matrix test: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got: {b_matrix_str}")
    return passed

def test_c_vector():
    """Test C vector conversion."""
    print("Testing C vector conversion...")
    gnn_spec = create_basic_gnn_spec(
        obs_modality_names=['Reward'], 
        num_obs_modalities=[3], 
        C_spec={'Reward': 'np.array([0.0, 1.0, -1.0])'}
    )
    converter = GnnToPyMdpConverter(gnn_spec)
    c_vector_str = converter.convert_C_vector()
    
    expected = 'C_Reward = np.array([0.0, 1.0, -1.0])'
    passed = expected.replace(' ', '') in c_vector_str.replace(' ', '')
    print(f"  C vector test: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got: {c_vector_str}")
    return passed

def test_d_vector():
    """Test D vector conversion."""
    print("Testing D vector conversion...")
    gnn_spec = create_basic_gnn_spec(
        hidden_state_factor_names=['Belief'], 
        num_hidden_states_factors=[4], 
        D_spec={'Belief': 'np.array([0.1, 0.2, 0.3, 0.4])'}
    )
    converter = GnnToPyMdpConverter(gnn_spec)
    d_vector_str = converter.convert_D_vector()
    
    expected = 'D_Belief = np.array([0.1, 0.2, 0.3, 0.4])'
    passed = expected.replace(' ', '') in d_vector_str.replace(' ', '')
    print(f"  D vector test: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got: {d_vector_str}")
    return passed

def test_e_vector():
    """Test E vector conversion."""
    print("Testing E vector conversion...")
    gnn_spec = create_basic_gnn_spec(
        hidden_state_factor_names=['S1'], 
        num_hidden_states_factors=[2], 
        num_control_factors=[2], 
        control_action_names_per_factor={0: ['a1', 'a2']}, 
        E_spec={'policy_prior': 'np.ones(3) / 3.0'}
    )
    converter = GnnToPyMdpConverter(gnn_spec)
    e_vector_str = converter.convert_E_vector()
    
    expected = 'E_policy_prior = np.ones(3) / 3.0'
    passed = expected.replace(' ', '') in e_vector_str.replace(' ', '')
    print(f"  E vector test: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got: {e_vector_str}")
    return passed

def main():
    """Run all tests."""
    print("Running PyMDP Converter Fix Tests...")
    print("=" * 50)
    
    tests = [
        test_a_matrix,
        test_b_matrix,
        test_c_vector,
        test_d_vector,
        test_e_vector
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ERROR in {test.__name__}: {e}")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! PyMDP converter fixes are working correctly.")
        return 0
    else:
        print("❌ Some tests FAILED. PyMDP converter needs more work.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 