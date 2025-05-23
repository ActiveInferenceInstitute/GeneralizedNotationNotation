#!/usr/bin/env python3
import sys
sys.path.append('src')

from src.tests.render.test_pymdp_converter import create_basic_gnn_spec
from src.render.pymdp_converter import GnnToPyMdpConverter
import numpy as np

# Create the test spec
gnn_spec = create_basic_gnn_spec(
    obs_modality_names=['Reward'], 
    num_obs_modalities=[3], 
    C_spec={'Reward': 'np.array([0.0, 1.0, -1.0])'}
)

print("GNN spec C_spec:", gnn_spec.get('C_spec'))

# Create converter
converter = GnnToPyMdpConverter(gnn_spec)
print("Converter C_spec:", converter.C_spec)

# Test parsing
test_string = 'np.array([0.0, 1.0, -1.0])'
print(f"Testing parsing of: {test_string}")

parsed = converter._parse_string_to_literal(test_string, 'test')
print(f"Parsed result: {parsed}, type: {type(parsed)}")

if parsed is not None:
    try:
        arr = np.array(parsed)
        print(f"Array: {arr}, shape: {arr.shape}")
    except Exception as e:
        print(f"Error creating array: {e}")

# Test the full conversion
print("\nFull C vector conversion:")
c_vector_str = converter.convert_C_vector()
print(c_vector_str) 