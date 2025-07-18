#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/trim/Documents/GitHub/GeneralizedNotationNotation/src')

from gnn.parsers.schema_parser import ZNotationParser
import inspect

# Create parser
parser = ZNotationParser()

# Check method
print("Method details:")
print(f"  Method: {parser._extract_embedded_json_data}")
print(f"  Type: {type(parser._extract_embedded_json_data)}")

# Get source
try:
    source = inspect.getsource(parser._extract_embedded_json_data)
    print(f"\nMethod source:")
    print(source)
except Exception as e:
    print(f"Could not get source: {e}")

# Test with simple content
test_content = '% MODEL_DATA: {"test": "value"}'
print(f"\nTesting with: {repr(test_content)}")
result = parser._extract_embedded_json_data(test_content)
print(f"Result: {result}")

# Check if the debug print appears
print("\nCalling with debug...")
result2 = parser._extract_embedded_json_data(test_content)
print(f"Result2: {result2}") 