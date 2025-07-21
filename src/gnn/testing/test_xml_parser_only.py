#!/usr/bin/env python3
"""
Test to isolate the XML parser duplication issue.
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.parsers.xml_parser import XMLGNNParser

def test_xml_parser_only():
    """Test XML parser with minimal embedded data."""
    print("🔍 Testing XML parser with minimal embedded data...")
    
    # Create minimal embedded data
    minimal_data = {
        'model_name': 'Test Model',
        'annotation': 'Test annotation',
        'variables': [
            {'name': 'A', 'var_type': 'likelihood_matrix', 'data_type': 'float', 'dimensions': [3, 3]},
            {'name': 'B', 'var_type': 'transition_matrix', 'data_type': 'float', 'dimensions': [3, 3]},
        ],
        'connections': [
            {'source_variables': ['A'], 'target_variables': ['B'], 'connection_type': 'directed'},
        ],
        'parameters': [
            {'name': 'param1', 'value': 'value1'},
            {'name': 'param2', 'value': 'value2'},
        ],
        'equations': [],
        'time_specification': None,
        'ontology_mappings': []
    }
    
    print(f"📊 Input data:")
    print(f"   Variables: {len(minimal_data['variables'])}")
    print(f"   Connections: {len(minimal_data['connections'])}")
    print(f"   Parameters: {len(minimal_data['parameters'])}")
    
    # Create XML content with embedded data
    xml_content = f"""<?xml version="1.0" ?>
<gnn_model name="Test Model" version="1.0">
  <metadata>
    <annotation>Test annotation</annotation>
  </metadata>
  <variables>
    <variable name="A" type="likelihood_matrix" data_type="float" dimensions="3,3"/>
    <variable name="B" type="transition_matrix" data_type="float" dimensions="3,3"/>
  </variables>
  <connections>
    <connection type="directed">
      <sources>A</sources>
      <targets>B</targets>
    </connection>
  </connections>
  <parameters>
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
  </parameters>
  <!-- MODEL_DATA: {json.dumps(minimal_data, separators=(',', ':'))} -->
</gnn_model>"""
    
    print(f"\n📄 Created XML content ({len(xml_content)} characters)")
    
    # Parse with XML parser
    xml_parser = XMLGNNParser()
    parsed_result = xml_parser.parse_string(xml_content)
    
    if not parsed_result.success:
        print(f"❌ Failed to parse XML: {parsed_result.errors}")
        return
    
    model = parsed_result.model
    print(f"\n✅ Parsed model: {model.model_name}")
    print(f"   └─ Variables: {len(model.variables)}")
    print(f"   └─ Connections: {len(model.connections)}")
    print(f"   └─ Parameters: {len(model.parameters)}")
    
    # Show variable names
    print(f"\n📊 Variable names:")
    for i, var in enumerate(model.variables):
        print(f"   {i+1:2d}. {var.name}")
    
    # Show parameter names
    print(f"\n📊 Parameter names:")
    for i, param in enumerate(model.parameters):
        print(f"   {i+1:2d}. {param.name}")
    
    # Check for duplicates
    var_names = [var.name for var in model.variables]
    param_names = [param.name for param in model.parameters]
    
    print(f"\n🔍 Checking for duplicates:")
    print(f"   Variables: {len(var_names)} total, {len(set(var_names))} unique")
    print(f"   Parameters: {len(param_names)} total, {len(set(param_names))} unique")
    
    if len(var_names) != len(set(var_names)):
        print(f"   ❌ DUPLICATE VARIABLES!")
        from collections import Counter
        var_counts = Counter(var_names)
        for name, count in var_counts.items():
            if count > 1:
                print(f"      '{name}' appears {count} times")
    
    if len(param_names) != len(set(param_names)):
        print(f"   ❌ DUPLICATE PARAMETERS!")
        from collections import Counter
        param_counts = Counter(param_names)
        for name, count in param_counts.items():
            if count > 1:
                print(f"      '{name}' appears {count} times")

if __name__ == "__main__":
    test_xml_parser_only() 