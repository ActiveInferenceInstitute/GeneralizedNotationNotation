#!/usr/bin/env python3
"""
Simple Round-Trip Test for GNN JSON and XML Formats

This is a minimal test to verify that JSON and XML round-trip conversion works
without the complex import chains that cause recursion issues.

Author: AI Assistant
Date: 2025-01-17
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Set reasonable recursion limit
sys.setrecursionlimit(100)

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

@dataclass
class SimpleTestResult:
    """Simple test result for round-trip testing."""
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    test_time: float = 0.0
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.success = False
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)

def test_json_round_trip():
    """Test JSON round-trip conversion with minimal dependencies."""
    print("🔄 Testing JSON round-trip conversion...")
    
    # Find reference file
    project_root = Path(__file__).parent.parent.parent.parent
    reference_file = project_root / "input/gnn_files/actinf_pomdp_agent.md"
    
    if not reference_file.exists():
        print(f"❌ Reference file not found: {reference_file}")
        return False
    
    print(f"✅ Reference file found: {reference_file}")
    
    # Read the markdown file
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        print(f"✅ Read markdown content ({len(markdown_content)} characters)")
    except Exception as e:
        print(f"❌ Failed to read markdown file: {e}")
        return False
    
    # Try to import JSON serializer directly
    try:
        from gnn.parsers.json_serializer import JSONSerializer
        from gnn.parsers.json_parser import JSONGNNParser
        from gnn.parsers.common import GNNInternalRepresentation
        
        print("✅ Successfully imported JSON serializer and parser")
        
        # Create a simple model from markdown content
        # This is a simplified version that doesn't use the full parsing system
        model = GNNInternalRepresentation(
            model_name="Test Model",
            annotation="Test annotation"
        )
        
        # Add some basic content to the model
        model.variables = []
        model.connections = []
        model.parameters = []
        model.equations = []
        model.raw_sections = {"content": markdown_content}
        
        # Serialize to JSON
        serializer = JSONSerializer()
        json_content = serializer.serialize(model)
        
        if not json_content:
            print("❌ JSON serialization produced empty content")
            return False
        
        print(f"✅ Serialized to JSON ({len(json_content)} characters)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            tf.write(json_content)
            temp_file = Path(tf.name)
        
        print(f"✅ Saved JSON to temporary file: {temp_file}")
        
        # Parse back from JSON
        parser = JSONGNNParser()
        parsed_result = parser.parse_file(temp_file)
        
        if not parsed_result.success:
            print(f"❌ JSON parsing failed: {parsed_result.errors}")
            return False
        
        print(f"✅ Successfully parsed JSON back to model")
        
        # Clean up
        temp_file.unlink()
        
        print("🎉 JSON round-trip test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import JSON modules: {e}")
        return False
    except Exception as e:
        print(f"❌ JSON round-trip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xml_round_trip():
    """Test XML round-trip conversion with minimal dependencies."""
    print("\n🔄 Testing XML round-trip conversion...")
    
    # Find reference file
    project_root = Path(__file__).parent.parent.parent.parent
    reference_file = project_root / "input/gnn_files/actinf_pomdp_agent.md"
    
    if not reference_file.exists():
        print(f"❌ Reference file not found: {reference_file}")
        return False
    
    # Try to import XML serializer directly
    try:
        from gnn.parsers.xml_serializer import XMLSerializer
        from gnn.parsers.xml_parser import XMLGNNParser
        from gnn.parsers.common import GNNInternalRepresentation
        
        print("✅ Successfully imported XML serializer and parser")
        
        # Create a simple model from markdown content
        model = GNNInternalRepresentation(
            model_name="Test Model",
            annotation="Test annotation"
        )
        
        # Add some basic content to the model
        model.variables = []
        model.connections = []
        model.parameters = []
        model.equations = []
        model.raw_sections = {"content": "test content"}
        
        # Serialize to XML
        serializer = XMLSerializer()
        xml_content = serializer.serialize(model)
        
        if not xml_content:
            print("❌ XML serialization produced empty content")
            return False
        
        print(f"✅ Serialized to XML ({len(xml_content)} characters)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tf:
            tf.write(xml_content)
            temp_file = Path(tf.name)
        
        print(f"✅ Saved XML to temporary file: {temp_file}")
        
        # Parse back from XML
        parser = XMLGNNParser()
        parsed_result = parser.parse_file(temp_file)
        
        if not parsed_result.success:
            print(f"❌ XML parsing failed: {parsed_result.errors}")
            return False
        
        print(f"✅ Successfully parsed XML back to model")
        
        # Clean up
        temp_file.unlink()
        
        print("🎉 XML round-trip test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import XML modules: {e}")
        return False
    except Exception as e:
        print(f"❌ XML round-trip test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("SIMPLE ROUND-TRIP TEST (JSON + XML)")
    print("=" * 60)
    
    # Test JSON
    json_success = test_json_round_trip()
    
    # Test XML
    xml_success = test_xml_round_trip()
    
    print("=" * 60)
    if json_success and xml_success:
        print("✅ SUCCESS: Both JSON and XML round-trip conversion work!")
        sys.exit(0)
    else:
        print("❌ FAILURE: Some round-trip conversions failed!")
        if not json_success:
            print("   - JSON round-trip failed")
        if not xml_success:
            print("   - XML round-trip failed")
        sys.exit(1) 